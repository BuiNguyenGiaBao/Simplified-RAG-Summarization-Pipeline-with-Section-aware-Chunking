import re
from typing import List, Tuple, Dict, Optional
import nltk



class ChunkConfig:
    """Configuration for chunking parameters"""
    MAX_HEADING_LENGTH = 120
    MIN_SECTION_WORDS = 20
    MAX_CHUNK_WORDS = 300
    OVERLAP_WORDS = 40
    MIN_CHUNK_WORDS = 50
    MAX_HEADING_WORDS = 12
    TITLE_CASE_THRESHOLD = 0.6


COMMON_SECTIONS = {
    "abstract", "introduction", "background", "related work", "literature review",
    "method", "methods", "methodology", "materials and methods",
    "experiments", "experimental setup", "results", "discussion",
    "conclusion", "conclusions", "limitations", "future work",
    "acknowledgments", "acknowledgements", "references", "appendix", 
    "supplementary material", "supplementary materials"
}

# Pattern for numbered sections (e.g., "1.2.3 Title" or "I. Title")
RE_NUMBERED = re.compile(
    r"""^(
        (\d+(\.\d+){0,3})          # 1 or 1.2 or 1.2.3 or 1.2.3.4
        |([IVXLCDM]+)              # Roman numerals: I, II, III, IV, V, etc.
    )[\.\)\:]?\s+([A-Za-z].+)$""",  # Added colon support
    re.VERBOSE
)


def normalize_space(s: str) -> str:
    """Normalize whitespace in a string"""
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def looks_like_heading(line: str) -> bool:
    """
    Determine if a line looks like a section heading based on multiple heuristics.
    Args:
        line: Text line to check
        
    Returns:
        True if line appears to be a heading
    """
    line = normalize_space(line)
    if not line:
        return False

    # Too long to be a heading
    if len(line) > ChunkConfig.MAX_HEADING_LENGTH:
        return False

    # Sentences ending with period (unless very short)
    if line.endswith(".") and len(line.split()) > 3:
        return False

    # (A) Check for numbered patterns (e.g., "1.2 Introduction")
    m = RE_NUMBERED.match(line)
    if m:
        title = m.group(5).strip()
        return 1 <= len(title.split()) <= ChunkConfig.MAX_HEADING_WORDS

    # (B) Short all-caps text (e.g., "INTRODUCTION")
    if line.isupper() and 1 <= len(line.split()) <= 10:
        return True

    # (C) Common section keywords (case-insensitive)
    low = line.lower()
    if low in COMMON_SECTIONS:
        return True

    # (D) Title case detection (e.g., "Literature Review")
    words = line.split()
    if 1 <= len(words) <= 10:
        title_case_ratio = sum(w[0].isupper() for w in words if w) / max(len(words), 1)
        if title_case_ratio >= ChunkConfig.TITLE_CASE_THRESHOLD:
            return True

    return False


def clean_heading(line: str) -> str:
    """
    Clean a heading by removing numbering and normalizing format.
    
    Args:
        line: Raw heading text
        
    Returns:
        Cleaned heading text
    """
    line = normalize_space(line)

    # Remove numbering prefix if present
    m = RE_NUMBERED.match(line)
    if m:
        return normalize_space(m.group(5))

    # Convert all-caps to title case
    return line.title() if line.isupper() else line


def split_sentences_advanced(text: str) -> List[str]:
    """
    Split text into sentences using NLTK with fallback to regex.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    try:
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        # Split on period/question/exclamation followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]


def rule_based_section_parse(text: str) -> List[Tuple[str, str]]:
    """
    Parse text into sections based on heading detection rules.
    
    Args:
        text: Raw text to parse
        
    Returns:
        List of (section_title, section_content) tuples
    """
    if not text or not text.strip():
        return []
    
    try:
        lines = [normalize_space(x) for x in text.splitlines()]
        sections: List[Tuple[str, List[str]]] = []

        current_title = "Body"
        current_buf: List[str] = []

        for line in lines:
            if not line:
                continue

            if looks_like_heading(line):
                # Save previous section if it has content
                if current_buf:
                    sections.append((current_title, " ".join(current_buf).strip()))
                    current_buf = []

                current_title = clean_heading(line)
            else:
                current_buf.append(line)

        # Don't forget the last section
        if current_buf:
            sections.append((current_title, " ".join(current_buf).strip()))
        
        # Filter out sections that are too short (likely noise)
        sections = [(t, c) for t, c in sections 
                   if len(c.split()) >= ChunkConfig.MIN_SECTION_WORDS]

        return sections
        
    except Exception as e:
        print(f"Error parsing sections: {e}")
        return []


def chunk_sections(
    sections: List[Tuple[str, str]],
    max_words: Optional[int] = None,
    overlap_words: Optional[int] = None,
    use_advanced_splitting: bool = True
) -> List[Dict[str, any]]:
    """
    Split sections into smaller chunks with optional overlap.
    
    Args:
        sections: List of (section_title, section_content) tuples
        max_words: Maximum words per chunk (default from config)
        overlap_words: Words to overlap between chunks (default from config)
        use_advanced_splitting: Use NLTK for sentence splitting if available
        
    Returns:
        List of chunk dictionaries with chunk_id, section, and text
    """
    if max_words is None:
        max_words = ChunkConfig.MAX_CHUNK_WORDS
    if overlap_words is None:
        overlap_words = ChunkConfig.OVERLAP_WORDS
    
    chunks = []
    chunk_id = 0

    for paper_section, content in sections:
        # Skip references section
        if paper_section.lower() in ["references", "bibliography"]:
            continue

        # Split into sentences
        if use_advanced_splitting:
            sentences = split_sentences_advanced(content)
        else:
            sentences = re.split(r'(?<=[\.\?\!])\s+', content)
            sentences = [s.strip() for s in sentences if s.strip()]

        current_chunk = []
        current_word_count = 0

        for sent in sentences:
            words = sent.split()
            w_len = len(words)

            if current_word_count + w_len > max_words and current_chunk:
                # Flush current chunk
                chunk_text = " ".join(current_chunk).strip()
                if len(chunk_text.split()) >= ChunkConfig.MIN_CHUNK_WORDS:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "section": paper_section,
                        "text": chunk_text,
                        "word_count": len(chunk_text.split())
                    })
                    chunk_id += 1

                # Create overlap from previous chunk
                overlap = []
                if overlap_words > 0 and chunk_text:
                    flat_words = chunk_text.split()
                    overlap = flat_words[-overlap_words:]

                current_chunk = [" ".join(overlap), sent] if overlap else [sent]
                current_word_count = len(overlap) + w_len
            else:
                current_chunk.append(sent)
                current_word_count += w_len

        # Flush remaining content in section
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if len(chunk_text.split()) >= ChunkConfig.MIN_CHUNK_WORDS:
                chunks.append({
                    "chunk_id": chunk_id,
                    "section": paper_section,
                    "text": chunk_text,
                    "word_count": len(chunk_text.split())
                })
                chunk_id += 1

    return chunks


def process_document(
    text: str,
    max_words: Optional[int] = None,
    overlap_words: Optional[int] = None
) -> Dict[str, any]:
    """
    Complete pipeline: parse sections and create chunks.
    
    Args:
        text: Raw document text
        max_words: Maximum words per chunk
        overlap_words: Words to overlap between chunks
        
    Returns:
        Dictionary with sections and chunks
    """
    sections = rule_based_section_parse(text)
    chunks = chunk_sections(
        sections, 
        max_words=max_words,
        overlap_words=overlap_words
    )
    
    return {
        "sections": sections,
        "chunks": chunks,
        "num_sections": len(sections),
        "num_chunks": len(chunks)
    }


