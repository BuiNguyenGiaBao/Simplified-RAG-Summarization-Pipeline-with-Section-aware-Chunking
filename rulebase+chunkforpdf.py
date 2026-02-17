import re
from typing import List, Tuple, Dict, Optional, Callable
import nltk


_TOKENIZER_BACKEND: str = "none"


def _try_tiktoken() -> Optional[Callable]:
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")

        def _count(text: str) -> int:
            return len(enc.encode(text))

        def _encode(text: str) -> List[int]:
            return enc.encode(text)

        return _count, _encode, 
    except Exception:
        return None


def _try_transformers() -> Optional[Callable]:
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")

        def _count(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))

        def _encode(text: str) -> List[int]:
            return tok.encode(text, add_special_tokens=False)

        return _count, _encode, "transformers (bert-base-uncased)"
    except Exception:
        return None


def _regex_tokenizer() -> Tuple[Callable, Callable, str]:
    _SPLITTER = re.compile(
        r"""
        \w+(?:'\w+)*   # words (with optional contractions: don't, it's …)
        |[^\w\s]       # any single non-word, non-space character (punctuation)
        """,
        re.VERBOSE,
    )

    def _count(text: str) -> int:
        return len(_SPLITTER.findall(text))

    def _encode(text: str) -> List[int]:
        # Returns simple character-sum based IDs — good enough for counting
        tokens = _SPLITTER.findall(text)
        return [sum(ord(c) for c in t) for t in tokens]

    return _count, _encode, "regex (built-in fallback)"


def _init_tokenizer():
    global _count_tokens_fn, _encode_fn, _TOKENIZER_BACKEND
    result = _try_tiktoken() or _try_transformers()
    if result:
        _count_tokens_fn, _encode_fn, _TOKENIZER_BACKEND = result
    else:
        _count_tokens_fn, _encode_fn, _TOKENIZER_BACKEND = _regex_tokenizer()
    print(f"[Tokenizer] Using backend: {_TOKENIZER_BACKEND}")


_count_tokens_fn: Callable[[str], int]
_encode_fn: Callable[[str], List[int]]
_init_tokenizer()


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return _count_tokens_fn(text)


def encode_text(text: str) -> List[int]:
    if not text:
        return []
    return _encode_fn(text)


def get_tokenizer_info() -> str:
    return _TOKENIZER_BACKEND


class ChunkConfig:
    MAX_HEADING_LENGTH   = 120   # characters
    MIN_SECTION_TOKENS   = 30    # was MIN_SECTION_WORDS = 20
    MAX_CHUNK_TOKENS     = 400   # was MAX_CHUNK_WORDS = 300
    OVERLAP_TOKENS       = 60    # was OVERLAP_WORDS = 40
    MIN_CHUNK_TOKENS     = 60    # was MIN_CHUNK_WORDS = 50
    MAX_HEADING_WORDS    = 12
    TITLE_CASE_THRESHOLD = 0.6


COMMON_SECTIONS = {
    "abstract", "introduction", "background", "related work", "literature review",
    "method", "methods", "methodology", "materials and methods",
    "experiments", "experimental setup", "results", "discussion",
    "conclusion", "conclusions", "limitations", "future work",
    "acknowledgments", "acknowledgements", "references", "appendix",
    "supplementary material", "supplementary materials"
}

RE_NUMBERED = re.compile(
    r"""^(
        (\d+(\.\d+){0,3})
        |([IVXLCDM]+)
    )[\.\)\:]?\s+([A-Za-z].+)$""",
    re.VERBOSE,
)

def normalize_space(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def looks_like_heading(line: str) -> bool:
    line = normalize_space(line)
    if not line:
        return False
    if len(line) > ChunkConfig.MAX_HEADING_LENGTH:
        return False
    if line.endswith(".") and len(line.split()) > 3:
        return False
    m = RE_NUMBERED.match(line)
    if m:
        title = m.group(5).strip()
        return 1 <= len(title.split()) <= ChunkConfig.MAX_HEADING_WORDS
    if line.isupper() and 1 <= len(line.split()) <= 10:
        return True
    low = line.lower()
    if low in COMMON_SECTIONS:
        return True
    words = line.split()
    if 1 <= len(words) <= 10:
        ratio = sum(w[0].isupper() for w in words if w) / max(len(words), 1)
        if ratio >= ChunkConfig.TITLE_CASE_THRESHOLD:
            return True
    return False


def clean_heading(line: str) -> str:
    line = normalize_space(line)
    m = RE_NUMBERED.match(line)
    if m:
        return normalize_space(m.group(5))
    return line.title() if line.isupper() else line




def split_sentences_advanced(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    try:
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]


def rule_based_section_parse(text: str) -> List[Tuple[str, str]]:
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
                if current_buf:
                    sections.append((current_title, " ".join(current_buf).strip()))
                    current_buf = []
                current_title = clean_heading(line)
            else:
                current_buf.append(line)

        if current_buf:
            sections.append((current_title, " ".join(current_buf).strip()))

        # Filter short sections by token count (more accurate than word count)
        sections = [
            (t, c) for t, c in sections
            if count_tokens(c) >= ChunkConfig.MIN_SECTION_TOKENS
        ]
        return sections

    except Exception as e:
        print(f"Error parsing sections: {e}")
        return []

def chunk_sections(
    sections: List[Tuple[str, str]],
    max_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
    use_advanced_splitting: bool = True,
) -> List[Dict]:
    if max_tokens is None:
        max_tokens = ChunkConfig.MAX_CHUNK_TOKENS
    if overlap_tokens is None:
        overlap_tokens = ChunkConfig.OVERLAP_TOKENS

    chunks: List[Dict] = []
    chunk_id = 0

    for section_title, content in sections:
        if section_title.lower() in {"references", "bibliography"}:
            continue
        if use_advanced_splitting:
            sentences = split_sentences_advanced(content)
        else:
            sentences = re.split(r'(?<=[\.\?\!])\s+', content)
            sentences = [s.strip() for s in sentences if s.strip()]

        # Pre-compute token counts for each sentence (avoids repeated calls)
        sent_tokens: List[int] = [count_tokens(s) for s in sentences]

        current_sents: List[str] = []
        current_token_count: int = 0

        def _flush(sents: List[str]) -> Optional[Dict]:
            nonlocal chunk_id
            text = " ".join(sents).strip()
            tc = count_tokens(text)
            if tc >= ChunkConfig.MIN_CHUNK_TOKENS:
                chunk = {
                    "chunk_id": chunk_id,
                    "section": section_title,
                    "text": text,
                    "token_count": tc,
                }
                chunk_id += 1
                return chunk
            return None

        def _build_overlap(sents: List[str]) -> Tuple[List[str], int]:
            if overlap_tokens <= 0 or not sents:
                return [], 0
            overlap_sents: List[str] = []
            acc = 0
            for s in reversed(sents):
                tc = count_tokens(s)
                if acc + tc > overlap_tokens and overlap_sents:
                    break
                overlap_sents.insert(0, s)
                acc += tc
            return overlap_sents, acc

        for sent, stc in zip(sentences, sent_tokens):
            # Edge case: single sentence longer than the whole window
            if stc >= max_tokens:
                # Flush any pending content first
                if current_sents:
                    c = _flush(current_sents)
                    if c:
                        chunks.append(c)
                    current_sents, current_token_count = [], 0

                # Emit the oversized sentence as its own chunk (truncation note)
                long_text = sent.strip()
                chunks.append({
                    "chunk_id": chunk_id,
                    "section": section_title,
                    "text": long_text,
                    "token_count": stc,
                    "oversized": True,   # flag for downstream handling
                })
                chunk_id += 1
                continue

            if current_token_count + stc > max_tokens and current_sents:
                c = _flush(current_sents)
                if c:
                    chunks.append(c)
                overlap_sents, overlap_tc = _build_overlap(current_sents)
                current_sents = overlap_sents + [sent]
                current_token_count = overlap_tc + stc
            else:
                current_sents.append(sent)
                current_token_count += stc

        # Flush remainder
        if current_sents:
            c = _flush(current_sents)
            if c:
                chunks.append(c)

    return chunks




def process_document(
    text: str,
    max_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
) -> Dict:
    sections = rule_based_section_parse(text)
    chunks = chunk_sections(
        sections,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    return {
        "tokenizer":    get_tokenizer_info(),
        "sections":     sections,
        "chunks":       chunks,
        "num_sections": len(sections),
        "num_chunks":   len(chunks),
    }