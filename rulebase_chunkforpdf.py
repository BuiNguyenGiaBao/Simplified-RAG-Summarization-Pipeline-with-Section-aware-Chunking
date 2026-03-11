import re
from typing import List, Tuple, Dict, Optional
import nltk


class ChunkConfig:

    MAX_HEADING_LENGTH = 120
    MAX_HEADING_WORDS = 12
    TITLE_CASE_THRESHOLD = 0.6

    MIN_SECTION_WORDS = 40

    MAX_CHUNK_WORDS = 220
    OVERLAP_WORDS = 40
    MIN_CHUNK_WORDS = 60

    MAX_CHUNK_CHARS = 2000


COMMON_SECTIONS = {
    "abstract","introduction","background","related work",
    "literature review","method","methods","methodology",
    "experiments","results","discussion","conclusion",
    "limitations","future work","references","appendix"
}


RE_NUMBERED = re.compile(
r"""^(
(\d+(\.\d+){0,3})
|([IVXLCDM]+)
)[\.\)\:]?\s+([A-Za-z].+)$""",
re.VERBOSE
)


def normalize_space(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def word_count(text: str) -> int:
    text = normalize_space(text)
    if not text:
        return 0
    return len(text.split())


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
        title = m.group(5)
        return 1 <= len(title.split()) <= ChunkConfig.MAX_HEADING_WORDS

    if line.isupper() and 1 <= len(line.split()) <= 10:
        return True

    low = line.lower()

    if low in COMMON_SECTIONS:
        return True

    words = line.split()

    if 1 <= len(words) <= 10:
        ratio = sum(w[0].isupper() for w in words if w) / max(len(words),1)

        if ratio >= ChunkConfig.TITLE_CASE_THRESHOLD:
            return True

    return False


def clean_heading(line: str) -> str:

    line = normalize_space(line)

    m = RE_NUMBERED.match(line)

    if m:
        return normalize_space(m.group(5))

    if line.isupper():
        return line.title()

    return line


def split_sentences(text: str) -> List[str]:

    if not text or not text.strip():
        return []

    try:
        sents = nltk.sent_tokenize(text)
        return [s.strip() for s in sents if s.strip()]
    except Exception:
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sents if s.strip()]


def rule_based_section_parse(text: str):

    lines = [normalize_space(x) for x in text.splitlines()]

    sections = []

    current_title = "Body"
    current_buf = []

    for line in lines:

        if not line:
            continue

        if looks_like_heading(line):

            if current_buf:
                sections.append((current_title," ".join(current_buf)))

            current_title = clean_heading(line)
            current_buf = []

        else:
            current_buf.append(line)

    if current_buf:
        sections.append((current_title," ".join(current_buf)))

    filtered = []

    for t,c in sections:

        if word_count(c) >= ChunkConfig.MIN_SECTION_WORDS:
            filtered.append((t,c))

    return filtered


def chunk_sections(
sections,
source_doc_id=None,
max_words=None,
overlap_words=None
):

    if max_words is None:
        max_words = ChunkConfig.MAX_CHUNK_WORDS

    if overlap_words is None:
        overlap_words = ChunkConfig.OVERLAP_WORDS

    chunks = []
    chunk_id = 0

    for sec_index,(section_title,content) in enumerate(sections):

        sents = split_sentences(content)

        current=[]
        current_words=0

        chunk_index_in_section = 0

        for s in sents:

            s_words = len(s.split())

            if current_words + s_words > max_words:

                text = normalize_space(" ".join(current))

                chunks.append({

                    "chunk_id":chunk_id,
                    "section":section_title,
                    "text":text,
                    "word_count":word_count(text),
                    "section_index":sec_index,
                    "chunk_index_in_section":chunk_index_in_section,
                    "source_doc_id":source_doc_id
                })

                chunk_id+=1
                chunk_index_in_section+=1

                current=[s]
                current_words=s_words

            else:

                current.append(s)
                current_words+=s_words

        if current:

            text = normalize_space(" ".join(current))

            chunks.append({

                "chunk_id":chunk_id,
                "section":section_title,
                "text":text,
                "word_count":word_count(text),
                "section_index":sec_index,
                "chunk_index_in_section":chunk_index_in_section,
                "source_doc_id":source_doc_id
            })

            chunk_id+=1

    return chunks


def process_document(text,source_doc_id=None):

    sections = rule_based_section_parse(text)

    chunks = chunk_sections(
        sections,
        source_doc_id=source_doc_id
    )

    return {
        "sections":sections,
        "chunks":chunks,
        "num_sections":len(sections),
        "num_chunks":len(chunks)
    }