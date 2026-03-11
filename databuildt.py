
import os
import json
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple
from datasets import load_from_disk, DatasetDict
from rulebase_chunkforpdf import process_document
from retrieval_tokenizer import DenseEncoder, MMRDenseRetriever, Document
from summarized import T5Summarizer


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ARXIV_DIR  = "./dataset/arxiv"

DEFAULT_QUERY_TEMPLATES = [
    "Summarize the main contribution of this paper",
    "Summarize the proposed method and main findings of this paper",
    "What problem does this paper address and how does it solve it?",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pubmed_txt(path: str) -> List[Dict[str, str]]:
    """
    Load a PubMed .txt file.
    Expected format per line:  <article>\\t<abstract>
    """
    records: List[Dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            article, abstract = parts[0].strip(), parts[1].strip()
            if article and abstract:
                records.append({"article": article, "abstract": abstract})
    return records


def load_arxiv_arrow(split_dir: str) -> List[Dict[str, str]]:
    split_info = os.path.join(split_dir, "dataset_info.json")
    if os.path.isfile(split_info):
        ds = load_from_disk(split_dir)
    else:
        split_name = os.path.basename(split_dir.rstrip("/\\"))
        root_dir = os.path.dirname(split_dir.rstrip("/\\"))
        dataset = load_from_disk(root_dir)
        if isinstance(dataset, DatasetDict):
            ds = dataset[split_name]
        else:
            ds = dataset

    records: List[Dict[str, str]] = []
    for sample in ds:
        article = _clean_field(sample.get("article", ""), join_with="\n")
        abstract = _clean_field(sample.get("abstract", ""), join_with=" ")
        if article and abstract:
            records.append({"article": article, "abstract": abstract})
    return records

def _clean_field(x: Any, join_with: str = "\n") -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return join_with.join(str(i).strip() for i in x if str(i).strip())
    return str(x).replace("/n", "\n").strip()



# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def make_documents_from_chunks(chunks: List[Dict[str, Any]]) -> List[Document]:
    return [
        Document(id=str(c["chunk_id"]), text=c["text"], metadata=c)
        for c in chunks
        if c.get("text", "").strip()
    ]


def choose_query(paper_id: str, use_multiple: bool = True) -> str:
    if not use_multiple:
        return DEFAULT_QUERY_TEMPLATES[0]
    return random.Random(paper_id).choice(DEFAULT_QUERY_TEMPLATES)



# Cross-document noise pool


def build_global_noise_pool(
    papers:      List[Dict[str, str]],
    limit:       int = 200,
    min_chunks:  int = 3,
) -> List[Document]:
    """
    Build a corpus-level pool of chunks from the first *limit* train papers.
    Used only for cross_document noise mode.
    """
    pool: List[Document] = []
    total = min(limit, len(papers))

    for idx in range(total):
        paper    = papers[idx]
        paper_id = f"noise_pool_{idx}"
        article  = paper.get("article", "")
        if not article:
            continue
        try:
            processed = process_document(article, source_doc_id=paper_id)
            chunks    = processed.get("chunks", [])
            if len(chunks) >= min_chunks:
                pool.extend(make_documents_from_chunks(chunks))
        except Exception as e:
            print(f"[WARN] noise_pool_{idx}: {e}")

        if (idx + 1) % 50 == 0:
            print(f"[noise_pool] {idx + 1}/{total} papers -> {len(pool)} chunks")

    print(f"[INFO] global noise pool: {len(pool)} chunks")
    return pool

# Noisy context builder
def make_noisy_context(
    documents:         List[Document],
    retrieved_items:   List[Any],
    final_k:           int = 3,
    noise_k:           int = 2,
    shuffle:           bool = True,
    rng:               Optional[random.Random] = None,
    noise_mode:        str = "same_document",
    global_noise_pool: Optional[List[Document]] = None,
    current_paper_id:  Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build clean and noisy context lists.

    Noise modes:
      same_document  — noise chunks come from non-retrieved chunks of the
                       same paper.
      cross_document — noise chunks come from the corpus-level noise pool
                       (other papers only).

    The noisy list = all final_k clean docs + noise_k noise docs
    (noise does NOT replace clean docs — it is appended).
    """
    if rng is None:
        rng = random.Random()

    clean_docs = [item.document for item in retrieved_items[:final_k]]
    clean_ids  = {doc.id for doc in clean_docs}

    # ---- build negative pool ----
    if noise_mode == "same_document":
        negative_pool = [doc for doc in documents if doc.id not in clean_ids]

    elif noise_mode == "cross_document":
        if not global_noise_pool:
            # Fallback: no pool available, return clean only
            contexts = [doc.text for doc in clean_docs]
            return {"clean_contexts": contexts,
                    "noisy_contexts": contexts,
                    "has_true_noise": False}

        negative_pool = [
            doc for doc in global_noise_pool
            if (doc.metadata or {}).get("source_doc_id") != current_paper_id
        ]

    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode!r}")

    actual_noise_k = min(noise_k, len(negative_pool))

    if actual_noise_k == 0:
        contexts = [doc.text for doc in clean_docs]
        return {"clean_contexts": contexts,
                "noisy_contexts": contexts,
                "has_true_noise": False}

    noise_docs  = rng.sample(negative_pool, actual_noise_k)
    noisy_docs  = list(clean_docs) + noise_docs   # append, never replace
    if shuffle:
        rng.shuffle(noisy_docs)

    return {
        "clean_contexts": [doc.text for doc in clean_docs],
        "noisy_contexts": [doc.text for doc in noisy_docs],
        "has_true_noise": True,
    }


# ---------------------------------------------------------------------------
# Per-paper example builder
# ---------------------------------------------------------------------------

def _process_one_paper(
    paper: Dict[str, str],
    paper_id: str,
    encoder: DenseEncoder,
    summarizer: T5Summarizer,
    split_name: str,
    final_k: int,
    noise_k: int,
    min_chunks: int,
    use_multiple_queries: bool,
    add_noisy: bool,
    shuffle_noisy: bool,
    noise_mode: str,
    global_noise_pool: Optional[List[Document]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Process one paper into training examples.

    Notes
    -----
    This function performs the full per-paper pipeline:

    1. Validate article/abstract
    2. Chunk the article into passages
    3. Convert chunks into Document objects
    4. Build a per-paper retriever index
    5. Retrieve top-k clean contexts
    6. Build one clean example
    7. Optionally inject retrieval noise and build one noisy example

    Important design choice:
    A paper is skipped only if it has zero chunks/documents.
    We do NOT require at least `min_chunks` documents anymore,
    because many arXiv papers collapse into a single large chunk
    after rule-based parsing. Keeping such papers improves dataset yield.
    """
    article = paper.get("article", "")
    abstract = paper.get("abstract", "")

    if not article or not abstract:
        print(f"[SKIP] {paper_id}: empty article/abstract")
        return []

    # Step 1: chunk the paper
    processed = process_document(article, source_doc_id=paper_id)
    chunks = processed.get("chunks", [])

    if len(chunks) == 0:
        print(f"[SKIP] {paper_id}: zero chunks")
        return []

    # Step 2: convert chunks to retriever documents
    documents = make_documents_from_chunks(chunks)

    if len(documents) == 0:
        print(f"[SKIP] {paper_id}: zero documents")
        return []

    # Step 3: build a per-paper retriever
    retriever = MMRDenseRetriever(encoder=encoder)
    retriever.build_index(documents)

    # Step 4: choose a pseudo-query for training
    query = choose_query(paper_id, use_multiple_queries)

    # Step 5: retrieve clean contexts
    retrieved = retriever.search(query, k=final_k)

    if not retrieved:
        print(f"[SKIP] {paper_id}: retrieval returned 0")
        return []

    # Step 6: build clean example
    clean_contexts = [r.document.text for r in retrieved]

    clean_ex = summarizer.build_training_example(
        query=query,
        contexts=clean_contexts,
        target_text=abstract,
        max_contexts=3,
    )
    clean_ex.update({
        "paper_id": paper_id,
        "split": split_name,
        "sample_type": "clean",
        "num_contexts": len(clean_contexts),
        "num_chunks": len(chunks),
        "num_documents": len(documents),
    })

    records = [clean_ex]

    # Step 7: optionally build noisy example
    if add_noisy:
        bundle = make_noisy_context(
            documents=documents,
            retrieved_items=retrieved,
            final_k=final_k,
            noise_k=noise_k,
            shuffle=shuffle_noisy,
            rng=rng,
            noise_mode=noise_mode,
            global_noise_pool=global_noise_pool,
            current_paper_id=paper_id,
        )

        if bundle["has_true_noise"]:
            noisy_ex = summarizer.build_training_example(
                query=query,
                contexts=bundle["noisy_contexts"],
                target_text=abstract,
                max_contexts=3,
            )
            noisy_ex.update({
                "paper_id": paper_id,
                "split": split_name,
                "sample_type": "noisy",
                "num_contexts": len(bundle["noisy_contexts"]),
                "noise_mode": noise_mode,
                "num_chunks": len(chunks),
                "num_documents": len(documents),
            })
            records.append(noisy_ex)

    return records


# ---------------------------------------------------------------------------
# Split iterators
# ---------------------------------------------------------------------------

def build_split(
    papers:            List[Dict[str, str]],
    split_name:        str,
    encoder:           DenseEncoder,
    summarizer:        T5Summarizer,
    limit:             Optional[int],
    final_k:           int,
    noise_k:           int,
    min_chunks:        int,
    use_multiple_queries: bool,
    add_noisy:         bool,
    shuffle_noisy:     bool,
    noise_mode:        str,
    global_noise_pool: Optional[List[Document]],
    rng:               random.Random,
) -> List[Dict[str, Any]]:
    """Build all examples for one split (train or valid)."""
    total   = len(papers) if limit is None else min(limit, len(papers))
    records: List[Dict[str, Any]] = []

    for idx in range(total):
        paper_id = f"{split_name}_{idx}"
        try:
            examples = _process_one_paper(
                paper=papers[idx],
                paper_id=paper_id,
                encoder=encoder,
                summarizer=summarizer,
                split_name=split_name,
                final_k=final_k,
                noise_k=noise_k if add_noisy else 0,
                min_chunks=min_chunks,
                use_multiple_queries=use_multiple_queries,
                add_noisy=add_noisy,
                shuffle_noisy=shuffle_noisy,
                noise_mode=noise_mode,
                global_noise_pool=global_noise_pool,
                rng=rng,
            )
            records.extend(examples)
        except Exception as e:
            print(f"[WARN] skip {paper_id}: {e}")

        if (idx + 1) % 50 == 0:
            print(f"[{split_name}] {idx + 1}/{total} -> {len(records)} records")

    return records


def build_test_split(
    papers:            List[Dict[str, str]],
    encoder:           DenseEncoder,
    summarizer:        T5Summarizer,
    limit:             Optional[int],
    final_k:           int,
    noise_k:           int,
    min_chunks:        int,
    use_multiple_queries: bool,
    shuffle_noisy:     bool,
    noise_mode:        str,
    global_noise_pool: Optional[List[Document]],
    rng:               random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build test examples and return them as two separate lists:
      (test_clean, test_noisy)
    so TRAIN.py can evaluate robustness separately.
    """
    total        = len(papers) if limit is None else min(limit, len(papers))
    clean_records: List[Dict[str, Any]] = []
    noisy_records: List[Dict[str, Any]] = []

    for idx in range(total):
        paper_id = f"test_{idx}"
        try:
            examples = _process_one_paper(
                paper=papers[idx],
                paper_id=paper_id,
                encoder=encoder,
                summarizer=summarizer,
                split_name="test",
                final_k=final_k,
                noise_k=noise_k,
                min_chunks=min_chunks,
                use_multiple_queries=use_multiple_queries,
                add_noisy=True,
                shuffle_noisy=shuffle_noisy,
                noise_mode=noise_mode,
                global_noise_pool=global_noise_pool,
                rng=rng,
            )
            for ex in examples:
                if ex["sample_type"] == "clean":
                    clean_records.append(ex)
                else:
                    noisy_records.append(ex)
        except Exception as e:
            print(f"[WARN] skip test_{idx}: {e}")

        if (idx + 1) % 50 == 0:
            print(
                f"[test] {idx + 1}/{total} "
                f"-> clean={len(clean_records)}, noisy={len(noisy_records)}"
            )

    return clean_records, noisy_records


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RAG training data from  ArXiv local dataset."
    )

    # Dataset source
# NOTE:
# This current experiment setup is restricted to the local arXiv dataset.
# PubMed loading code is kept for future extension, but the CLI currently
# exposes only "arxiv" as an allowed source.
    parser.add_argument("--source", type=str, default="arxiv", choices=["arxiv"])
    parser.add_argument("--arxiv_dir",  type=str, default=ARXIV_DIR)
    parser.add_argument("--output_dir", type=str, default="./prepared_data")

    # Size limits
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--valid_limit", type=int, default=None)
    parser.add_argument("--test_limit",  type=int, default=None)

    # Retrieval
    parser.add_argument("--final_k",    type=int, default=5)
    parser.add_argument("--min_chunks", type=int, default=3)
    parser.add_argument("--single_query", action="store_true")

    # Noise
    parser.add_argument(
        "--noise_mode", type=str, default="same_document",
        choices=["same_document", "cross_document"],
        help=(
            "same_document: noise from non-retrieved chunks of the same paper. "
            "cross_document: noise from other papers in the corpus."
        ),
    )
    parser.add_argument("--noise_k", type=int, default=2,
                        help="Number of noise chunks to inject per noisy example.")
    parser.add_argument("--noise_pool_limit", type=int, default=200,
                        help="Papers used to build cross-document noise pool.")
    parser.add_argument("--no_shuffle_noisy", action="store_true",
                        help="Keep noise docs at the end instead of shuffling.")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load raw papers
    # ------------------------------------------------------------------
    if args.source == "pubmed":
        print("Loading PubMed txt dataset...")
        train_papers = load_pubmed_txt(os.path.join(args.pubmed_dir, "train.txt"))
        valid_papers = load_pubmed_txt(os.path.join(args.pubmed_dir, "val.txt"))
        test_papers  = load_pubmed_txt(os.path.join(args.pubmed_dir, "test.txt"))
    else:
        print("Loading ArXiv Arrow dataset...")
        train_papers = load_arxiv_arrow(os.path.join(args.arxiv_dir, "train"))
        valid_papers = load_arxiv_arrow(os.path.join(args.arxiv_dir, "validation"))
        test_papers  = load_arxiv_arrow(os.path.join(args.arxiv_dir, "test"))

    print(
        f"Loaded  train={len(train_papers):,}  "
        f"valid={len(valid_papers):,}  "
        f"test={len(test_papers):,}"
    )

    # ------------------------------------------------------------------
    # Cross-document noise pool (built once from train papers)
    # ------------------------------------------------------------------
    global_noise_pool: Optional[List[Document]] = None
    if args.noise_mode == "cross_document":
        print("\nBuilding cross-document noise pool...")
        global_noise_pool = build_global_noise_pool(
            papers=train_papers,
            limit=args.noise_pool_limit,
            min_chunks=args.min_chunks,
        )

    # ------------------------------------------------------------------
    # Shared models (loaded once)
    # ------------------------------------------------------------------
    print("\nLoading encoder and summarizer...")
    encoder    = DenseEncoder()
    summarizer = T5Summarizer()

    use_multiple_queries = not args.single_query
    shuffle_noisy        = not args.no_shuffle_noisy

    # ------------------------------------------------------------------
    # Build splits
    # ------------------------------------------------------------------
    print("\nBuilding TRAIN records (clean + noisy)...")
    train_records = build_split(
        papers=train_papers, split_name="train",
        encoder=encoder, summarizer=summarizer,
        limit=args.train_limit, final_k=args.final_k, noise_k=args.noise_k,
        min_chunks=args.min_chunks, use_multiple_queries=use_multiple_queries,
        add_noisy=True, shuffle_noisy=shuffle_noisy,
        noise_mode=args.noise_mode, global_noise_pool=global_noise_pool, rng=rng,
    )

    print("\nBuilding VALID records (clean only)...")
    valid_records = build_split(
        papers=valid_papers, split_name="validation",
        encoder=encoder, summarizer=summarizer,
        limit=args.valid_limit, final_k=args.final_k, noise_k=0,
        min_chunks=args.min_chunks, use_multiple_queries=use_multiple_queries,
        add_noisy=False, shuffle_noisy=False,
        noise_mode=args.noise_mode, global_noise_pool=None, rng=rng,
    )

    print("\nBuilding TEST records (clean + noisy, saved separately)...")
    test_clean, test_noisy = build_test_split(
        papers=test_papers,
        encoder=encoder, summarizer=summarizer,
        limit=args.test_limit, final_k=args.final_k, noise_k=args.noise_k,
        min_chunks=args.min_chunks, use_multiple_queries=use_multiple_queries,
        shuffle_noisy=shuffle_noisy,
        noise_mode=args.noise_mode, global_noise_pool=global_noise_pool, rng=rng,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    train_path      = os.path.join(args.output_dir, "train.jsonl")
    valid_path      = os.path.join(args.output_dir, "valid.jsonl")
    test_clean_path = os.path.join(args.output_dir, "test_clean.jsonl")
    test_noisy_path = os.path.join(args.output_dir, "test_noisy.jsonl")

    write_jsonl(train_path,      train_records)
    write_jsonl(valid_path,      valid_records)
    write_jsonl(test_clean_path, test_clean)
    write_jsonl(test_noisy_path, test_noisy)

    print("\nDone.")
    print(f"  train       : {len(train_records):>6,}  ->  {train_path}")
    print(f"  valid       : {len(valid_records):>6,}  ->  {valid_path}")
    print(f"  test clean  : {len(test_clean):>6,}  ->  {test_clean_path}")
    print(f"  test noisy  : {len(test_noisy):>6,}  ->  {test_noisy_path}")


if __name__ == "__main__":
    main()