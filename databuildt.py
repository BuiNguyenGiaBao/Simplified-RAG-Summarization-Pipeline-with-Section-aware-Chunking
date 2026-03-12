"""
databuildt.py  —  Fast 3-stage pipeline
========================================

Stage 1  [CPU parallel]   chunk every paper with ProcessPoolExecutor
Stage 2  [GPU batch]      encode ALL chunk texts in one big batched call
Stage 3  [CPU sequential] FAISS retrieval + noise injection per paper
                          (fast because embeddings are already computed)

Typical speed-up over the original sequential pipeline:
  - 4× CPU workers  →  chunking ~4× faster
  - batch encoding  →  GPU utilisation near 100 %  (~5-10× faster than per-paper)
  - pre-built index →  no repeated tokenizer calls in build_training_example
"""

import os
import json
import random
import argparse
import time
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import faiss
from datasets import load_from_disk, DatasetDict

from rulebase_chunkforpdf import process_document
from retrieval_tokenizer import DenseEncoder, MMRDenseRetriever, Document
from summarized import T5Summarizer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):  # type: ignore
        return x


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ARXIV_DIR = "./dataset/arxiv"

DEFAULT_QUERY_TEMPLATES = [
    "Summarize the main contribution of this paper",
    "Summarize the proposed method and main findings of this paper",
    "What problem does this paper address and how does it solve it?",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _clean_field(x: Any, join_with: str = "\n") -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return join_with.join(str(i).strip() for i in x if str(i).strip())
    return str(x).replace("/n", "\n").strip()


def load_pubmed_txt(path: str) -> List[Dict[str, str]]:
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
        root_dir   = os.path.dirname(split_dir.rstrip("/\\"))
        dataset    = load_from_disk(root_dir)
        from datasets import DatasetDict
        ds = dataset[split_name] if isinstance(dataset, DatasetDict) else dataset

    records: List[Dict[str, str]] = []
    for sample in ds:
        article  = _clean_field(sample.get("article",  ""), join_with="\n")
        abstract = _clean_field(sample.get("abstract", ""), join_with=" ")
        if article and abstract:
            records.append({"article": article, "abstract": abstract})
    return records


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def make_documents_from_chunks(chunks: List[Dict[str, Any]]) -> List[Document]:
    return [
        Document(
            # BUG FIX: globally unique id = source_doc_id + chunk_id to avoid cross-paper collisions
            id=f"{c.get('source_doc_id', 'unk')}_{c['chunk_id']}",
            text=c["text"],
            metadata=c,
        )
        for c in chunks
        if c.get("text", "").strip()
    ]


def choose_query(paper_id: str, use_multiple: bool = True) -> str:
    if not use_multiple:
        return DEFAULT_QUERY_TEMPLATES[0]
    return random.Random(paper_id).choice(DEFAULT_QUERY_TEMPLATES)


# ---------------------------------------------------------------------------
# Stage 1 — CPU-parallel chunking
# ---------------------------------------------------------------------------

def _chunk_one_paper(task: Tuple[int, Dict[str, str], str]) -> Optional[Dict[str, Any]]:
    """Top-level worker (must be picklable → module-level function)."""
    idx, paper, split_name = task
    article  = paper.get("article",  "")
    abstract = paper.get("abstract", "")
    if not article or not abstract:
        return None
    try:
        processed = process_document(article, source_doc_id=f"{split_name}_{idx}")
        chunks = processed.get("chunks", [])
        if not chunks:
            return None
        return {
            "paper_id": f"{split_name}_{idx}",
            "abstract": abstract,
            "chunks":   chunks,
        }
    except Exception:
        return None


def batch_chunk_papers(
    papers:      List[Dict[str, str]],
    split_name:  str,
    num_workers: int = 4,
    limit:       Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Chunk all papers in parallel using threads.
    ThreadPoolExecutor (not Process) avoids Windows spawn overhead
    where each process would re-import torch + transformers (~30-60s).
    Chunking is I/O + regex heavy → threads work well here.
    """
    papers = papers[:limit] if limit is not None else papers
    tasks  = [(idx, p, split_name) for idx, p in enumerate(papers)]

    print(f"  [chunk:{split_name}] {len(tasks)} papers  |  workers={num_workers}")
    t0 = time.time()

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_chunk_one_paper, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"  chunk [{split_name}]", unit="paper"):
            res = fut.result()
            if res is not None:
                results.append(res)

    results.sort(key=lambda r: int(r["paper_id"].split("_")[-1]))
    print(f"  [chunk:{split_name}] {len(results)}/{len(tasks)} valid  "
          f"| {time.time()-t0:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Stage 2 — GPU batch encoding  (VRAM-safe for 6GB cards)
# ---------------------------------------------------------------------------

def batch_encode_all_chunks(
    chunked_papers:  List[Dict[str, Any]],
    encoder:         DenseEncoder,
    paper_batch:     int = 500,   # encode this many papers at a time
                                  # 500 papers × ~15 chunks × 384-dim ≈ 400MB RAM
) -> Dict[str, Dict[str, Any]]:
    """
    Encode chunks in groups of `paper_batch` papers to keep
    CPU RAM and GPU VRAM under control on 6 GB cards.

    Returns:
        { paper_id: {"chunks": [...], "embeddings": np.ndarray, "abstract": str} }
    """
    result: Dict[str, Dict[str, Any]] = {}
    total  = len(chunked_papers)

    for batch_start in range(0, total, paper_batch):
        group = chunked_papers[batch_start : batch_start + paper_batch]

        # Flatten texts for this group
        all_texts:    List[str]                = []
        paper_slices: Dict[str, Tuple[int,int]] = {}

        for cp in group:
            pid   = cp["paper_id"]
            texts = [c["text"] for c in cp["chunks"]]
            s     = len(all_texts)
            all_texts.extend(texts)
            paper_slices[pid] = (s, s + len(texts))

        batch_end = min(batch_start + paper_batch, total)
        print(f"  [encode] papers {batch_start+1}-{batch_end}/{total} "
              f"| {len(all_texts)} chunks", flush=True)

        all_embs = encoder.encode(all_texts)   # GPU batched

        # Redistribute embeddings back
        for cp in group:
            pid  = cp["paper_id"]
            s, e = paper_slices[pid]
            result[pid] = {
                "chunks":     cp["chunks"],
                "embeddings": all_embs[s:e],
                "abstract":   cp["abstract"],
            }

        # Free intermediate arrays explicitly
        del all_texts, all_embs, paper_slices

    return result


# ---------------------------------------------------------------------------
# Stage 3 helpers — per-paper retrieval using pre-built embeddings
# ---------------------------------------------------------------------------

def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def _mmr_select(
    q_emb:       np.ndarray,     # (1, D)
    cand_embs:   np.ndarray,     # (C, D)
    cand_scores: List[float],
    k:           int,
    mmr_lambda:  float = 0.5,
) -> List[int]:
    """Pure-numpy MMR — no encoder call needed."""
    sim_matrix = cand_embs @ cand_embs.T
    selected:  List[int] = []
    remaining: set       = set(range(len(cand_scores)))

    for _ in range(min(k, len(cand_scores))):
        best_pos, best_score = -1, float("-inf")
        for pos in remaining:
            rel = cand_scores[pos]
            red = float(np.max(sim_matrix[pos, selected])) if selected else 0.0
            score = mmr_lambda * rel - (1 - mmr_lambda) * red
            if score > best_score:
                best_score, best_pos = score, pos
        if best_pos == -1:
            break
        selected.append(best_pos)
        remaining.remove(best_pos)
    return selected


def retrieve_clean(
    query_emb:   np.ndarray,
    documents:   List[Document],
    doc_embs:    np.ndarray,
    index:       faiss.IndexFlatIP,
    final_k:     int,
    noise_k:     int,
    mmr_lambda:  float = 0.5,
) -> Tuple[List[Document], List[int], np.ndarray]:
    """
    Fast retrieval using pre-built FAISS index + numpy MMR.
    Returns (retrieved_docs, retrieved_indices, all_scores).
    all_scores: cosine similarity of ALL docs against the query — used for easy/hard noise selection.
    """
    num_docs    = len(documents)
    effective_k = min(final_k, max(1, num_docs - noise_k))
    candidate_k = min(max(effective_k * 5, 50), num_docs)

    scores, idxs = index.search(query_emb, candidate_k)
    cand_indices = idxs[0].tolist()
    cand_scores  = scores[0].tolist()
    cand_embs    = doc_embs[cand_indices]

    selected_pos = _mmr_select(query_emb, cand_embs, cand_scores,
                               k=effective_k, mmr_lambda=mmr_lambda)

    retrieved_docs    = [documents[cand_indices[p]] for p in selected_pos]
    retrieved_indices = [cand_indices[p] for p in selected_pos]

    # Score all docs in this paper's corpus (used for easy/hard noise classification)
    all_scores = (doc_embs @ query_emb.T).squeeze()   # (N,)

    return retrieved_docs, retrieved_indices, all_scores


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def make_noisy_context(
    clean_docs:         List[Document],
    documents:          List[Document],
    noise_k:            int   = 2,
    rng:                Any   = None,
    shuffle:            bool  = True,
    global_noise_pool:  Optional[List[Document]]  = None,
    global_pool_embs:   Optional[np.ndarray]      = None,
    query_emb:          Optional[np.ndarray]       = None,
    current_paper_id:   Optional[str]             = None,
    noise_mode:         str   = "cross_doc_easy",
) -> Dict[str, Any]:
    """
    noise_mode="cross_doc_easy":
        Top 50–90th percentile similarity with query.
        → Semantically similar chunks → retriever likely confuses these → most confusing for model.

    noise_mode="cross_doc_hard":
        Bottom 25th percentile similarity with query.
        → Low-relevance chunks → model can more easily ignore them.

    noise_mode="cross_document":
        Random from pool (baseline, no similarity filtering).

    Thresholds computed DYNAMICALLY via percentile → adapts to any embedding space.
    """
    rng       = rng or random
    clean_ids = {doc.id for doc in clean_docs}
    noise_docs: List[Document] = []
    noise_source = noise_mode

    if global_noise_pool is None or len(global_noise_pool) == 0:
        return {
            "clean_contexts": [d.text for d in clean_docs],
            "noisy_contexts": [d.text for d in clean_docs],
            "has_true_noise": False,
            "noise_source":   "none",
        }

    # Filter: exclude chunks from the same paper as the query
    candidate_pool = [
        (i, d) for i, d in enumerate(global_noise_pool)
        if d.id not in clean_ids
        and (d.metadata or {}).get("source_doc_id") != current_paper_id
    ]

    if noise_mode in ("cross_doc_easy", "cross_doc_hard") \
            and global_pool_embs is not None \
            and query_emb is not None \
            and len(candidate_pool) > 0:          # BUG FIX 2: guard empty pool

        pool_indices = [i for i, _ in candidate_pool]
        pool_embs    = global_pool_embs[pool_indices]      # (M, D)

        # BUG FIX 1: ensure query_emb is 2D (1,D) for matmul
        qe = query_emb.reshape(1, -1) if query_emb.ndim == 1 else query_emb
        sims = (pool_embs @ qe.T).squeeze()                # (M,) or scalar

        # BUG FIX 3: scalar when M=1
        sims = np.atleast_1d(sims)

        # Dynamic percentile thresholds
        if noise_mode == "cross_doc_easy":
            lo = float(np.percentile(sims, 50))
            hi = float(np.percentile(sims, 90))
            mask = (sims >= lo) & (sims < hi)
        else:  # cross_doc_hard
            threshold = float(np.percentile(sims, 25))
            mask = sims <= threshold

        filtered = [candidate_pool[j] for j in range(len(candidate_pool)) if mask[j]]

        if len(filtered) < noise_k:
            filtered = candidate_pool
            noise_source = f"{noise_mode}_fallback_random"

        sampled    = rng.sample(filtered, min(noise_k, len(filtered)))
        noise_docs = [d for _, d in sampled]

    elif len(candidate_pool) > 0:
        # cross_document baseline: random
        sampled    = rng.sample(candidate_pool, min(noise_k, len(candidate_pool)))
        noise_docs = [d for _, d in sampled]

    if len(noise_docs) == 0:
        return {
            "clean_contexts": [d.text for d in clean_docs],
            "noisy_contexts": [d.text for d in clean_docs],
            "has_true_noise": False,
            "noise_source":   "none",
        }

    noisy_docs = list(clean_docs) + noise_docs
    if shuffle:
        rng.shuffle(noisy_docs)

    return {
        "clean_contexts": [d.text for d in clean_docs],
        "noisy_contexts": [d.text for d in noisy_docs],
        "has_true_noise": True,
        "noise_source":   noise_source,
    }


# ---------------------------------------------------------------------------
# Stage 3 — per-paper example assembly (fast, no encoder calls)
# ---------------------------------------------------------------------------

def _assemble_one_paper(
    pid:               str,
    paper_data:        Dict[str, Any],
    query_emb:         np.ndarray,        # (1, D)
    query:             str,
    summarizer:        T5Summarizer,
    split_name:        str,
    final_k:           int,
    noise_k:           int,
    add_noisy:         bool,
    shuffle_noisy:     bool,
    noise_mode:        str,
    global_noise_pool: Optional[List[Document]],
    global_pool_embs:  Optional[np.ndarray],   # (N_pool, D)
    rng:               random.Random,
) -> List[Dict[str, Any]]:
    chunks   = paper_data["chunks"]
    doc_embs = paper_data["embeddings"]
    abstract = paper_data["abstract"]

    documents = make_documents_from_chunks(chunks)
    if not documents:
        return []

    index = _build_faiss_index(doc_embs)

    retrieved_docs, _, _ = retrieve_clean(
        query_emb=query_emb,
        documents=documents,
        doc_embs=doc_embs,
        index=index,
        final_k=final_k,
        noise_k=noise_k if add_noisy else 0,
    )
    if not retrieved_docs:
        return []

    clean_contexts = [d.text for d in retrieved_docs]

    clean_ex = summarizer.build_training_example(
        query=query,
        contexts=clean_contexts,
        target_text=abstract,
        max_contexts=3,
    )
    clean_ex.update({
        "paper_id":      pid,
        "split":         split_name,
        "sample_type":   "clean",
        "num_contexts":  len(clean_contexts),
        "num_chunks":    len(chunks),
        "num_documents": len(documents),
    })

    records = [clean_ex]

    if add_noisy:
        bundle = make_noisy_context(
            clean_docs=retrieved_docs,
            documents=documents,
            noise_k=noise_k,
            rng=rng,
            shuffle=shuffle_noisy,
            global_noise_pool=global_noise_pool,
            global_pool_embs=global_pool_embs,
            query_emb=query_emb,
            current_paper_id=pid,
            noise_mode=noise_mode,
        )
        if bundle["has_true_noise"]:
            noisy_ex = summarizer.build_training_example(
                query=query,
                contexts=bundle["noisy_contexts"],
                target_text=abstract,
                max_contexts=3,
            )
            noisy_ex.update({
                "paper_id":      pid,
                "split":         split_name,
                "sample_type":   "noisy",
                "num_contexts":  len(bundle["noisy_contexts"]),
                "noise_mode":    noise_mode,
                "noise_source":  bundle["noise_source"],
                "num_chunks":    len(chunks),
                "num_documents": len(documents),
            })
            records.append(noisy_ex)

    return records


# ---------------------------------------------------------------------------
# Full split builder — 3-stage
# ---------------------------------------------------------------------------

def build_split_fast(
    papers:               List[Dict[str, str]],
    split_name:           str,
    encoder:              DenseEncoder,
    summarizer:           T5Summarizer,
    limit:                Optional[int],
    final_k:              int,
    noise_k:              int,
    min_chunks:           int,
    use_multiple_queries: bool,
    add_noisy:            bool,
    shuffle_noisy:        bool,
    noise_mode:           str,
    global_noise_pool:    Optional[List[Document]],
    global_pool_embs:     Optional[np.ndarray],
    rng:                  random.Random,
    num_chunk_workers:    int = 4,
    paper_batch:          int = 500,
) -> List[Dict[str, Any]]:

    print(f"\n── {split_name.upper()} [{noise_mode}] ──────────────────────")

    chunked = batch_chunk_papers(papers, split_name,
                                 num_workers=num_chunk_workers, limit=limit)
    if not chunked:
        return []

    # BUG FIX 6: filter papers with too few chunks (same as noise pool)
    chunked = [c for c in chunked if len(c["chunks"]) >= min_chunks]
    if not chunked:
        return []

    paper_map = batch_encode_all_chunks(chunked, encoder, paper_batch=paper_batch)

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    query_emb_cache = {q: encoder.encode(q) for q in DEFAULT_QUERY_TEMPLATES}

    records: List[Dict[str, Any]] = []
    total = len(paper_map)

    for i, (pid, pdata) in enumerate(
            tqdm(paper_map.items(), desc=f"  assemble [{split_name}]", unit="paper")):
        try:
            query     = choose_query(pid, use_multiple_queries)
            query_emb = query_emb_cache[query]

            examples = _assemble_one_paper(
                pid=pid,
                paper_data=pdata,
                query_emb=query_emb,
                query=query,
                summarizer=summarizer,
                split_name=split_name,
                final_k=final_k,
                noise_k=noise_k if add_noisy else 0,
                add_noisy=add_noisy,
                shuffle_noisy=shuffle_noisy,
                noise_mode=noise_mode,
                global_noise_pool=global_noise_pool,
                global_pool_embs=global_pool_embs,
                rng=rng,
            )
            records.extend(examples)
        except Exception as e:
            print(f"[WARN] {pid}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  [{split_name}] {i+1}/{total} -> {len(records)} records")

    return records


def build_test_split_fast(
    papers:               List[Dict[str, str]],
    encoder:              DenseEncoder,
    summarizer:           T5Summarizer,
    limit:                Optional[int],
    final_k:              int,
    noise_k:              int,
    min_chunks:           int,
    use_multiple_queries: bool,
    shuffle_noisy:        bool,
    global_noise_pool:    Optional[List[Document]],
    global_pool_embs:     Optional[np.ndarray],
    rng:                  random.Random,
    num_chunk_workers:    int = 4,
    paper_batch:          int = 500,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    BUG FIX 5: chunk + encode ONCE, run easy AND hard noise in same pass.
    Returns (test_clean, test_noisy_easy, test_noisy_hard).
    """
    print(f"\n── TEST [easy + hard noise — single encode pass] ──────────────────────")

    chunked = batch_chunk_papers(papers, "test",
                                 num_workers=num_chunk_workers, limit=limit)
    if not chunked:
        return [], [], []

    paper_map = batch_encode_all_chunks(chunked, encoder, paper_batch=paper_batch)

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    query_emb_cache = {q: encoder.encode(q) for q in DEFAULT_QUERY_TEMPLATES}

    clean_records: List[Dict[str, Any]] = []
    easy_records:  List[Dict[str, Any]] = []
    hard_records:  List[Dict[str, Any]] = []

    for i, (pid, pdata) in enumerate(
            tqdm(paper_map.items(), desc="  assemble [test]", unit="paper")):
        try:
            query     = choose_query(pid, use_multiple_queries)
            query_emb = query_emb_cache[query]

            chunks   = pdata["chunks"]
            doc_embs = pdata["embeddings"]
            abstract = pdata["abstract"]

            documents = make_documents_from_chunks(chunks)
            if not documents:
                continue

            index = _build_faiss_index(doc_embs)
            retrieved_docs, _, _ = retrieve_clean(
                query_emb=query_emb, documents=documents,
                doc_embs=doc_embs, index=index,
                final_k=final_k, noise_k=noise_k,
            )
            if not retrieved_docs:
                continue

            clean_contexts = [d.text for d in retrieved_docs]
            clean_ex = summarizer.build_training_example(
                query=query, contexts=clean_contexts,
                target_text=abstract, max_contexts=3,
            )
            clean_ex.update({
                "paper_id": pid, "split": "test", "sample_type": "clean",
                "num_contexts": len(clean_contexts),
                "num_chunks": len(chunks), "num_documents": len(documents),
            })
            clean_records.append(clean_ex)

            # Easy noise
            bundle_easy = make_noisy_context(
                clean_docs=retrieved_docs, documents=documents,
                noise_k=noise_k, rng=rng, shuffle=shuffle_noisy,
                global_noise_pool=global_noise_pool,
                global_pool_embs=global_pool_embs,
                query_emb=query_emb, current_paper_id=pid,
                noise_mode="cross_doc_easy",
            )
            if bundle_easy["has_true_noise"]:
                ex = summarizer.build_training_example(
                    query=query, contexts=bundle_easy["noisy_contexts"],
                    target_text=abstract, max_contexts=3,
                )
                ex.update({
                    "paper_id": pid, "split": "test", "sample_type": "noisy",
                    "noise_mode": "cross_doc_easy",
                    "noise_source": bundle_easy["noise_source"],
                    "num_chunks": len(chunks), "num_documents": len(documents),
                    "num_contexts": len(bundle_easy["noisy_contexts"]),
                })
                easy_records.append(ex)

            # Hard noise
            bundle_hard = make_noisy_context(
                clean_docs=retrieved_docs, documents=documents,
                noise_k=noise_k, rng=rng, shuffle=shuffle_noisy,
                global_noise_pool=global_noise_pool,
                global_pool_embs=global_pool_embs,
                query_emb=query_emb, current_paper_id=pid,
                noise_mode="cross_doc_hard",
            )
            if bundle_hard["has_true_noise"]:
                ex = summarizer.build_training_example(
                    query=query, contexts=bundle_hard["noisy_contexts"],
                    target_text=abstract, max_contexts=3,
                )
                ex.update({
                    "paper_id": pid, "split": "test", "sample_type": "noisy",
                    "noise_mode": "cross_doc_hard",
                    "noise_source": bundle_hard["noise_source"],
                    "num_chunks": len(chunks), "num_documents": len(documents),
                    "num_contexts": len(bundle_hard["noisy_contexts"]),
                })
                hard_records.append(ex)

        except Exception as e:
            print(f"[WARN] {pid}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  [test] {i+1}/{len(paper_map)} "
                  f"clean={len(clean_records)} easy={len(easy_records)} hard={len(hard_records)}")

    return clean_records, easy_records, hard_records


# ---------------------------------------------------------------------------
# Noise pool builder (also benefits from batch encode)
# ---------------------------------------------------------------------------

def build_global_noise_pool(
    papers:      List[Dict[str, str]],
    encoder:     DenseEncoder,
    limit:       int = 300,
    min_chunks:  int = 3,
    num_workers: int = 4,
    paper_batch: int = 500,
) -> Tuple[List[Document], np.ndarray]:
    """
    Returns (pool_docs, pool_embeddings).
    pool_embeddings shape: (N_chunks, D) — used to score easy/hard noise candidates.
    """
    print("\nBuilding cross-document noise pool...")
    chunked = batch_chunk_papers(papers, "noise_pool",
                                 num_workers=num_workers, limit=limit)
    chunked = [c for c in chunked if len(c["chunks"]) >= min_chunks]

    if not chunked:
        return [], np.empty((0, 384), dtype=np.float32)

    paper_map = batch_encode_all_chunks(chunked, encoder, paper_batch=paper_batch)

    pool_docs: List[Document] = []
    pool_embs_list: List[np.ndarray] = []

    for pdata in paper_map.values():
        docs = make_documents_from_chunks(pdata["chunks"])
        pool_docs.extend(docs)
        pool_embs_list.append(pdata["embeddings"])

    pool_embs = np.vstack(pool_embs_list).astype(np.float32)
    print(f"  [noise_pool] {len(pool_docs)} chunks from {len(chunked)} papers")
    return pool_docs, pool_embs


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
        description="Build RAG training data — easy/hard cross-doc noise."
    )
    parser.add_argument("--source",      type=str, default="arxiv", choices=["arxiv"])
    parser.add_argument("--arxiv_dir",   type=str, default=ARXIV_DIR)
    parser.add_argument("--output_dir",  type=str, default="./prepared_data")

    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--valid_limit", type=int, default=None)
    parser.add_argument("--test_limit",  type=int, default=None)

    parser.add_argument("--final_k",  type=int, default=3)
    parser.add_argument("--noise_k",  type=int, default=2)
    parser.add_argument("--min_chunks", type=int, default=3)
    parser.add_argument("--single_query", action="store_true")
    parser.add_argument("--no_shuffle_noisy", action="store_true")

    # Easy noise: similarity in range [lo, hi) with query
    parser.add_argument("--easy_sim_lo", type=float, default=0.3)
    parser.add_argument("--easy_sim_hi", type=float, default=0.6)
    # Hard noise: similarity < threshold
    parser.add_argument("--hard_sim_threshold", type=float, default=0.3)

    # Pool large enough to supply both easy and hard noise candidates
    parser.add_argument("--noise_pool_limit", type=int, default=500)

    # Train noise mode: easy, hard, or random (cross_document)
    parser.add_argument(
        "--train_noise_mode", type=str, default="cross_doc_easy",
        choices=["cross_doc_easy", "cross_doc_hard", "cross_document"],
        help="Noise mode used when building the train split.",
    )

    # Performance
    parser.add_argument("--num_workers",       type=int, default=4)
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--paper_batch",       type=int, default=500)
    parser.add_argument("--seed",              type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------------------------
    # Load raw papers
    # ------------------------------------------------------------------
    print("Loading dataset...")
    train_papers = load_arxiv_arrow(os.path.join(args.arxiv_dir, "train"))
    valid_papers = load_arxiv_arrow(os.path.join(args.arxiv_dir, "validation"))
    test_papers  = load_arxiv_arrow(os.path.join(args.arxiv_dir, "test"))
    print(f"  train={len(train_papers):,}  valid={len(valid_papers):,}  "
          f"test={len(test_papers):,}")

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    print("\nLoading encoder & summarizer...")
    encoder              = DenseEncoder(batch_size=args.encode_batch_size)
    summarizer           = T5Summarizer()
    use_multiple_queries = not args.single_query
    shuffle_noisy        = not args.no_shuffle_noisy

    # ------------------------------------------------------------------
    # Build noise pool ONCE — shared for both easy and hard noise
    # ------------------------------------------------------------------
    print(f"\n  noise_pool_limit={args.noise_pool_limit}")
    global_noise_pool, global_pool_embs = build_global_noise_pool(
        papers=train_papers,
        encoder=encoder,
        limit=args.noise_pool_limit,
        min_chunks=args.min_chunks,
        num_workers=args.num_workers,
        paper_batch=args.paper_batch,
    )

    # Shared kwargs cho build_split_fast / build_test_split_fast
    _shared = dict(
        encoder=encoder, summarizer=summarizer,
        min_chunks=args.min_chunks,
        use_multiple_queries=use_multiple_queries,
        final_k=args.final_k, noise_k=args.noise_k,
        global_noise_pool=global_noise_pool,
        global_pool_embs=global_pool_embs,
        rng=rng,
        num_chunk_workers=args.num_workers,
        paper_batch=args.paper_batch,
    )

    # ------------------------------------------------------------------
    # Train split — uses train_noise_mode (default: easy)
    # ------------------------------------------------------------------
    train_records = build_split_fast(
        papers=train_papers, split_name="train",
        limit=args.train_limit,
        add_noisy=True, shuffle_noisy=shuffle_noisy,
        noise_mode=args.train_noise_mode,
        **_shared,
    )

    # ------------------------------------------------------------------
    # Validation split — no noise needed
    # ------------------------------------------------------------------
    valid_records = build_split_fast(
        papers=valid_papers, split_name="validation",
        limit=args.valid_limit,
        encoder=encoder, summarizer=summarizer,
        min_chunks=args.min_chunks,
        use_multiple_queries=use_multiple_queries,
        final_k=args.final_k, noise_k=0,
        add_noisy=False, shuffle_noisy=False,
        noise_mode="cross_document",
        global_noise_pool=None,
        global_pool_embs=None,
        rng=rng,
        num_chunk_workers=args.num_workers,
        paper_batch=args.paper_batch,
    )

    # ------------------------------------------------------------------
    # Test split — chunk + encode once, generate easy + hard noise in a single pass
    # ------------------------------------------------------------------
    test_clean, test_noisy_easy, test_noisy_hard = build_test_split_fast(
        papers=test_papers,
        encoder=encoder, summarizer=summarizer,
        limit=args.test_limit, final_k=args.final_k, noise_k=args.noise_k,
        min_chunks=args.min_chunks, use_multiple_queries=use_multiple_queries,
        shuffle_noisy=shuffle_noisy,
        global_noise_pool=global_noise_pool,
        global_pool_embs=global_pool_embs,
        rng=rng,
        num_chunk_workers=args.num_workers,
        paper_batch=args.paper_batch,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    paths = {
        "train":           os.path.join(args.output_dir, "train.jsonl"),
        "valid":           os.path.join(args.output_dir, "valid.jsonl"),
        "test_clean":      os.path.join(args.output_dir, "test_clean.jsonl"),
        "test_noisy_easy": os.path.join(args.output_dir, "test_noisy_easy.jsonl"),
        "test_noisy_hard": os.path.join(args.output_dir, "test_noisy_hard.jsonl"),
    }
    write_jsonl(paths["train"],           train_records)
    write_jsonl(paths["valid"],           valid_records)
    write_jsonl(paths["test_clean"],      test_clean)
    write_jsonl(paths["test_noisy_easy"], test_noisy_easy)
    write_jsonl(paths["test_noisy_hard"], test_noisy_hard)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    train_c = sum(1 for r in train_records if r["sample_type"] == "clean")
    train_n = sum(1 for r in train_records if r["sample_type"] == "noisy")

    def pair_rate(noisy, clean):
        pct = 100 * len(noisy) / max(len(clean), 1)
        return f"{len(noisy)}/{len(clean)} ({pct:.1f}%)"

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  train clean      : {train_c:>7,}  ->  {paths['train']}")
    print(f"  train noisy      : {train_n:>7,}      mode={args.train_noise_mode}")
    print(f"  valid            : {len(valid_records):>7,}  ->  {paths['valid']}")
    print(f"  test clean       : {len(test_clean):>7,}  ->  {paths['test_clean']}")
    print(f"  test noisy easy  : {len(test_noisy_easy):>7,}  ->  {paths['test_noisy_easy']}")
    print(f"    pair rate easy : {pair_rate(test_noisy_easy, test_clean)}")
    print(f"  test noisy hard  : {len(test_noisy_hard):>7,}  ->  {paths['test_noisy_hard']}")
    print(f"    pair rate hard : {pair_rate(test_noisy_hard, test_clean)}")
    print(f"  noise strategy   : dynamic percentile (easy=p50-p90, hard=bottom p25)")
    print(f"  noise pool       : {len(global_noise_pool):,} chunks from {args.noise_pool_limit} papers")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()