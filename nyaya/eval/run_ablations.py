"""Retrieval ablation harness — populates RESULTS.md section 1.

Runs each retrieval configuration over the gold set and reports
recall@5/10, MRR, and nDCG@10. Needs the index + embedding model, so it
runs on CHAMP (not the laptop). The metric math lives in nyaya/eval/metrics.py
and is unit-tested separately.

Configs:
    dense   — bge-m3 dense vectors only (Qdrant)
    bm25    — lexical BM25 only
    hybrid  — dense + BM25 merged with RRF (the production path)

Usage (on CHAMP, venv active):
    python -m nyaya.eval.run_ablations
"""

from __future__ import annotations

import re
from pathlib import Path

from nyaya.eval.gold import GoldQuestion, load_gold
from nyaya.eval.metrics import aggregate
from nyaya.schema import Chunk

CONFIGS = ("dense", "bm25", "hybrid")
GOLD_PATH = "data/gold/gold_set.jsonl"
TOP_K = 10  # retrieve depth for the ablation (recall@10 needs at least 10)


def _section_keys(chunks: list[Chunk]) -> list[str]:
    """Ordered, de-duplicated '<doc_id>:<section>' keys from ranked chunks."""
    keys: list[str] = []
    seen: set[str] = set()
    for c in chunks:
        if not c.section:
            continue
        key = f"{c.doc_id}:{c.section}"
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _ordered_chunks(retriever, ids: list[str]) -> list[Chunk]:
    """Fetch chunks for ids and restore the ranked order."""
    by_id = {c.id: c for c in retriever._fetch_chunks(ids)}
    return [by_id[i] for i in ids if i in by_id]


def predicted_keys(retriever, q: GoldQuestion, config: str, k: int) -> list[str]:
    """Ranked section keys a config returns for one gold question."""
    era_values = retriever._era_filter(q.era)
    if config == "dense":
        vec = retriever._embedder.encode(
            [q.question], normalize_embeddings=True
        ).tolist()[0]
        ids = retriever._dense_search(vec, k * 3, era_values)
        chunks = _ordered_chunks(retriever, ids)
    elif config == "bm25":
        ids = retriever._bm25_search(q.question, k * 3, era_values)
        chunks = _ordered_chunks(retriever, ids)
    elif config == "hybrid":
        chunks = retriever.retrieve(q.question, era=q.era, final_k=k)
    else:
        raise ValueError(f"unknown config {config!r}")
    if era_values:
        chunks = [c for c in chunks if c.era.value in era_values]
    return _section_keys(chunks)[:k]


def run() -> dict[str, dict[str, float]]:
    from nyaya.retrieval.hybrid import HybridRetriever

    gold = load_gold(GOLD_PATH)
    retriever = HybridRetriever()
    results: dict[str, dict[str, float]] = {}
    for config in CONFIGS:
        rows = [
            (predicted_keys(retriever, q, config, TOP_K), set(q.relevant))
            for q in gold
        ]
        results[config] = aggregate(rows, ks=(5, 10))
        print(f"[{config}] {results[config]}")
    _write_results(results, n=len(gold))
    return results


_LABELS = {"dense": "dense (bge-m3)", "bm25": "BM25", "hybrid": "hybrid (RRF)"}


def _write_results(results: dict[str, dict[str, float]], n: int) -> None:
    """Replace section 1 of RESULTS.md with the computed table."""
    rows = ["| Config | recall@5 | recall@10 | MRR | nDCG@10 |", "|---|---|---|---|---|"]
    for cfg in CONFIGS:
        m = results[cfg]
        rows.append(
            f"| {_LABELS[cfg]} | {m['recall@5']:.3f} | {m['recall@10']:.3f} "
            f"| {m['mrr']:.3f} | {m['ndcg@10']:.3f} |"
        )
    rows.append("| hybrid + rerank | – | – | – | – |  <!-- reranker not yet wired -->")
    block = f"## 1. Retrieval ablation (gold set, n={n})\n" + "\n".join(rows) + "\n"

    path = Path("RESULTS.md")
    if not path.exists():
        path.write_text("# Results\n\n" + block, encoding="utf-8")
        print(f"RESULTS.md created (n={n}).")
        return

    text = path.read_text(encoding="utf-8")
    if re.search(r"## 1\. Retrieval ablation", text):
        # replace from the section-1 header up to (but not including) the next '## '
        new = re.sub(
            r"## 1\. Retrieval ablation.*?(?=\n## )",
            block + "\n",
            text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        new = text.rstrip() + "\n\n" + block
    path.write_text(new, encoding="utf-8")
    print(f"RESULTS.md updated (n={n}).")


if __name__ == "__main__":
    run()
