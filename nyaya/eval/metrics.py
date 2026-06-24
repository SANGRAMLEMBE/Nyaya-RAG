"""Retrieval metrics for the ablation harness (RESULTS.md section 1).

All functions operate on *section keys* (``"<doc_id>:<section>"``): a ranked
list of predicted keys from the retriever, and the set of gold-relevant keys
for the question. Pure functions — no GPU, no model — so they are unit-tested
on the laptop; ``run_ablations`` feeds them real retriever output on CHAMP.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(predicted: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant sections found in the top-k predictions."""
    if not relevant:
        return 0.0
    topk = set(predicted[:k])
    return len(topk & relevant) / len(relevant)


def reciprocal_rank(predicted: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant prediction (0 if none retrieved)."""
    for i, key in enumerate(predicted, start=1):
        if key in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(predicted: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, key in enumerate(predicted[:k], start=1):
        if key in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def aggregate(
    rows: list[tuple[Sequence[str], set[str]]],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Mean metrics over many (predicted, relevant) pairs.

    Returns keys like ``recall@5``, ``recall@10``, ``mrr``, ``ndcg@10``.
    """
    n = len(rows) or 1
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(p, r, k) for p, r in rows) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(p, r, k) for p, r in rows) / n
    out["mrr"] = sum(reciprocal_rank(p, r) for p, r in rows) / n
    return out
