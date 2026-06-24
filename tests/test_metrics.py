"""Tests for retrieval metrics — pure functions, no GPU."""

import math

from nyaya.eval.metrics import (
    aggregate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_full_and_partial():
    pred = ["a", "b", "c", "d"]
    assert recall_at_k(pred, {"a", "c"}, 5) == 1.0
    assert recall_at_k(pred, {"a", "x"}, 5) == 0.5
    assert recall_at_k(pred, {"d"}, 2) == 0.0  # d is rank 4, outside top-2


def test_recall_empty_relevant():
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_perfect_is_one():
    # single relevant item at rank 1 -> perfect
    assert ndcg_at_k(["a", "b"], {"a"}, 10) == 1.0


def test_ndcg_rank2_matches_formula():
    # one relevant item at rank 2: dcg = 1/log2(3), idcg = 1/log2(2) = 1
    expected = (1.0 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_at_k(["x", "a"], {"a"}, 10), expected)


def test_ndcg_empty_relevant():
    assert ndcg_at_k(["a"], set(), 5) == 0.0


def test_aggregate_means():
    rows = [
        (["a", "b"], {"a"}),       # recall@5=1, rr=1, ndcg=1
        (["x", "y"], {"a"}),       # recall@5=0, rr=0, ndcg=0
    ]
    out = aggregate(rows, ks=(5,))
    assert out["recall@5"] == 0.5
    assert out["mrr"] == 0.5
    assert out["ndcg@5"] == 0.5
