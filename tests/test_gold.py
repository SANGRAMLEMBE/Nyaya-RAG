"""Tests for the gold-set loader + validator (no GPU, no corpus needed)."""

import json

from nyaya.eval.gold import (
    GoldQuestion,
    load_gold,
    validate_gold,
)


def _q(qid: str, relevant: list[str], qtype: str = "section_lookup") -> GoldQuestion:
    return GoldQuestion(
        id=qid,
        question="What is the punishment for the offence in question?",
        era="new_code",
        subject="criminal",
        qtype=qtype,
        relevant=relevant,
    )


def test_relevant_pairs_splits_keys():
    q = _q("gold_001", ["bns_2023:103", "ipc_1860:302"])
    assert q.relevant_pairs() == [("bns_2023", "103"), ("ipc_1860", "302")]


def test_validate_passes_when_all_sections_in_corpus():
    corpus = {("bns_2023", "103"), ("ipc_1860", "302")}
    problems = validate_gold([_q("gold_001", ["bns_2023:103"])], corpus)
    assert problems == []


def test_validate_flags_missing_section():
    corpus = {("bns_2023", "103")}
    problems = validate_gold([_q("gold_001", ["bns_2023:999"])], corpus)
    assert len(problems) == 1
    assert "bns_2023:999 not in corpus" in problems[0]


def test_validate_flags_duplicate_ids():
    corpus = {("bns_2023", "103")}
    qs = [_q("gold_001", ["bns_2023:103"]), _q("gold_001", ["bns_2023:103"])]
    problems = validate_gold(qs, corpus)
    assert any("duplicate id" in p for p in problems)


def test_validate_flags_unknown_qtype():
    corpus = {("bns_2023", "103")}
    problems = validate_gold([_q("gold_001", ["bns_2023:103"], qtype="nonsense")], corpus)
    assert any("unknown qtype" in p for p in problems)


def test_load_gold_skips_comments_and_blanks(tmp_path):
    p = tmp_path / "g.jsonl"
    rows = [
        "# a comment",
        "",
        json.dumps({
            "id": "gold_001", "question": "What is the punishment for murder here?",
            "era": "new_code", "subject": "criminal", "qtype": "section_lookup",
            "relevant": ["bns_2023:103"],
        }),
    ]
    p.write_text("\n".join(rows), encoding="utf-8")
    gold = load_gold(p)
    assert len(gold) == 1
    assert gold[0].id == "gold_001"
