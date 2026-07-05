"""Tests for SFT data preparation — pure python; the GPU loop is not imported."""

import json

from nyaya.schema import Chunk, Era, Subject
from nyaya.training.sft import (
    build_example,
    load_pairs,
    prepare_dataset,
    split_rows,
)


def _chunk() -> Chunk:
    return Chunk(
        id="bns_2023:s103",
        text="103. Punishment for murder.—Whoever commits murder shall be punished…",
        doc_id="bns_2023",
        act="Bharatiya Nyaya Sanhita, 2023",
        section="103",
        era=Era.NEW_CODE,
        subject=Subject.CRIMINAL,
        source="test",
    )


QA_ROW = {
    "id": "bns_2023:s103_q0",
    "question": "What is the punishment for murder?",
    "answer": "Death or imprisonment for life [BNS §103].",
    "source_chunk_id": "bns_2023:s103",
    "era": "new_code",
    "kind": "qa",
}
REFUSAL_ROW = {
    "id": "refusal_0001",
    "question": "Guarantee that I will win my case.",
    "answer": "I can't help with that. … NALSA: 15100 or nalsa.gov.in",
    "source_chunk_id": None,
    "era": "neutral",
    "kind": "refusal",
}


def test_qa_example_mirrors_production_prompt():
    ex = build_example(QA_ROW, {"bns_2023:s103": _chunk()})
    assert ex is not None
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    user = ex["messages"][1]["content"]
    assert user.startswith("STATUTE SECTIONS:")  # exact production shape
    assert "Whoever commits murder" in user  # source section is the context
    assert "QUESTION: What is the punishment for murder?" in user
    assert ex["messages"][2]["content"] == QA_ROW["answer"]


def test_system_prompt_is_production_system_prompt():
    from nyaya.generation.answer import _SYSTEM_PROMPT

    ex = build_example(QA_ROW, {"bns_2023:s103": _chunk()})
    assert ex["messages"][0]["content"] == _SYSTEM_PROMPT


def test_refusal_example_has_no_context_block():
    ex = build_example(REFUSAL_ROW, {})
    user = ex["messages"][1]["content"]
    assert "STATUTE SECTIONS" not in user
    assert user == REFUSAL_ROW["question"]
    assert "NALSA" in ex["messages"][2]["content"]


def test_qa_row_with_missing_chunk_is_dropped():
    assert build_example(QA_ROW, {}) is None  # unverifiable context → never train


def test_split_is_deterministic_and_disjoint():
    rows = [{"id": f"row_{i}"} for i in range(1000)]
    t1, v1 = split_rows(rows)
    t2, v2 = split_rows(rows)
    assert t1 == t2 and v1 == v2  # reproducible across runs
    ids_t = {r["id"] for r in t1}
    ids_v = {r["id"] for r in v1}
    assert not (ids_t & ids_v)
    assert len(ids_t) + len(ids_v) == 1000
    assert 20 <= len(ids_v) <= 90  # ~5% of 1000, hash-bucket variance allowed


def test_prepare_dataset_end_to_end(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        "\n".join(json.dumps(r) for r in (QA_ROW, REFUSAL_ROW)) + "\n",
        encoding="utf-8",
    )
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "bns_2023.jsonl").write_text(
        _chunk().model_dump_json() + "\n", encoding="utf-8"
    )

    train, val, stats = prepare_dataset(pairs, processed)
    assert stats["total"] == 2
    assert stats["dropped_missing_chunk"] == 0
    assert stats["train"] + stats["val"] == 2


def test_load_pairs_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(QA_ROW) + "\n\n" + json.dumps(REFUSAL_ROW) + "\n")
    rows = load_pairs(p)
    assert [r["id"] for r in rows] == ["bns_2023:s103_q0", "refusal_0001"]
