"""Tests for DPO preference-pair building — pure python, no GPU, no vLLM."""

from nyaya.schema import Chunk, Era, Subject
from nyaya.training.dpo import build_preference_pairs


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


CHUNK_MAP = {"bns_2023:s103": _chunk()}


def _record(**over) -> dict:
    rec = {
        "question": "What is the punishment for murder?",
        "era": "new_code",
        "chunk_ids": ["bns_2023:s103"],
        "answer": "Death or life imprisonment [BNS §103] and also [BNS §999].",
        "clean_answer": "Death or life imprisonment [BNS §103] and also "
                        "I could not verify this in my sources.",
        "n_hallucinated": 1,
        "n_ungrounded": 0,
    }
    rec.update(over)
    return rec


def test_failure_record_becomes_pair():
    pairs, stats = build_preference_pairs([_record()], CHUNK_MAP)
    assert stats["pairs"] == 1
    pair = pairs[0]
    # rejected = raw hallucinated answer; chosen = verifier-corrected answer
    assert "[BNS §999]" in pair["rejected"][0]["content"]
    assert "[BNS §999]" not in pair["chosen"][0]["content"]
    assert "could not verify" in pair["chosen"][0]["content"]


def test_prompt_mirrors_production_shape():
    pairs, _ = build_preference_pairs([_record()], CHUNK_MAP)
    prompt = pairs[0]["prompt"]
    assert [m["role"] for m in prompt] == ["system", "user"]
    user = prompt[1]["content"]
    assert user.startswith("STATUTE SECTIONS:")
    assert "Whoever commits murder" in user  # the harvested retrieval context
    assert "QUESTION: What is the punishment for murder?" in user


def test_clean_answer_is_not_a_pair():
    rec = _record(n_hallucinated=0, n_ungrounded=0)
    pairs, stats = build_preference_pairs([rec], CHUNK_MAP)
    assert pairs == []
    assert stats["clean"] == 1


def test_unchanged_correction_is_skipped():
    rec = _record(clean_answer=_record()["answer"])  # corrected == raw
    pairs, stats = build_preference_pairs([rec], CHUNK_MAP)
    assert pairs == []
    assert stats["unchanged"] == 1


def test_missing_context_chunks_skipped():
    rec = _record(chunk_ids=["gone:s1"])
    pairs, stats = build_preference_pairs([rec], CHUNK_MAP)
    assert pairs == []
    assert stats["missing_chunks"] == 1


def test_ungrounded_only_failure_also_becomes_pair():
    rec = _record(n_hallucinated=0, n_ungrounded=1)
    pairs, stats = build_preference_pairs([rec], CHUNK_MAP)
    assert stats["pairs"] == 1


def test_mixed_batch_stats_add_up():
    records = [
        _record(),
        _record(n_hallucinated=0, n_ungrounded=0),
        _record(chunk_ids=[]),
    ]
    pairs, stats = build_preference_pairs(records, CHUNK_MAP)
    assert stats == {"pairs": 1, "clean": 1, "unchanged": 0, "missing_chunks": 1}
    assert len(pairs) == 1
