"""Tests for the synthetic-data pipeline — fake LLM, real verifier gate."""

import json

from nyaya.eval.verify import CitationVerifier
from nyaya.pipelines.synth import (
    SynthGenerator,
    _word_set,
    build_prompt,
    is_contaminated,
    make_refusals,
    parse_pairs,
)
from nyaya.schema import Chunk, Era, Subject


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


def _verifier() -> CitationVerifier:
    # corpus = the source chunk plus one other real section (for the
    # ungrounded-citation case)
    other = _chunk().model_copy(update={"id": "bns_2023:s104", "section": "104"})
    return CitationVerifier.from_chunks([_chunk(), other])


def _llm_returning(payload) -> callable:
    return lambda prompt: json.dumps(payload)


# --- prompt ---------------------------------------------------------------------

def test_prompt_contains_section_text_and_citation_form():
    p = build_prompt(_chunk(), n=3)
    assert "Whoever commits murder" in p
    assert "[BNS §103]" in p  # the exact citation form the answer must use
    assert "JSON array" in p


# --- output parsing ---------------------------------------------------------------

def test_parse_valid_json_array():
    raw = '[{"q": "What is X?", "a": "X is Y [BNS §103]."}]'
    assert parse_pairs(raw) == [{"q": "What is X?", "a": "X is Y [BNS §103]."}]


def test_parse_json_wrapped_in_prose():
    raw = 'Here are the pairs:\n[{"q": "Q1?", "a": "A1."}]\nHope this helps!'
    assert len(parse_pairs(raw)) == 1


def test_parse_garbage_returns_empty():
    assert parse_pairs("I cannot generate that.") == []
    assert parse_pairs("[not json") == []


def test_parse_filters_malformed_items():
    raw = '[{"q": "ok?", "a": "ok."}, {"q": ""}, {"a": "orphan"}, "string", 42]'
    assert parse_pairs(raw) == [{"q": "ok?", "a": "ok."}]


# --- verifier gate ---------------------------------------------------------------

def test_pair_citing_source_section_is_kept():
    llm = _llm_returning([{"q": "What is the punishment for murder?",
                           "a": "Death or life imprisonment [BNS §103]."}])
    gen = SynthGenerator(llm=llm, verifier=_verifier())
    rows = gen.generate_for_chunk(_chunk())
    assert len(rows) == 1
    assert rows[0]["source_chunk_id"] == "bns_2023:s103"
    assert rows[0]["kind"] == "qa"
    assert gen.stats["kept"] == 1


def test_pair_citing_wrong_section_is_rejected():
    # §104 exists in the corpus but is NOT the source chunk — ungrounded
    llm = _llm_returning([{"q": "Q?", "a": "Answer citing [BNS §104]."}])
    gen = SynthGenerator(llm=llm, verifier=_verifier())
    assert gen.generate_for_chunk(_chunk()) == []
    assert gen.stats["unverified"] == 1


def test_pair_citing_invented_section_is_rejected():
    llm = _llm_returning([{"q": "Q?", "a": "Answer citing [BNS §999]."}])
    gen = SynthGenerator(llm=llm, verifier=_verifier())
    assert gen.generate_for_chunk(_chunk()) == []
    assert gen.stats["unverified"] == 1


def test_pair_without_any_citation_is_rejected():
    llm = _llm_returning([{"q": "Q?", "a": "An answer with no citation at all."}])
    gen = SynthGenerator(llm=llm, verifier=_verifier())
    assert gen.generate_for_chunk(_chunk()) == []
    assert gen.stats["unverified"] == 1


def test_unparsable_output_counted():
    gen = SynthGenerator(llm=lambda p: "refusing to answer", verifier=_verifier())
    assert gen.generate_for_chunk(_chunk()) == []
    assert gen.stats["unparsable"] == 1


# --- contamination guard -----------------------------------------------------------

GOLD_Q = "What is the punishment for murder under the Bharatiya Nyaya Sanhita?"


def test_gold_near_duplicate_question_rejected():
    gold_sets = [_word_set(GOLD_Q)]
    assert is_contaminated(
        "What is the punishment for murder under the Bharatiya Nyaya Sanhita 2023?",
        gold_sets,
    )


def test_unrelated_question_passes():
    gold_sets = [_word_set(GOLD_Q)]
    assert not is_contaminated("How is a driving licence renewed?", gold_sets)


def test_contamination_enforced_in_generator():
    llm = _llm_returning([{"q": GOLD_Q, "a": "Death or life [BNS §103]."}])
    gen = SynthGenerator(llm=llm, verifier=_verifier(), gold_questions=[GOLD_Q])
    assert gen.generate_for_chunk(_chunk()) == []
    assert gen.stats["contaminated"] == 1


# --- refusals ------------------------------------------------------------------------

def test_refusals_carry_escalation_and_kind():
    rows = make_refusals(5)
    assert len(rows) == 5
    assert all(r["kind"] == "refusal" for r in rows)
    assert all("NALSA" in r["answer"] for r in rows)
    assert all(r["source_chunk_id"] is None for r in rows)


def test_refusals_deterministic():
    assert make_refusals(12) == make_refusals(12)
