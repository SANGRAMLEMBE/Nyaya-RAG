"""Tests for the citation verifier (ADR-005) — no GPU, no model, no network."""

import json
from types import SimpleNamespace

from nyaya.eval.verify import (
    DISCLAIMER,
    CitationStatus,
    CitationVerifier,
)

# A small, fixed corpus index: (ABBR, SECTION) pairs.
CORPUS = {
    ("BNS", "103"),
    ("BNS", "303"),
    ("IPC", "302"),
    ("IPC", "120B"),
    ("CRPC", "41"),
    ("IT", "66"),
}


def _chunk(doc_id: str, section: str | None) -> SimpleNamespace:
    """Minimal stand-in for schema.Chunk — verifier only needs doc_id + section."""
    return SimpleNamespace(doc_id=doc_id, section=section)


def _verifier(aliases=None) -> CitationVerifier:
    return CitationVerifier(set(CORPUS), aliases=aliases)


# --- VERIFIED ----------------------------------------------------------------

def test_citation_in_retrieved_context_is_verified():
    v = _verifier()
    answer = "The punishment for murder is death or life imprisonment [BNS §103]."
    result = v.verify(answer, [_chunk("bns_2023", "103")])
    assert result.total == 1
    assert result.citations[0].status is CitationStatus.VERIFIED
    assert result.clean_answer == answer  # nothing stripped
    assert result.precision == 1.0
    assert result.hallucination_rate == 0.0


# --- HALLUCINATED ------------------------------------------------------------

def test_citation_absent_from_corpus_is_hallucinated_and_stripped():
    v = _verifier()
    answer = "This is governed by [BNS §999]."
    result = v.verify(answer, [_chunk("bns_2023", "103")])
    assert result.citations[0].status is CitationStatus.HALLUCINATED
    assert "[BNS §999]" not in result.clean_answer
    assert DISCLAIMER in result.clean_answer
    assert result.hallucination_rate == 1.0
    assert result.precision == 0.0


# --- UNGROUNDED --------------------------------------------------------------

def test_citation_in_corpus_but_not_context_is_ungrounded_and_stripped():
    v = _verifier()
    # IPC §302 exists in the corpus but was NOT among the retrieved chunks.
    answer = "Compare with [IPC §302]."
    result = v.verify(answer, [_chunk("bns_2023", "103")])
    assert result.citations[0].status is CitationStatus.UNGROUNDED
    assert "[IPC §302]" not in result.clean_answer
    assert DISCLAIMER in result.clean_answer
    # Ungrounded is not a hallucination — it exists, just wasn't retrieved.
    assert result.hallucination_rate == 0.0
    assert result.n_ungrounded == 1


# --- no citations ------------------------------------------------------------

def test_answer_without_citations():
    v = _verifier()
    answer = "Please consult a qualified lawyer."
    result = v.verify(answer, [_chunk("bns_2023", "103")])
    assert result.total == 0
    assert result.clean_answer == answer
    assert result.precision == 1.0
    assert result.hallucination_rate == 0.0


# --- parsing edge cases ------------------------------------------------------

def test_subsection_is_consumed_and_section_matched():
    v = _verifier()
    answer = "Theft is defined in [BNS §303(1)]."
    result = v.verify(answer, [_chunk("bns_2023", "303")])
    assert result.total == 1
    assert result.citations[0].section == "303"
    assert result.citations[0].status is CitationStatus.VERIFIED


def test_section_letter_suffix_matched():
    v = _verifier()
    answer = "Criminal conspiracy: [IPC §120B]."
    result = v.verify(answer, [_chunk("ipc_1860", "120B")])
    assert result.citations[0].section == "120B"
    assert result.citations[0].status is CitationStatus.VERIFIED


def test_abbreviation_match_is_case_insensitive():
    v = _verifier()
    answer = "Arrest powers under [CrPC §41]."
    result = v.verify(answer, [_chunk("crpc_1973", "41")])
    assert result.citations[0].status is CitationStatus.VERIFIED


def test_alias_maps_citation_abbr_to_corpus_prefix():
    v = _verifier(aliases={"ITA": "IT"})
    answer = "Cyber offence under [ITA §66]."
    result = v.verify(answer, [_chunk("it_act_2000", "66")])
    assert result.citations[0].status is CitationStatus.VERIFIED


# --- dedup + multiple occurrences -------------------------------------------

def test_duplicate_verified_citation_deduped_in_list_kept_in_text():
    v = _verifier()
    answer = "See [BNS §103]. As stated, [BNS §103] applies."
    result = v.verify(answer, [_chunk("bns_2023", "103")])
    assert result.total == 1  # deduped
    assert result.clean_answer.count("[BNS §103]") == 2  # both kept


def test_duplicate_hallucinated_citation_all_occurrences_stripped():
    v = _verifier()
    answer = "[BNS §999] and again [BNS §999]."
    result = v.verify(answer, [_chunk("bns_2023", "103")])
    assert result.total == 1
    assert "[BNS §999]" not in result.clean_answer
    assert result.clean_answer.count(DISCLAIMER) == 2


def test_mixed_statuses_counts_and_rates():
    v = _verifier()
    answer = (
        "Murder [BNS §103], conspiracy [IPC §120B] (not retrieved), "
        "and a fake [BNS §999]."
    )
    retrieved = [_chunk("bns_2023", "103")]
    result = v.verify(answer, retrieved)
    assert result.total == 3
    assert result.n_verified == 1
    assert result.n_ungrounded == 1  # IPC §120B exists in corpus, not retrieved
    assert result.n_hallucinated == 1  # BNS §999 not in corpus
    assert result.precision == 1 / 3
    assert result.hallucination_rate == 1 / 3
    # Only the verified one survives in the text.
    assert "[BNS §103]" in result.clean_answer
    assert "[IPC §120B]" not in result.clean_answer
    assert "[BNS §999]" not in result.clean_answer


# --- constructors ------------------------------------------------------------

def test_from_chunks_builds_corpus_index():
    chunks = [_chunk("bns_2023", "103"), _chunk("ipc_1860", "302")]
    v = CitationVerifier.from_chunks(chunks)
    # In corpus but not retrieved (empty context) -> ungrounded, not hallucinated.
    result = v.verify("See [BNS §103].", [])
    assert result.citations[0].status is CitationStatus.UNGROUNDED


def test_from_chunks_ignores_sectionless_units():
    chunks = [_chunk("bns_2023", "103"), _chunk("preamble", None)]
    v = CitationVerifier.from_chunks(chunks)
    result = v.verify("[BNS §103]", [_chunk("bns_2023", "103")])
    assert result.citations[0].status is CitationStatus.VERIFIED


def test_from_processed_dir(tmp_path):
    p = tmp_path / "bns_2023.jsonl"
    rows = [
        {"id": "bns_2023:s103", "doc_id": "bns_2023", "section": "103"},
        {"id": "bns_2023:s104", "doc_id": "bns_2023", "section": "104"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    v = CitationVerifier.from_processed_dir(tmp_path)
    # §104 exists in corpus (loaded from disk) but isn't retrieved -> ungrounded.
    result = v.verify("[BNS §104]", [])
    assert result.citations[0].status is CitationStatus.UNGROUNDED
    # §200 was never written -> hallucinated.
    result2 = v.verify("[BNS §200]", [])
    assert result2.citations[0].status is CitationStatus.HALLUCINATED
