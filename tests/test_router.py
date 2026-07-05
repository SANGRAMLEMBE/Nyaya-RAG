"""Tests for the deterministic query router — pure logic, no GPU, no index."""

from nyaya.retrieval.router import extract_sections, route
from nyaya.schema import Era, QueryType

# --- era resolution ------------------------------------------------------------

def test_explicit_era_wins_over_keywords():
    d = route("What does the IPC say about murder?", explicit_era="new_code")
    assert d.era is Era.NEW_CODE
    assert d.era_source == "explicit"


def test_old_code_keyword_detected():
    d = route("What is the punishment for murder under the IPC?")
    assert d.era is Era.OLD_CODE
    assert d.era_source == "keyword"


def test_new_code_keyword_detected():
    d = route("What does the BNS say about theft?")
    assert d.era is Era.NEW_CODE
    assert d.era_source == "keyword"


def test_no_signal_defaults_to_current_law():
    d = route("What is the punishment for cheating?")
    assert d.era is Era.NEW_CODE
    assert d.era_source == "default"  # UI can disclose "assuming current law"


def test_invalid_explicit_era_falls_back_to_keywords():
    d = route("What does the IPC provide for theft?", explicit_era="garbage")
    assert d.era is Era.OLD_CODE
    assert d.era_source == "keyword"


# --- intent classification -------------------------------------------------------

def test_procedure_intent():
    d = route("How do I file a complaint against a builder?")
    assert d.qtype is QueryType.PROCEDURE


def test_rights_intent():
    d = route("What is the right to freedom of speech in India?")
    assert d.qtype is QueryType.RIGHTS


def test_case_intent():
    d = route("What did the Supreme Court hold about privacy as a precedent?")
    assert d.qtype is QueryType.CASE


def test_section_intent_when_section_is_whole_point():
    d = route("What does section 302 say?")
    assert d.qtype is QueryType.SECTION
    assert d.sections == ["302"]


def test_general_fallback():
    d = route("Tell me about theft in India.")
    assert d.qtype is QueryType.GENERAL


def test_procedure_with_section_stays_procedure():
    """Naming a section doesn't override intent — the section travels alongside."""
    d = route("How do I file a cheque-bounce case u/s 138?")
    assert d.qtype is QueryType.PROCEDURE
    assert "138" in d.sections


# --- section extraction ----------------------------------------------------------

def test_extract_word_forms():
    assert extract_sections("under section 302 of the code") == ["302"]
    assert extract_sections("see sec. 420 for cheating") == ["420"]
    assert extract_sections("punishable u/s 138") == ["138"]
    assert extract_sections("guaranteed by article 21") == ["21"]


def test_extract_act_number_style():
    assert extract_sections("What does IPC 302 say?") == ["302"]
    assert extract_sections("Explain BNS 103 to me") == ["103"]


def test_extract_letter_suffix_and_dedup():
    got = extract_sections("Section 120B and section 120b of the IPC")
    assert got == ["120B"]  # normalized upper + deduplicated


def test_extract_multiple_in_order():
    got = extract_sections("Compare section 302 with article 21 and u/s 138")
    assert got == ["302", "21", "138"]


def test_no_sections_in_plain_text():
    assert extract_sections("What is the punishment for murder?") == []
