"""Tests for the judgment parser + chunker — synthetic fixtures, no real cases."""

from datetime import date

from nyaya.pipelines.judgments import judgment_to_chunks, parse_judgment
from nyaya.schema import Era, Subject

# Obviously-synthetic SC-format record (year 9999 citation; placeholder names).
SC_FORMAT = """PETITIONER:
Test Appellant

RESPONDENT:
State of Testland

DATE OF JUDGMENT: 15/03/1995

BENCH:
A. Test Judge, B. Sample Judge

CITATION:
AIR 9999 SC 9999

The appellant challenged the order of the test tribunal on three grounds
that concern the interpretation of the placeholder provisions.

The tribunal had examined the record at length and recorded findings of
fact that were not seriously disputed before us during the hearing.

We hold that the appeal must fail on the first ground, as the provision
admits of no other construction on its plain language.
"""


def test_title_from_petitioner_respondent_blocks():
    j = parse_judgment(SC_FORMAT, doc_id="sc_1995_test", source="test")
    assert j.title == "Test Appellant v. State of Testland"


def test_title_from_versus_line():
    text = "Sample Person v. Union of Testland\n\nJudgment text follows here."
    j = parse_judgment(text, doc_id="sc_2001_sample", source="test")
    assert j.title == "Sample Person v. Union of Testland"


def test_citation_extracted_as_printed():
    j = parse_judgment(SC_FORMAT, doc_id="sc_1995_test", source="test")
    assert j.citation == "AIR 9999 SC 9999"


def test_date_parsed():
    j = parse_judgment(SC_FORMAT, doc_id="sc_1995_test", source="test")
    assert j.judgment_date == date(1995, 3, 15)


def test_bench_names_as_printed():
    j = parse_judgment(SC_FORMAT, doc_id="sc_1995_test", source="test")
    assert "A. Test Judge" in j.bench
    assert "B. Sample Judge" in j.bench


def test_held_paragraph_indexed():
    j = parse_judgment(SC_FORMAT, doc_id="sc_1995_test", source="test")
    assert j.held_paras, "holding language paragraph should be flagged"
    held_text = j.paragraphs[j.held_paras[0]]
    assert "We hold that" in held_text


def test_missing_metadata_stays_empty_not_fabricated():
    j = parse_judgment("Some plain paragraph.", doc_id="sc_0000_bare", source="test")
    assert j.title is None
    assert j.citation is None
    assert j.judgment_date is None
    assert j.bench == []


def test_impossible_printed_date_returns_none():
    text = "DATE OF JUDGMENT: 32/13/1995\n\nBody paragraph."
    j = parse_judgment(text, doc_id="sc_1995_bad", source="test")
    assert j.judgment_date is None  # never guess


# --- chunking -------------------------------------------------------------------

def test_chunks_neutral_era_and_id_format():
    j = parse_judgment(SC_FORMAT, doc_id="sc_1995_test", source="test")
    chunks = judgment_to_chunks(j, subject=Subject.CONSTITUTIONAL)
    assert chunks
    assert chunks[0].id == "sc_1995_test:p000"
    assert all(c.era is Era.NEUTRAL for c in chunks)
    assert all(c.subject is Subject.CONSTITUTIONAL for c in chunks)
    assert all(c.doc_id == "sc_1995_test" for c in chunks)


def test_long_judgment_splits_into_windows():
    paras = "\n\n".join(f"Paragraph {i}: " + ("finding of fact " * 30) for i in range(20))
    j = parse_judgment(paras, doc_id="sc_2000_long", source="test")
    chunks = judgment_to_chunks(j, subject=Subject.CRIMINAL)
    assert len(chunks) > 1
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))  # unique window ids


def test_empty_judgment_yields_no_chunks():
    j = parse_judgment("", doc_id="sc_0000_empty", source="test")
    assert judgment_to_chunks(j, subject=Subject.CRIMINAL) == []
