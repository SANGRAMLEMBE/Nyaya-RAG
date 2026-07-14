"""Tests for judgment ingestion — synthetic records, no network."""

import json

from nyaya.pipelines.ingest_judgments import load_records, record_to_chunks
from nyaya.schema import Era, Subject

RECORD = {
    "id": "sc_test_1999",
    "catalog_title": "Test Appellant v. State of Testland",
    "ik_title": "Test Appellant vs State Of Testland on 1 January, 1999",
    "subject": "constitutional",
    "source": "Indian Kanoon (API)",
    "source_url": "https://indiankanoon.org/doc/999/",
    "text": "\n\n".join(f"Paragraph {i} discussing the constitutional question at length "
                        + "and its implications " * 8 for i in range(12)),
}


def test_record_to_chunks_neutral_era_and_metadata():
    chunks = record_to_chunks(RECORD)
    assert chunks
    assert all(c.era is Era.NEUTRAL for c in chunks)  # precedent is era-neutral
    assert all(c.subject is Subject.CONSTITUTIONAL for c in chunks)
    assert all(c.doc_id == "sc_test_1999" for c in chunks)
    # clean catalog title wins as the act label, not the raw header parse
    assert all(c.act == RECORD["ik_title"] for c in chunks)
    assert chunks[0].id.startswith("sc_test_1999:p")


def test_long_judgment_splits_into_multiple_chunks():
    chunks = record_to_chunks(RECORD)
    assert len(chunks) > 1
    assert len({c.id for c in chunks}) == len(chunks)  # unique ids


def test_load_records_skips_meta_json(tmp_path):
    (tmp_path / "sc_a.json").write_text(json.dumps(RECORD), encoding="utf-8")
    (tmp_path / "sc_b.meta.json").write_text(json.dumps({"id": "x"}), encoding="utf-8")
    (tmp_path / "empty.json").write_text(json.dumps({"id": "y", "text": ""}),
                                         encoding="utf-8")
    recs = load_records(tmp_path)
    assert [r["id"] for r in recs] == ["sc_test_1999"]  # meta + empty skipped
