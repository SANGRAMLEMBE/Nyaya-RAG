"""Tests for the judgments catalog + its loader — no network."""

import pytest

from nyaya.pipelines.download_judgments import load_catalog
from nyaya.schema import JudgmentCatalogEntry, Subject


def test_shipped_catalog_parses_and_ids_unique():
    entries = load_catalog()
    assert len(entries) >= 15  # the landmark checklist
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids))


def test_every_entry_has_valid_subject_and_slug_id():
    for e in load_catalog():
        assert isinstance(e.subject, Subject)
        assert e.id.replace("_", "").isalnum()


def test_https_url_enforced():
    with pytest.raises(ValueError):
        JudgmentCatalogEntry(
            id="sc_bad", title="X v. Y", subject=Subject.CRIMINAL,
            url="http://insecure.example/doc.pdf",
        )


def test_url_optional_until_filled():
    e = JudgmentCatalogEntry(id="sc_tbd", title="X v. Y", subject=Subject.CRIMINAL)
    assert e.url is None  # a not-yet-filled entry is valid; downloader skips it


def test_loader_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text(
        "judgments:\n"
        "  - id: sc_x\n    title: A v. B\n    subject: criminal\n"
        "  - id: sc_x\n    title: C v. D\n    subject: criminal\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_catalog(p)
