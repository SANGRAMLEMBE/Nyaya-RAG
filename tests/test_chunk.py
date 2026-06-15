"""Tests for the chunking pipeline — no GPU, no file I/O needed."""

import json
import tempfile
from pathlib import Path

import pytest

from nyaya.pipelines.chunk import _windows, chunk_document
from nyaya.schema import Era, Subject


# --- _windows unit tests -----------------------------------------------------

def test_short_text_returns_single_window():
    text = "This is a short section."
    assert _windows(text) == [text]


def test_long_text_splits_into_multiple_windows():
    # 6 paragraphs of ~500 chars each → should split
    para = "A" * 500
    text = "\n\n".join([para] * 6)
    result = _windows(text)
    assert len(result) > 1


def test_windows_have_overlap():
    # Build text with named paragraphs so we can track carry-over
    paras = [f"Paragraph {i}: " + "word " * 80 for i in range(10)]
    text = "\n\n".join(paras)
    windows = _windows(text)
    # Each window after the first should start with a paragraph that appeared
    # at the end of the previous window (overlap carry-over)
    assert len(windows) >= 2
    # Last paragraph of window[0] should appear in window[1]
    last_para_of_first = windows[0].split("\n\n")[-1]
    assert last_para_of_first in windows[1]


def test_200kb_cap_applied():
    # Use exactly 201KB — just over the 200KB cap. Don't go higher or laptop OOMs.
    huge = "word " * 41_000  # ~205KB
    result = _windows(huge)
    total = sum(len(w) for w in result)
    assert total <= 200_000 + 2_000  # capped + one extra target window


def test_no_paragraph_breaks_uses_sentence_split():
    # One long block with sentences but no double-newlines — must exceed TARGET_CHARS
    sentences = [f"Sentence {i} about the law and its provisions." for i in range(200)]
    text = " ".join(sentences)
    assert len(text) > 2000  # sanity: text must actually exceed TARGET_CHARS
    result = _windows(text)
    assert len(result) >= 2


# --- chunk_document unit tests -----------------------------------------------

def _make_interim(tmp_path: Path, sections: list[dict]) -> Path:
    doc = {
        "id": "test_act",
        "era": Era.NEW_CODE.value,
        "subject": Subject.CRIMINAL.value,
        "sections": sections,
    }
    p = tmp_path / "test_act.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


class _FakeCatalogEntry:
    title = "Test Criminal Act, 2024"
    source = "India Code"


def test_chunk_document_short_sections(tmp_path):
    sections = [
        {"section": "1", "chapter": "Chapter I", "text": "Short section one."},
        {"section": "2", "chapter": "Chapter I", "text": "Short section two."},
    ]
    path = _make_interim(tmp_path, sections)
    chunks = chunk_document(path, _FakeCatalogEntry())
    assert len(chunks) == 2
    assert chunks[0].id == "test_act:s1"
    assert chunks[1].section == "2"
    assert chunks[0].era == Era.NEW_CODE
    assert chunks[0].act == "Test Criminal Act, 2024"


def test_chunk_document_long_section_creates_multiple_chunks(tmp_path):
    long_text = "\n\n".join(["paragraph about the law " * 20] * 8)
    sections = [{"section": "100", "chapter": None, "text": long_text}]
    path = _make_interim(tmp_path, sections)
    chunks = chunk_document(path, _FakeCatalogEntry())
    assert len(chunks) > 1
    # All chunks reference the same section
    for c in chunks:
        assert c.section == "100"
        assert c.id.startswith("test_act:s100w")


def test_chunk_document_empty_sections(tmp_path):
    path = _make_interim(tmp_path, [])
    chunks = chunk_document(path, _FakeCatalogEntry())
    assert chunks == []


def test_chunk_ids_are_unique(tmp_path):
    sections = [{"section": str(i), "chapter": None, "text": "text " * 10} for i in range(20)]
    path = _make_interim(tmp_path, sections)
    chunks = chunk_document(path, _FakeCatalogEntry())
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
