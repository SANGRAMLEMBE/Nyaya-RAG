"""Probe tests against a synthetic PDF built in-memory — no real corpus needed."""

from pathlib import Path

import pymupdf

from nyaya.pipelines.probe import probe_pdf, render_report


def _make_pdf(path: Path, pages: int = 6) -> None:
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 50), "THE GAZETTE OF INDIA EXTRAORDINARY")  # header
        body = (
            f"Section {i + 1}. Whoever, being bound by law to do a thing, "
            "omits to do that thing, shall be punished in the manner provided. " * 6
        )
        page.insert_text((72, 120), body, fontsize=10)
        page.insert_text((290, 800), f"Page {i + 1}")  # footer with page number
    doc.save(path)
    doc.close()


def test_probe_detects_digital_pdf_and_repeated_lines(tmp_path: Path) -> None:
    pdf = tmp_path / "fake_act.pdf"
    _make_pdf(pdf)
    r = probe_pdf(pdf)
    assert r["pages"] == 6
    assert r["verdict"] == "digital"
    assert r["pct_text_pages"] == 100.0
    assert any("GAZETTE" in h for h in r["header_candidates"])
    assert "Page #" in r["footer_candidates"]
    assert "Section" in r["sample"]


def test_probe_flags_scanned_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    for _ in range(4):
        doc.new_page()  # empty pages: zero extractable text, like a pure scan
    doc.save(pdf)
    doc.close()
    r = probe_pdf(pdf)
    assert r["verdict"] == "likely_scanned"
    assert "OCR" in r["sample"]


def test_render_report_contains_all_docs(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    _make_pdf(pdf, pages=4)
    report = render_report({"x": probe_pdf(pdf)})
    assert "## x" in report and "digital" in report
