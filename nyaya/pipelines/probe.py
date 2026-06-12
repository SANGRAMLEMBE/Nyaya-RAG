"""Corpus probe: inspect every downloaded PDF and write a layout report.

Run after `make download`. For each PDF in data/raw/ it reports page count,
text density (digital vs scanned verdict), header/footer candidates
(lines repeated across pages — these become the Day-2 cleaning regexes),
and a short sample of real body text.

The report (docs/probe_report.md) contains everything needed to design the
extraction + section-tree parser without shipping the PDFs anywhere.

Usage:
    python -m nyaya.pipelines.probe
    python -m nyaya.pipelines.probe --only ipc_1860 constitution
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from nyaya.config import REPO_ROOT, settings

log = logging.getLogger("nyaya.probe")

# below this many extracted characters per page, the page is image-only
SCANNED_CHARS_PER_PAGE = 100
# a first/last line counts as a header/footer candidate if it recurs on
# at least this fraction of pages
REPEAT_FRACTION = 0.4
SAMPLE_CHARS = 400


def _normalize_line(line: str) -> str:
    """Collapse digits so 'Page 12' and 'Page 13' cluster together."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip()))


def probe_pdf(path: Path) -> dict[str, Any]:
    doc = pymupdf.open(path)
    page_count = doc.page_count
    chars_per_page: list[int] = []
    first_lines: Counter[str] = Counter()
    last_lines: Counter[str] = Counter()
    sample = ""

    for i, page in enumerate(doc):  # type: ignore[arg-type, var-annotated]
        text = page.get_text("text")
        chars_per_page.append(len(text))
        lines = [ln for ln in (raw.strip() for raw in text.splitlines()) if ln]
        if lines:
            for ln in lines[:2]:
                first_lines[_normalize_line(ln)] += 1
            for ln in lines[-2:]:
                last_lines[_normalize_line(ln)] += 1
        # body sample from page 2 (page 1 is usually a cover / gazette banner)
        if not sample and i >= 1 and len(text) > SCANNED_CHARS_PER_PAGE:
            sample = re.sub(r"\s+", " ", text)[:SAMPLE_CHARS]
    doc.close()

    text_pages = sum(1 for c in chars_per_page if c >= SCANNED_CHARS_PER_PAGE)
    density = text_pages / page_count if page_count else 0.0
    verdict = "digital" if density >= 0.9 else "mixed" if density >= 0.3 else "likely_scanned"

    min_repeats = max(2, int(page_count * REPEAT_FRACTION))
    header_candidates = [ln for ln, n in first_lines.most_common(8) if n >= min_repeats]
    footer_candidates = [ln for ln, n in last_lines.most_common(8) if n >= min_repeats]

    return {
        "pages": page_count,
        "avg_chars_per_page": int(sum(chars_per_page) / page_count) if page_count else 0,
        "pct_text_pages": round(100 * density, 1),
        "verdict": verdict,
        "header_candidates": header_candidates,
        "footer_candidates": footer_candidates,
        "sample": sample or "(no extractable body text found — OCR path needed)",
    }


def render_report(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Corpus probe report",
        "",
        "Paste this whole file back to Claude to design the Day-2 parser.",
        "",
        "| doc | pages | chars/page | text pages | verdict |",
        "|---|---|---|---|---|",
    ]
    for doc_id, r in sorted(results.items()):
        lines.append(
            f"| {doc_id} | {r['pages']} | {r['avg_chars_per_page']} "
            f"| {r['pct_text_pages']}% | **{r['verdict']}** |"
        )
    for doc_id, r in sorted(results.items()):
        lines += [
            "",
            f"## {doc_id}",
            f"- header candidates: {r['header_candidates'] or 'none'}",
            f"- footer candidates: {r['footer_candidates'] or 'none'}",
            f"- body sample: `{r['sample']}`",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these doc ids")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs" / "probe_report.md"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pdfs = sorted(settings.raw_dir.glob("*.pdf"))
    if args.only:
        wanted = set(args.only)
        pdfs = [p for p in pdfs if p.stem in wanted]
    if not pdfs:
        log.error("no PDFs found in %s — run `make download` first", settings.raw_dir)
        return 1

    results: dict[str, dict[str, Any]] = {}
    for pdf in pdfs:
        log.info("probing %s …", pdf.name)
        try:
            results[pdf.stem] = probe_pdf(pdf)
        except Exception as exc:  # noqa: BLE001 — report and continue, never die mid-corpus
            log.error("  failed: %s", exc)
            results[pdf.stem] = {
                "pages": 0, "avg_chars_per_page": 0, "pct_text_pages": 0.0,
                "verdict": f"error: {exc}", "header_candidates": [],
                "footer_candidates": [], "sample": "",
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(results), encoding="utf-8")
    log.info("report written to %s (%d documents)", args.out, len(results))
    for doc_id, r in sorted(results.items()):
        log.info("  %-18s %4s pages  %s", doc_id, r["pages"], r["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
