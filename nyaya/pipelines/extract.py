"""Extracts text from raw PDFs and splits into sections, writing data/interim/{id}.json.

Reads every catalog entry that has a PDF on disk, strips headers/footers,
detects chapter and section/article boundaries, and writes one JSON file per act.

Output schema (data/interim/{id}.json):
    {
      "id": "bns_2023",
      "era": "new_code",
      "subject": "criminal",
      "sections": [
        {"section": "1", "chapter": null, "text": "1. Short title..."},
        ...
      ]
    }

Usage:
    python -m nyaya.pipelines.extract
    python -m nyaya.pipelines.extract --only bns_2023 ipc_1860
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import fitz  # type: ignore[import-untyped]

from nyaya.config import settings
from nyaya.pipelines.download import load_catalog
from nyaya.schema import CatalogEntry

log = logging.getLogger("nyaya.extract")

# Matches the start of a numbered section or article.
# Covers: "1.", "45A.", "Article 12.", "12A." at the start of a line.
_SECTION_START = re.compile(
    r"(?m)^[ \t]*(?:Article[ \t]+)?(\d+[A-Z]*)\.\s+\S",
)

# Chapter headings: "CHAPTER I", "CHAPTER IV", "CHAPTER 3"
_CHAPTER_HEADING = re.compile(
    r"(?m)^[ \t]*CHAPTER[ \t]+([IVXLCDM]+|\d+)(?:[ \t]+(.+?))?[ \t]*$",
    re.IGNORECASE,
)

# Lone page-number lines that headers/footers leave behind
_PAGE_NUM = re.compile(r"(?m)^\s*\d{1,4}\s*$")

# The dashed separator lines India Code uses in old acts (IPC, IEA)
_DASH_LINE = re.compile(r"-{20,}")

# Minimum body length (chars) to keep a section — filters TOC stubs
_MIN_SECTION_CHARS = 120

# Amendment-history footnotes in old India Code PDFs are numbered like real sections.
# Their bodies always open with one of these phrases — skip them.
_FOOTNOTE_OPEN = re.compile(
    r"^\s*\d+[A-Z]*\.\s+"
    r"(?:Subs\.|Ins\.|Rep\.|Omitted|Added|See\s|Now\s|Re-?numbered|"
    r"The\s+(?:words?|expression|Act\s+has\s+been|section|proviso)|"
    r"Substituted|Inserted|Renumbered|Extended|Enforced)",
    re.IGNORECASE,
)


def _clean(raw: str) -> str:
    raw = _PAGE_NUM.sub("", raw)
    raw = _DASH_LINE.sub("", raw)
    # collapse horizontal whitespace runs (two-column PDF artefacts)
    raw = re.sub(r"[ \t]{2,}", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))  # type: ignore[arg-type]
    parts: list[str] = []
    for page in doc:  # type: ignore[arg-type, var-annotated]
        text: str = page.get_text()
        if len(text.strip()) > 80:  # skip near-blank / scanned pages
            parts.append(text)
    return "\n".join(parts)


def _chapter_at(text: str, pos: int) -> str | None:
    """Return the most recent chapter heading before position `pos`."""
    best: str | None = None
    for m in _CHAPTER_HEADING.finditer(text):
        if m.start() >= pos:
            break
        num = m.group(1).upper()
        title = (m.group(2) or "").strip()
        best = f"CHAPTER {num}" + (f" {title}" if title else "")
    return best


def _split_sections(text: str) -> list[dict[str, object]]:
    matches = list(_SECTION_START.finditer(text))
    if not matches:
        log.warning("no section boundaries detected")
        return []

    seen: set[str] = set()
    sections: list[dict[str, object]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start() : end].strip()

        # Drop TOC stubs (title only, no body)
        if len(body) < _MIN_SECTION_CHARS:
            continue

        # Drop amendment-history footnotes (old India Code PDFs)
        if _FOOTNOTE_OPEN.match(body):
            continue

        sec_num = m.group(1)

        # First occurrence wins — footnotes appear after the real section in page flow
        if sec_num in seen:
            continue
        seen.add(sec_num)

        sections.append(
            {
                "section": sec_num,
                "chapter": _chapter_at(text, m.start()),
                "text": body,
            }
        )
    return sections


def extract_one(entry: CatalogEntry, raw_dir: Path, out_dir: Path) -> bool:
    pdf_path = raw_dir / f"{entry.id}.pdf"
    if not pdf_path.exists():
        log.warning("[%s] PDF not found at %s — skipping", entry.id, pdf_path)
        return False

    log.info("[%s] extracting %s", entry.id, pdf_path.name)
    raw_text = _pdf_text(pdf_path)
    text = _clean(raw_text)

    sections = _split_sections(text)
    log.info("[%s] %d sections extracted", entry.id, len(sections))

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": entry.id,
        "era": entry.era.value,
        "subject": entry.subject.value,
        "sections": sections,
    }
    out_path = out_dir / f"{entry.id}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("[%s] written to %s", entry.id, out_path)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    entries = load_catalog(settings.catalog_path)
    if args.only:
        entries = [e for e in entries if e.id in set(args.only)]

    entries = [e for e in entries if e.url is not None]

    ok = fail = 0
    for entry in entries:
        try:
            ok += int(extract_one(entry, settings.raw_dir, settings.interim_dir))
        except Exception as exc:
            log.error("[%s] extraction failed: %s", entry.id, exc, exc_info=True)
            fail += 1

    log.info("done: %d ok, %d failed", ok, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
