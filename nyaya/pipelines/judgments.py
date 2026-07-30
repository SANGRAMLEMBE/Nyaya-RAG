"""Judgment parser — raw SC judgment text → Judgment → paragraph-window Chunks.

Source-agnostic (works on text from Indian Kanoon, e-SCR, or PDF extraction):
the parser only reads what is printed. Extraction is strictly conservative —
a field the text does not state comes back None/empty, never guessed
(project rule: unverifiable → flag, never fabricate).

What gets extracted (PLAN M2: title / citation / bench / held):
  title    — "X v. Y" from the header, or PETITIONER/RESPONDENT blocks
  citation — a reporter citation printed in the header (AIR / SCC / INSC)
  date     — "DATE OF JUDGMENT" / "decided on" style lines
  bench    — the BENCH: block, names as printed
  held     — indexes of paragraphs containing holding language ("we hold…")

Era: judgment chunks carry Era.NEUTRAL — precedent persists across the 2024
transition (an IPC-era ruling still guides BNS interpretation), so filtering
judgments by criminal era would wrongly hide valid case law. The judgment
date is preserved in metadata for future, finer-grained treatment.

Usage::

    from nyaya.pipelines.judgments import parse_judgment, judgment_to_chunks
    j = parse_judgment(raw_text, doc_id="sc_1973_kesavananda", source="e-SCR")
    chunks = judgment_to_chunks(j, subject=Subject.CONSTITUTIONAL)
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime

from nyaya.schema import Chunk, Era, Judgment, Subject

log = logging.getLogger("nyaya.pipelines")

# paragraph-window sizing (chars) — same ballpark as statute chunking
TARGET_CHARS = 1800
MAX_WINDOW_CHARS = 2600

# --- extraction patterns --------------------------------------------------------

_VERSUS_RE = re.compile(
    r"^(.{3,120}?)\s+(?:v(?:s)?\.?|versus)\s+(.{3,120}?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_PETITIONER_RE = re.compile(r"PETITIONER[S]?\s*:\s*\n?(.+)", re.IGNORECASE)
_RESPONDENT_RE = re.compile(r"RESPONDENT[S]?\s*:\s*\n?(.+)", re.IGNORECASE)

_CITATION_RES = (
    re.compile(r"\bAIR\s+\d{4}\s+SC\s+\d+\b"),
    re.compile(r"\(\d{4}\)\s+\d+\s+SCC\s+\d+\b"),
    re.compile(r"\b\d{4}\s+INSC\s+\d+\b"),
)

_DATE_RE = re.compile(
    r"(?:DATE\s+OF\s+JUDGMENT|decided\s+on|dated)\s*:?\s*"
    r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})",
    re.IGNORECASE,
)

_BENCH_RE = re.compile(r"BENCH\s*:\s*\n?((?:.+\n?)+?)(?:\n\s*\n|$)", re.IGNORECASE)

_HELD_RE = re.compile(
    r"\b(?:we\s+hold|held\s+that|it\s+is\s+held|hereby\s+held"
    r"|we\s+are\s+of\s+the\s+(?:considered\s+)?view)\b",
    re.IGNORECASE,
)


def _extract_title(header: str) -> str | None:
    """Title from 'X v. Y' or PETITIONER/RESPONDENT blocks — or None."""
    pet = _PETITIONER_RE.search(header)
    res = _RESPONDENT_RE.search(header)
    if pet and res:
        return f"{pet.group(1).strip()} v. {res.group(1).strip()}"
    m = _VERSUS_RE.search(header)
    if m:
        return f"{m.group(1).strip()} v. {m.group(2).strip()}"
    return None


def _extract_citation(header: str) -> str | None:
    for rx in _CITATION_RES:
        m = rx.search(header)
        if m:
            return m.group(0)
    return None


def _extract_date(text: str) -> date | None:
    m = _DATE_RE.search(text)
    if not m:
        return None
    d, mo, y = (int(g) for g in m.groups())
    if y < 100:  # two-digit year as printed in some older records
        y += 1900 if y > 40 else 2000
    try:
        return datetime(y, mo, d).date()
    except ValueError:
        return None  # impossible date printed — do not guess


def _extract_bench(header: str) -> list[str]:
    m = _BENCH_RE.search(header)
    if not m:
        return []
    names = []
    for line in m.group(1).splitlines():
        for part in line.split(","):
            name = part.strip().strip(".").strip()
            if len(name) >= 3:
                names.append(name)
    return names


def _split_paragraphs(text: str) -> list[str]:
    """Blocks separated by blank lines, whitespace-normalized, empties dropped."""
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    paras = []
    for b in blocks:
        p = " ".join(b.split())
        if p:
            paras.append(p)
    return paras


def parse_judgment(
    text: str, doc_id: str, source: str, url: str | None = None
) -> Judgment:
    """Parse raw judgment text. Fields absent from the text come back empty."""
    header = text[:4000]  # metadata lives at the top of SC judgment records
    paragraphs = _split_paragraphs(text)
    held = [i for i, p in enumerate(paragraphs) if _HELD_RE.search(p)]

    j = Judgment(
        id=doc_id,
        title=_extract_title(header),
        citation=_extract_citation(header),
        judgment_date=_extract_date(header),
        bench=_extract_bench(header),
        paragraphs=paragraphs,
        held_paras=held,
        source=source,
        url=url,
    )
    if j.title is None:
        log.warning("[%s] no title found — flagged, not fabricated", doc_id)
    return j


def judgment_to_chunks(j: Judgment, subject: Subject) -> list[Chunk]:
    """Group paragraphs into ~TARGET_CHARS windows (1-paragraph overlap).

    The caller supplies `subject` — inferring it is future work (e.g. from the
    statutes a judgment cites); defaulting one silently would fabricate metadata.
    """
    if not j.paragraphs:
        return []

    windows: list[list[str]] = []
    current: list[str] = []
    size = 0
    for p in j.paragraphs:
        p = p[:MAX_WINDOW_CHARS]  # a pathological single paragraph cannot blow a window
        if current and size + len(p) > TARGET_CHARS:
            windows.append(current)
            current = [current[-1]] if len(current[-1]) < TARGET_CHARS else []
            size = sum(len(x) for x in current)
        current.append(p)
        size += len(p)
    if current:
        windows.append(current)

    act_label = j.title or j.id  # printed title if we have one; id otherwise
    chunks = []
    for w_idx, window in enumerate(windows):
        chunks.append(
            Chunk(
                id=f"{j.id}:p{w_idx:03d}",
                text="\n\n".join(window),
                doc_id=j.id,
                act=act_label,
                section=None,
                chapter=None,
                era=Era.NEUTRAL,  # precedent persists across the 2024 transition
                subject=subject,
                source=j.source,
            )
        )
    return chunks
