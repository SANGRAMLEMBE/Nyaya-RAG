"""Deterministic query router — intent + era in one explainable decision.

Implements the PLAN M2 router (rights / procedure / section / case) wired to
the ADR-003 era filter. Deliberately regex/keyword-based, not a model: routing
must be instant, run on CPU, and — because a wrong-era answer is a critical
bug — every decision must be explainable ("matched 'ipc' → old_code"), which
a classifier cannot promise.

The router answers three questions about a query:
  1. qtype — what kind of help is being asked for (drives index choice and
     answer style; CASE routes to judgments once M2 lands them).
  2. era — which criminal-law era applies (explicit > keyword > default),
     with era_source recording *why*, so the UI can say "assuming current
     law" when the era was only defaulted.
  3. sections — statutory sections named in the query ("u/s 138", "§103",
     "article 21", "IPC 302"), for direct-lookup boosting.

Usage::

    from nyaya.retrieval.router import route
    decision = route("How do I file a complaint u/s 138 of the NI Act?")
    # → qtype=procedure, era=new_code (default), sections=['138']
"""

from __future__ import annotations

import re

from nyaya.schema import Era, QueryType, RouteDecision

# --- era keywords (single source of truth; app.py delegates here) -------------

_OLD_RE = re.compile(
    r"\b(ipc|crpc|iea|indian penal code|criminal procedure|evidence act"
    r"|before\s+2024|before\s+july|pre.?2024|old\s+law|old\s+code)\b",
    re.IGNORECASE,
)
_NEW_RE = re.compile(
    r"\b(bns|bnss|bsa|bharatiya|after\s+2024|after\s+july|post.?2024"
    r"|new\s+law|new\s+code)\b",
    re.IGNORECASE,
)

# --- intent keywords -----------------------------------------------------------

_CASE_RE = re.compile(
    r"\b(judg?ment|case\s+law|precedent|ruling|held\s+that|supreme\s+court"
    r"|high\s+court|landmark\s+case|\bv\.?s?\.\s)\b",
    re.IGNORECASE,
)
_PROCEDURE_RE = re.compile(
    r"\b(how\s+(?:do|can|to|should)\b|procedure|process\s+(?:for|of|to)"
    r"|steps?\s+(?:for|to)|file\s+(?:a|an|the)|filing|register\s+(?:a|an|the)"
    r"|apply\s+for|appeal(?:\s+against)?|where\s+(?:do|can|should)\s+i)\b",
    re.IGNORECASE,
)
_RIGHTS_RE = re.compile(
    r"\b(right\s+to|rights?\s+of|my\s+rights?|fundamental\s+right"
    r"|am\s+i\s+entitled|entitled\s+to|is\s+it\s+legal|can\s+(?:i|they|he|she)"
    r"|freedom\s+of|protection\s+(?:from|of|against))\b",
    re.IGNORECASE,
)

# Section references as Indians actually write them: "section 302", "sec. 420",
# "u/s 138" (under section), "§103", "s. 302", "article 21" / "art. 14".
_SECTION_WORD_RE = re.compile(
    r"(?:\bsections?\b\.?|\bsecs?\b\.?|\bu/s\.?|\bs\.|§|\barticles?\b|\barts?\b\.?)"
    r"\s*(\d+[A-Za-z]{0,2})\b",
    re.IGNORECASE,
)
# Bare "ACT-abbrev NUMBER" style: "IPC 302", "BNS 103", "CrPC 41".
_ACT_NUM_RE = re.compile(
    r"\b(?:ipc|crpc|iea|bns|bnss|bsa)\s+(\d+[A-Za-z]{0,2})\b", re.IGNORECASE
)


def extract_sections(question: str) -> list[str]:
    """Section/article numbers named in the query, normalized, first-seen order."""
    found: list[str] = []
    for m in _SECTION_WORD_RE.finditer(question):
        found.append(m.group(1).upper())
    for m in _ACT_NUM_RE.finditer(question):
        found.append(m.group(1).upper())
    seen: set[str] = set()
    unique = []
    for s in found:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _detect_era(question: str, explicit: str | None) -> tuple[Era, str]:
    """Resolve the era filter and record how it was chosen."""
    if explicit:
        try:
            return Era(explicit), "explicit"
        except ValueError:
            pass  # unknown value from the API — fall through to keywords
    if _OLD_RE.search(question):
        return Era.OLD_CODE, "keyword"
    if _NEW_RE.search(question):
        return Era.NEW_CODE, "keyword"
    return Era.NEW_CODE, "default"  # current law governs unless stated otherwise


def _detect_qtype(question: str, sections: list[str]) -> QueryType:
    """Intent priority: case > procedure > rights > section-only > general.

    A procedure question naming a section ("how do I file u/s 138") is still a
    procedure question — the section list travels alongside, it does not
    override intent. SECTION is only assigned when the section reference is
    the *whole* intent ("what does IPC 302 say?").
    """
    if _CASE_RE.search(question):
        return QueryType.CASE
    if _PROCEDURE_RE.search(question):
        return QueryType.PROCEDURE
    if _RIGHTS_RE.search(question):
        return QueryType.RIGHTS
    if sections:
        return QueryType.SECTION
    return QueryType.GENERAL


def route(question: str, explicit_era: str | None = None) -> RouteDecision:
    """Classify one query: intent, era (with provenance), named sections."""
    sections = extract_sections(question)
    era, era_source = _detect_era(question, explicit_era)
    return RouteDecision(
        qtype=_detect_qtype(question, sections),
        era=era,
        era_source=era_source,
        sections=sections,
    )
