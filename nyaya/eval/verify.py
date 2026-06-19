"""Citation verifier — implements ADR-005 ("Citations are verified, not trusted").

A post-generation check over the model's answer. Every ``[ACT §SECTION]``
citation is parsed and classified:

* ``VERIFIED``    — the cited section appeared in the *retrieved context*
                   (passes ADR-005 checks 1 *and* 2).
* ``UNGROUNDED``  — the section exists in the corpus but was *not* in the
                   retrieved context (fails check 2).
* ``HALLUCINATED`` — the section does not exist in the corpus at all
                   (fails check 1).

Unverifiable citations (``UNGROUNDED`` or ``HALLUCINATED``) are stripped from
the answer text and replaced with a disclaimer. The pre- and post-verifier
hallucinated-citation rates feed RESULTS.md section 2.

Act abbreviations are derived mechanically from the ``doc_id`` prefix
(``bns_2023`` → ``BNS``). Acts whose conventional abbreviation differs from
that prefix (e.g. the IT Act) can be supplied via the ``aliases`` map —
mapping a citation abbreviation to the corpus prefix — rather than guessed.

Usage::

    from nyaya.eval.verify import CitationVerifier
    verifier = CitationVerifier.from_processed_dir(settings.processed_dir)
    result = verifier.verify(answer_text, retrieved_chunks)
    print(result.clean_answer)          # unverifiable citations stripped
    print(result.hallucination_rate)    # for RESULTS.md
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

# Replacement text for a citation that fails verification (ADR-005).
DISCLAIMER = "I could not verify this in my sources"

# [BNS §103] / [IPC §120B] / [CrPC §41] / [BNS §303(1)]
# group 1 = act abbreviation, group 2 = section number (with optional letter
# suffix). A trailing sub-section like "(1)" or "(a)" is consumed but ignored,
# since the corpus is indexed at section granularity (ADR-004: one section =
# one chunk).
CITATION_RE = re.compile(
    r"\[\s*([A-Za-z]{2,6})\s+§\s*(\d+[A-Z]*)(?:\([0-9A-Za-z]+\))?\s*\]"
)


class CitationStatus(StrEnum):
    VERIFIED = "verified"
    UNGROUNDED = "ungrounded"
    HALLUCINATED = "hallucinated"


class _HasSection(Protocol):
    """Minimal shape the verifier needs from a retrieved unit (a schema.Chunk)."""

    doc_id: str
    section: str | None


@dataclass(frozen=True)
class Citation:
    """One parsed citation and its verification outcome."""

    raw: str  # the matched text, e.g. "[BNS §103]"
    act_abbr: str  # normalised upper-case abbreviation, e.g. "BNS"
    section: str  # normalised section, e.g. "103" or "120B"
    status: CitationStatus

    @property
    def verified(self) -> bool:
        return self.status is CitationStatus.VERIFIED


@dataclass
class VerificationResult:
    original_answer: str
    clean_answer: str
    citations: list[Citation] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.citations)

    @property
    def n_verified(self) -> int:
        return sum(c.status is CitationStatus.VERIFIED for c in self.citations)

    @property
    def n_ungrounded(self) -> int:
        return sum(c.status is CitationStatus.UNGROUNDED for c in self.citations)

    @property
    def n_hallucinated(self) -> int:
        return sum(c.status is CitationStatus.HALLUCINATED for c in self.citations)

    @property
    def precision(self) -> float:
        """Share of citations that are grounded in the retrieved context."""
        return self.n_verified / self.total if self.total else 1.0

    @property
    def hallucination_rate(self) -> float:
        """Share of citations that point at sections absent from the corpus."""
        return self.n_hallucinated / self.total if self.total else 0.0


def _abbr(doc_id: str) -> str:
    """Mechanical abbreviation from a corpus doc_id: 'bns_2023' -> 'BNS'."""
    return doc_id.split("_")[0].upper()


def _norm_section(section: str) -> str:
    return section.strip().upper()


class CitationVerifier:
    """Checks model citations against the corpus and the retrieved context (ADR-005)."""

    def __init__(
        self,
        corpus_sections: set[tuple[str, str]],
        aliases: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            corpus_sections: every ``(ABBR, SECTION)`` pair present in the corpus.
            aliases: optional map from a *citation* abbreviation to the corpus
                abbreviation, for acts whose conventional short form differs
                from the doc_id prefix (e.g. ``{"ITA": "IT"}``). Keys/values are
                upper-cased on use.
        """
        self._corpus = corpus_sections
        self._aliases = {k.upper(): v.upper() for k, v in (aliases or {}).items()}

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[_HasSection],
        aliases: dict[str, str] | None = None,
    ) -> CitationVerifier:
        corpus = {
            (_abbr(c.doc_id), _norm_section(c.section))
            for c in chunks
            if c.section
        }
        return cls(corpus, aliases=aliases)

    @classmethod
    def from_processed_dir(
        cls,
        processed_dir: Path | str,
        aliases: dict[str, str] | None = None,
    ) -> CitationVerifier:
        """Build the corpus index from ``data/processed/*.jsonl`` chunk files."""
        corpus: set[tuple[str, str]] = set()
        for path in Path(processed_dir).glob("*.jsonl"):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    section = rec.get("section")
                    doc_id = rec.get("doc_id")
                    if doc_id and section:
                        corpus.add((_abbr(doc_id), _norm_section(str(section))))
        return cls(corpus, aliases=aliases)

    # -- verification ---------------------------------------------------------

    def _canon(self, abbr: str) -> str:
        up = abbr.upper()
        return self._aliases.get(up, up)

    def _classify(
        self, abbr: str, section: str, context: set[tuple[str, str]]
    ) -> CitationStatus:
        pair = (self._canon(abbr), _norm_section(section))
        if pair in context:
            return CitationStatus.VERIFIED
        if pair in self._corpus:
            return CitationStatus.UNGROUNDED
        return CitationStatus.HALLUCINATED

    def verify(
        self, answer_text: str, retrieved_chunks: Iterable[_HasSection]
    ) -> VerificationResult:
        """Classify every citation and strip the unverifiable ones from the text."""
        context = {
            (_abbr(c.doc_id), _norm_section(c.section))
            for c in retrieved_chunks
            if c.section
        }

        # Collect unique citations (dedup by abbr+section, like the answerer).
        seen: set[tuple[str, str]] = set()
        citations: list[Citation] = []
        for m in CITATION_RE.finditer(answer_text):
            abbr = m.group(1).upper()
            section = _norm_section(m.group(2))
            key = (self._canon(abbr), section)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    raw=m.group(0),
                    act_abbr=abbr,
                    section=section,
                    status=self._classify(abbr, section, context),
                )
            )

        # Strip every textual occurrence of an unverifiable citation.
        def _sub(m: re.Match[str]) -> str:
            status = self._classify(m.group(1), m.group(2), context)
            return m.group(0) if status is CitationStatus.VERIFIED else DISCLAIMER

        clean = CITATION_RE.sub(_sub, answer_text)

        return VerificationResult(
            original_answer=answer_text,
            clean_answer=clean,
            citations=citations,
        )
