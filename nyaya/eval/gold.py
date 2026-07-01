"""Gold evaluation set — questions with known-correct statute sections.

Each gold question names the section(s) that correctly answer it, keyed by
``<doc_id>:<section>`` (e.g. ``bns_2023:103``). Section granularity — not
chunk id — so it is robust to section windowing (``bns_2023:s303w0`` etc.).

The set is built ONLY from sections that exist in the corpus; ``validate_gold``
enforces this so no fabricated citation can slip in (ADR-005).
Public per ADR-007 (the gold eval set ships in the repo for research credibility).

Usage::

    from nyaya.eval.gold import load_gold, load_corpus_sections, validate_gold
    gold = load_gold("data/gold/gold_set.jsonl")
    corpus = load_corpus_sections("data/processed")
    problems = validate_gold(gold, corpus)   # [] means every answer is grounded
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from nyaya.schema import Era, Subject

# question categories — drive the query router work in M2 (ADR-003)
QTYPES = {"section_lookup", "cross_era", "rights", "procedure", "definition"}


class GoldQuestion(BaseModel):
    """One evaluation question with its known-correct answer section(s)."""

    id: str = Field(pattern=r"^gold_\d{3,}$")
    question: str = Field(min_length=10, max_length=500)
    era: Era
    subject: Subject
    qtype: str = Field(description="one of QTYPES")
    relevant: list[str] = Field(
        min_length=1,
        description="section keys '<doc_id>:<section>' that answer the question",
    )
    note: str = ""

    def relevant_pairs(self) -> list[tuple[str, str]]:
        """Split each 'doc_id:section' key into (doc_id, section)."""
        pairs: list[tuple[str, str]] = []
        for key in self.relevant:
            doc_id, _, section = key.partition(":")
            pairs.append((doc_id, section))
        return pairs


def load_gold(path: Path | str) -> list[GoldQuestion]:
    """Load and parse a gold-set JSONL file."""
    questions: list[GoldQuestion] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            questions.append(GoldQuestion.model_validate_json(line))
    return questions


def load_corpus_sections(processed_dir: Path | str) -> set[tuple[str, str]]:
    """Every ``(doc_id, section)`` pair present in ``data/processed/*.jsonl``."""
    corpus: set[tuple[str, str]] = set()
    for path in Path(processed_dir).glob("*.jsonl"):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                doc_id, section = rec.get("doc_id"), rec.get("section")
                if doc_id and section:
                    corpus.add((doc_id, str(section)))
    return corpus


def validate_gold(
    questions: list[GoldQuestion], corpus: set[tuple[str, str]]
) -> list[str]:
    """Return a list of problems; empty means every gold answer is grounded.

    Checks (1) ids unique, (2) qtype valid, (3) every relevant section exists
    in the corpus.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if q.qtype not in QTYPES:
            problems.append(f"{q.id}: unknown qtype {q.qtype!r}")
        for doc_id, section in q.relevant_pairs():
            if (doc_id, section) not in corpus:
                problems.append(
                    f"{q.id}: relevant section {doc_id}:{section} not in corpus"
                )
    return problems
