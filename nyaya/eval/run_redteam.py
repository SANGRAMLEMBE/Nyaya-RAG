"""Red-team runner — executes the adversarial prompts, writes RESULTS.md §3.

Runs each RedTeamPrompt through the real pipeline (retrieve -> answer ->
verify), scores it, and writes the section-3 table. Needs vLLM + the index,
so it runs on CHAMP via scripts/answer_eval-style batching. The scoring logic
lives in nyaya/eval/redteam.py and is unit-tested separately.

Usage (on CHAMP, vLLM serving)::

    python -m nyaya.eval.run_redteam
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from nyaya.config import settings
from nyaya.eval.redteam import PROMPTS, score_response, summarize

log = logging.getLogger("nyaya.redteam")

_LABELS = {
    "fabricated_citation": "fabricated-citation pressure",
    "guaranteed_outcome": "guaranteed-outcome promises",
    "illegal_assistance": "illegal-act assistance",
    "advocate_impersonation": "advocate impersonation",
}


def run() -> dict[str, dict]:
    from nyaya.eval.verify import CitationVerifier
    from nyaya.generation.answer import LegalAnswerer
    from nyaya.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever()
    answerer = LegalAnswerer()
    verifier = CitationVerifier.from_processed_dir(settings.processed_dir)

    results = []
    for i, p in enumerate(PROMPTS, 1):
        chunks = retriever.retrieve(p.prompt, final_k=8, rerank=True)
        result = answerer.answer(p.prompt, chunks)
        if result.model == "":
            log.warning("[%s] LLM unavailable — skipped", p.id)
            continue
        verdict = verifier.verify(result.answer, chunks)
        passed = score_response(
            p.category, verdict.clean_answer, verdict.n_hallucinated, verdict.n_ungrounded
        )
        results.append({"id": p.id, "category": p.category, "passed": passed})
        log.info("[%d/%d] %s %s -> %s", i, len(PROMPTS), p.id, p.category,
                 "PASS" if passed else "FAIL")

    summary = summarize(results)
    log.info("RED-TEAM SUMMARY: %s", summary)
    _write_results(summary, n=len(results))
    return summary


def _write_results(summary: dict[str, dict], n: int) -> None:
    rows = [
        f"## 3. Red-team (n={n})",
        "| Category | pass rate |",
        "|---|---|",
    ]
    for cat, label in _LABELS.items():
        s = summary.get(cat)
        cell = f"{s['passed']}/{s['n']} ({s['pass_rate']:.0%})" if s else "–"
        rows.append(f"| {label} | {cell} |")
    block = "\n".join(rows) + "\n"

    path = Path("RESULTS.md")
    text = path.read_text(encoding="utf-8") if path.exists() else "# Results\n\n"
    if re.search(r"## 3\. Red-team", text):
        # section 3 is last — replace to end of file
        new = re.sub(r"## 3\. Red-team.*", block, text, count=1, flags=re.DOTALL)
    else:
        new = text.rstrip() + "\n\n" + block
    path.write_text(new, encoding="utf-8")
    log.info("RESULTS.md section 3 updated (n=%d)", n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
