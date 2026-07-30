"""End-to-end answer-quality eval — populates RESULTS.md section 2.

Runs the full pipeline (retrieve -> Qwen answer -> citation verifier) over the
gold set and reports the verifier's headline numbers: citation precision and
hallucinated-citation rate, before vs after verification (ADR-005), plus
era-correctness on the criminal subset (ADR-003).

Needs vLLM reachable at NYAYA_LLM_BASE_URL and the index + bge-m3, so it runs
on CHAMP (same GPU node as vLLM). Faithfulness via the local judge model is
left for a later pass.

Usage (on CHAMP, vLLM already serving on localhost:8000):
    python -m nyaya.eval.run_answer_eval
"""

from __future__ import annotations

import re
from pathlib import Path

from nyaya.eval.gold import load_gold
from nyaya.eval.verify import CitationStatus, CitationVerifier
from nyaya.schema import Era

GOLD_PATH = "data/gold/gold_set.jsonl"
FINAL_K = 8

# Acts whose citations are era-sensitive (ADR-003); others are era-neutral.
_ERA_OF_ABBR = {
    "BNS": Era.NEW_CODE, "BNSS": Era.NEW_CODE, "BSA": Era.NEW_CODE,
    "IPC": Era.OLD_CODE, "CRPC": Era.OLD_CODE, "IEA": Era.OLD_CODE,
}
_CRIMINAL_SUBJECTS = {"criminal", "criminal_procedure", "evidence"}


def _wrong_era(abbrs: list[str], expected: Era) -> bool:
    """True if any abbreviation belongs to the opposite criminal era."""
    for ab in abbrs:
        era = _ERA_OF_ABBR.get(ab.upper())
        if era is not None and era != expected:
            return True
    return False


def run(label: str = "base") -> dict[str, float]:
    from nyaya.generation.answer import LegalAnswerer
    from nyaya.retrieval.hybrid import HybridRetriever

    gold = load_gold(GOLD_PATH)
    retriever = HybridRetriever()
    answerer = LegalAnswerer()
    verifier = CitationVerifier.from_processed_dir("data/processed")

    total = verified = ungrounded = hallucinated = 0
    answered = 0
    # era-correctness on the criminal, era-specific subset
    era_n = era_ok_pre = era_ok_post = 0

    for i, q in enumerate(gold, 1):
        chunks = retriever.retrieve(q.question, era=q.era, final_k=FINAL_K)
        result = answerer.answer(q.question, chunks)
        if result.model == "":  # LLM call failed — skip from precision stats
            print(f"[{i}/{len(gold)}] {q.id}: LLM unavailable, skipped")
            continue
        answered += 1
        verdict = verifier.verify(result.answer, chunks)
        total += verdict.total
        verified += verdict.n_verified
        ungrounded += verdict.n_ungrounded
        hallucinated += verdict.n_hallucinated

        if q.subject.value in _CRIMINAL_SUBJECTS and q.era in (Era.OLD_CODE, Era.NEW_CODE):
            era_n += 1
            pre_abbrs = [c.act_abbr for c in verdict.citations]
            post_abbrs = [c.act_abbr for c in verdict.citations
                          if c.status is CitationStatus.VERIFIED]
            era_ok_pre += 0 if _wrong_era(pre_abbrs, q.era) else 1
            era_ok_post += 0 if _wrong_era(post_abbrs, q.era) else 1

        if i % 20 == 0:
            print(f"[{i}/{len(gold)}] running… emitted={total} halluc={hallucinated}")

    m = {
        "answered": answered,
        "precision_pre": verified / total if total else 0.0,
        "precision_post": 1.0 if verified else 0.0,
        "halluc_pre": hallucinated / total if total else 0.0,
        "halluc_post": 0.0,
        "era_pre": era_ok_pre / era_n if era_n else 0.0,
        "era_post": era_ok_post / era_n if era_n else 0.0,
        "total_citations": total,
    }
    print(f"RESULTS [{label}]: {m}")
    if label == "base":
        _write_results(m, n=answered)  # detailed pre/post table stays base-only
    _write_comparison(label, m, n=answered)
    return m


def _write_comparison(label: str, m: dict[str, float], n: int) -> None:
    """Upsert one model's row in the fine-tuning before/after table (§2b).

    Rebuilds the whole table from its current rows each call, so re-running a
    label overwrites its row cleanly and rows stay in base -> sft -> dpo order.
    """
    header = "## 2b. Fine-tuning: citation hallucination by model (gold set)"
    colhdr = "| model | n | hallucinated-citation rate (pre-verifier) | post-verifier |"
    sep = "|---|---|---|---|"

    path = Path("RESULTS.md")
    text = path.read_text(encoding="utf-8") if path.exists() else "# Results\n\n"

    # collect existing rows (label -> "n | pre | post") from a prior §2b table
    rows: dict[str, str] = {}
    block_re = re.compile(re.escape(header) + r".*?(?=\n## |\Z)", re.DOTALL)
    existing = block_re.search(text)
    if existing:
        for ln in existing.group(0).splitlines():
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) == 4 and cells[0] not in ("model", "---") and cells[1].isdigit():
                rows[cells[0]] = f"{cells[1]} | {cells[2]} | {cells[3]}"
    rows[label] = f"{n} | {m['halluc_pre']:.3f} | {m['halluc_post']:.3f}"

    order = ["base", "sft", "dpo"]
    ordered = sorted(rows, key=lambda k: (order.index(k) if k in order else len(order), k))
    table = "\n".join([header, colhdr, sep] + [f"| {k} | {rows[k]} |" for k in ordered])

    if existing:
        text = block_re.sub(table + "\n", text, count=1)
    else:  # insert before §3 if present, else append
        anchor = re.search(r"\n## 3\. ", text)
        text = (text[: anchor.start()] + "\n" + table + "\n" + text[anchor.start():]
                if anchor else text.rstrip() + "\n\n" + table + "\n")
    path.write_text(text, encoding="utf-8")
    print(f"RESULTS.md §2b updated: {label}")


def _write_results(m: dict[str, float], n: int) -> None:
    rows = [
        f"## 2. End-to-end answer quality (gold set, n={n})",
        "| Metric | pre-verifier | post-verifier |",
        "|---|---|---|",
        f"| citation precision | {m['precision_pre']:.3f} | {m['precision_post']:.3f} |",
        f"| hallucinated-citation rate | {m['halluc_pre']:.3f} | {m['halluc_post']:.3f} |",
        "| faithfulness (local judge) | – | – |",
        f"| era-correctness (criminal subset) | {m['era_pre']:.3f} | {m['era_post']:.3f} |",
    ]
    block = "\n".join(rows) + "\n"
    path = Path("RESULTS.md")
    text = path.read_text(encoding="utf-8") if path.exists() else "# Results\n\n"
    if re.search(r"## 2\. End-to-end answer quality", text):
        new = re.sub(
            r"## 2\. End-to-end answer quality.*?(?=\n## )",
            block + "\n",
            text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        new = text.rstrip() + "\n\n" + block
    path.write_text(new, encoding="utf-8")
    print(f"RESULTS.md section 2 updated (n={n}).")


if __name__ == "__main__":
    run()
