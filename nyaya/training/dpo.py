"""DPO on preference pairs manufactured by the citation verifier (M4 headline).

The preference signal is mechanical, not opined (PLAN week 7: "hallucinated vs
corrected answers"): run the model over questions it was never trained on,
verify every answer's citations (ADR-005), and wherever an unverifiable
citation appears —

    rejected = the model's raw answer (contains the fabricated citation)
    chosen   = the verifier-corrected answer (same content, fabrication
               stripped and disclaimed)

so DPO pushes probability mass away from emitting unverifiable citations at
all. No human labels, no judge model, no invented preferences.

Contamination hygiene: harvest questions come from the *validation bucket* of
the deterministic SFT split (never trained on) — the gold eval set is never
touched by training.

Three subcommands, in pipeline order::

    python -m nyaya.training.dpo harvest   # GPU node + vLLM: collect answers
    python -m nyaya.training.dpo pairs     # anywhere: records -> DPO pairs
    python -m nyaya.training.dpo train     # GPU node: DPO over the pairs

Outputs live in data/synthetic/ and models/ — both gitignored (ADR-007).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from nyaya.config import settings
from nyaya.generation.answer import _SYSTEM_PROMPT, _format_context
from nyaya.schema import Chunk
from nyaya.training.sft import load_chunk_map, load_pairs, split_rows

log = logging.getLogger("nyaya.training")

RECORDS_PATH = "data/synthetic/dpo_records.jsonl"
PAIRS_PATH = "data/synthetic/dpo_pairs.jsonl"
SFT_ADAPTER = "models/sft-qwen2.5-14b/final"
OUTPUT_DIR = "models/dpo-qwen2.5-14b"


# --- pair building (pure python, unit-tested) ------------------------------------


def _prompt_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    """The production prompt shape — identical to inference and SFT."""
    context = _format_context(chunks)
    user = f"STATUTE SECTIONS:\n{context}\n\nQUESTION: {question}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_preference_pairs(
    records: list[dict], chunk_map: dict[str, Chunk]
) -> tuple[list[dict], dict[str, int]]:
    """Harvest records → DPO triples. Only genuine failures become pairs.

    A record yields a pair iff its answer contained at least one unverifiable
    citation AND the corrected text actually differs — everything else is
    skipped with a counted reason (clean answers are not preferences).
    """
    stats = {"pairs": 0, "clean": 0, "unchanged": 0, "missing_chunks": 0}
    pairs: list[dict] = []
    for rec in records:
        if (rec.get("n_hallucinated", 0) + rec.get("n_ungrounded", 0)) == 0:
            stats["clean"] += 1
            continue
        raw, corrected = rec.get("answer", ""), rec.get("clean_answer", "")
        if not raw or not corrected or raw == corrected:
            stats["unchanged"] += 1
            continue
        chunks = [chunk_map[cid] for cid in rec.get("chunk_ids", []) if cid in chunk_map]
        if not chunks:
            stats["missing_chunks"] += 1
            continue
        pairs.append(
            {
                "prompt": _prompt_messages(rec["question"], chunks),
                "chosen": [{"role": "assistant", "content": corrected}],
                "rejected": [{"role": "assistant", "content": raw}],
            }
        )
    stats["pairs"] = len(pairs)
    return pairs, stats


def load_records(path: Path | str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# --- harvest (needs index + vLLM; runs on CHAMP) ----------------------------------


def harvest(out_path: str, pairs_path: str, limit: int | None) -> None:
    """Run the pipeline over SFT-val questions; record answer + verification."""
    from nyaya.eval.verify import CitationVerifier
    from nyaya.generation.answer import LegalAnswerer
    from nyaya.retrieval.hybrid import HybridRetriever

    rows = [r for r in load_pairs(pairs_path) if r.get("kind") == "qa"]
    _, val_rows = split_rows(rows)  # same deterministic split SFT used
    if limit:
        val_rows = val_rows[:limit]
    log.info("harvesting over %d val questions (never trained on)", len(val_rows))

    retriever = HybridRetriever()
    answerer = LegalAnswerer()
    verifier = CitationVerifier.from_processed_dir(settings.processed_dir)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r.get("question") for r in load_records(out)} if out.exists() else set()

    n_fail = 0
    with out.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(val_rows, 1):
            q = row["question"]
            if q in done:
                continue
            chunks = retriever.retrieve(q, era=row.get("era"), final_k=8)
            result = answerer.answer(q, chunks)
            if result.model == "":
                log.warning("[%d] LLM unavailable — skipped", i)
                continue
            verdict = verifier.verify(result.answer, chunks)
            n_fail += 1 if (verdict.n_hallucinated + verdict.n_ungrounded) else 0
            fh.write(
                json.dumps(
                    {
                        "question": q,
                        "era": row.get("era"),
                        "chunk_ids": [c.id for c in chunks],
                        "answer": result.answer,
                        "clean_answer": verdict.clean_answer,
                        "n_hallucinated": verdict.n_hallucinated,
                        "n_ungrounded": verdict.n_ungrounded,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()
            if i % 25 == 0:
                log.info("[%d/%d] failures so far: %d", i, len(val_rows), n_fail)
    log.info("harvest done — %d records with unverifiable citations", n_fail)


# --- train (GPU only — imports guarded) --------------------------------------------


def train(pairs_file: str, sft_adapter: str, output_dir: str, epochs: float) -> None:
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    pairs = load_records(pairs_file)
    log.info("training on %d preference pairs", len(pairs))

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        settings.generation_model,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # continue from the SFT adapter — DPO refines the fine-tuned policy
    model = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(settings.generation_model)

    out = Path(output_dir)
    resume = any(out.glob("checkpoint-*")) if out.exists() else False

    config = DPOConfig(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,  # DPO wants a much gentler LR than SFT
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=0.1,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        max_length=2048,
        max_prompt_length=1536,
        report_to="none",  # fully local (ADR-001)
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(pairs),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out / "final"))
    log.info("DPO adapter saved to %s/final", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="collect model answers + verification")
    h.add_argument("--out", default=RECORDS_PATH)
    h.add_argument("--pairs", default="data/synthetic/sft_pairs.jsonl")
    h.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("pairs", help="records -> DPO preference pairs (no GPU)")
    p.add_argument("--records", default=RECORDS_PATH)
    p.add_argument("--out", default=PAIRS_PATH)

    t = sub.add_parser("train", help="DPO over the pairs (GPU)")
    t.add_argument("--pairs-file", default=PAIRS_PATH)
    t.add_argument("--sft-adapter", default=SFT_ADAPTER)
    t.add_argument("--output-dir", default=OUTPUT_DIR)
    t.add_argument("--epochs", type=float, default=1.0)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.cmd == "harvest":
        harvest(args.out, args.pairs, args.limit)
    elif args.cmd == "pairs":
        records = load_records(args.records)
        chunk_map = load_chunk_map(settings.processed_dir)
        pairs, stats = build_preference_pairs(records, chunk_map)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for pair in pairs:
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
        log.info("wrote %d pairs -> %s  stats=%s", len(pairs), out, stats)
    elif args.cmd == "train":
        train(args.pairs_file, args.sft_adapter, args.output_dir, args.epochs)


if __name__ == "__main__":
    main()
