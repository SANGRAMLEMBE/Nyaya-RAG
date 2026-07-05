"""QLoRA SFT on Qwen2.5-14B — citation discipline, grounding, refusals (M3).

What the model is taught (PLAN week 6): to USE provided statute context and
cite it — never legal facts from memory. Every training example therefore
reconstructs the exact production input: the same system prompt and the same
``STATUTE SECTIONS:`` context block that nyaya/generation/answer.py sends at
inference time, with the pair's verified source section as context. Refusal
examples carry no context — question → standard refusal.

Train/serve consistency is deliberate: a model fine-tuned on a different
prompt shape than production would un-learn at deploy time.

Data preparation (this module's pure part) is unit-tested on CPU; the training
loop imports torch/trl/peft only inside ``main()`` so CI never needs them.

One-time setup on CHAMP login node (proxy set) — pinned for torch 2.5.1+cu121
and transformers 4.47.1 already on the venv:

    pip install "trl==0.12.2" "peft==0.14.0" "bitsandbytes==0.45.0" \
                "datasets==3.2.0" "accelerate==1.2.1"

Run (GPU node, via scripts/sft_train.pbs — resumes from the last checkpoint
automatically when resubmitted after a walltime kill)::

    python -m nyaya.training.sft --pairs data/synthetic/sft_pairs.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from nyaya.config import settings
from nyaya.generation.answer import _SYSTEM_PROMPT, _format_context
from nyaya.schema import Chunk

log = logging.getLogger("nyaya.training")

OUTPUT_DIR = "models/sft-qwen2.5-14b"
VAL_PCT = 5  # deterministic id-hash split — reproducible across runs/machines


# --- data preparation (pure python, unit-tested) --------------------------------


def load_pairs(path: Path | str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_chunk_map(processed_dir: Path | str) -> dict[str, Chunk]:
    """chunk id → Chunk for joining pairs back to their source sections."""
    chunks: dict[str, Chunk] = {}
    for path in Path(processed_dir).glob("*.jsonl"):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    c = Chunk.model_validate(json.loads(line))
                    chunks[c.id] = c
    return chunks


def build_example(row: dict, chunk_map: dict[str, Chunk]) -> dict | None:
    """One synthetic pair → chat messages mirroring the production prompt.

    QA rows whose source chunk cannot be found are dropped (returning None):
    training on context we cannot reproduce would break the grounding
    guarantee the verifier gate established at generation time.
    """
    if row.get("kind") == "refusal":
        return {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ]
        }

    chunk = chunk_map.get(row.get("source_chunk_id") or "")
    if chunk is None:
        return None
    context = _format_context([chunk])
    user = f"STATUTE SECTIONS:\n{context}\n\nQUESTION: {row['question']}"
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["answer"]},
        ]
    }


def _bucket(row_id: str) -> int:
    """Stable 0-99 bucket from the row id (blake2b — not the salted hash())."""
    digest = hashlib.blake2b(row_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % 100


def split_rows(rows: list[dict], val_pct: int = VAL_PCT) -> tuple[list[dict], list[dict]]:
    """Deterministic train/val split keyed on row id — same split every run."""
    train, val = [], []
    for row in rows:
        (val if _bucket(str(row.get("id"))) < val_pct else train).append(row)
    return train, val


def prepare_dataset(
    pairs_path: Path | str, processed_dir: Path | str
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Pairs JSONL → (train_examples, val_examples, stats)."""
    rows = load_pairs(pairs_path)
    chunk_map = load_chunk_map(processed_dir)
    train_rows, val_rows = split_rows(rows)

    stats = {"total": len(rows), "dropped_missing_chunk": 0}
    train, val = [], []
    for src, dst in ((train_rows, train), (val_rows, val)):
        for row in src:
            ex = build_example(row, chunk_map)
            if ex is None:
                stats["dropped_missing_chunk"] += 1
            else:
                dst.append(ex)
    stats["train"], stats["val"] = len(train), len(val)
    return train, val, stats


# --- training loop (GPU only — imports guarded) ----------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="data/synthetic/sft_pairs.jsonl")
    ap.add_argument("--base-model", default=settings.generation_model)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument(
        "--prepare-only", action="store_true",
        help="build + count examples without loading any GPU dependency",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    train, val, stats = prepare_dataset(args.pairs, settings.processed_dir)
    log.info("dataset: %s", stats)
    if args.prepare_only:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    quant = BitsAndBytesConfig(  # QLoRA: 4-bit NF4 base, bf16 compute
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    out = Path(args.output_dir)
    resume = any(out.glob("checkpoint-*")) if out.exists() else False
    if resume:
        log.info("checkpoint found in %s — resuming", out)

    config = SFTConfig(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        save_steps=200,  # walltime insurance: never lose more than 200 steps
        save_total_limit=3,
        eval_strategy="epoch",
        max_seq_length=args.max_seq_length,
        packing=False,
        report_to="none",  # fully local — no external trackers (ADR-001)
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(val),
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out / "final"))
    log.info("adapter saved to %s/final", out)


if __name__ == "__main__":
    main()
