"""Ingest fetched judgments → chunks in data/processed/judgments.jsonl.

Reads the raw judgment records (data/raw/judgments/*.json from the Indian
Kanoon fetcher), de-duplicates them, parses each into paragraph-window Chunks
(era=neutral — precedent persists across the 2024 transition), and writes them
alongside the statute chunks so the indexer picks them up automatically.

The clean case title from the fetch record is used as the chunk's act label,
and the subject comes from the catalog — neither is guessed. Everything else
(paragraph windowing) is the same code path statutes use.

Run (laptop; then scp data/processed/judgments.jsonl to CHAMP and re-index)::

    python -m nyaya.pipelines.ingest_judgments
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from nyaya.config import settings
from nyaya.pipelines.dedup import dedup
from nyaya.pipelines.judgments import judgment_to_chunks, parse_judgment
from nyaya.schema import Chunk, Subject

log = logging.getLogger("nyaya.pipelines")

RAW_DIR = "data/raw/judgments"
OUT_NAME = "judgments.jsonl"


def load_records(raw_dir: Path | str) -> list[dict]:
    """Load judgment json records (skips the acts downloader's .meta.json)."""
    records = []
    for path in sorted(Path(raw_dir).glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("text"):
            records.append(rec)
    return records


def record_to_chunks(rec: dict) -> list[Chunk]:
    """One fetch record → paragraph-window Chunks (era=neutral)."""
    j = parse_judgment(
        rec["text"],
        doc_id=rec["id"],
        source=rec.get("source", "Indian Kanoon"),
        url=rec.get("source_url"),
    )
    # trust the catalog's clean case name over whatever the header parse found
    title = rec.get("ik_title") or rec.get("catalog_title") or rec["id"]
    j = j.model_copy(update={"title": title})
    return judgment_to_chunks(j, subject=Subject(rec["subject"]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    records = load_records(RAW_DIR)
    if not records:
        log.warning("no judgment records in %s — fetch them first", RAW_DIR)
        return

    # de-duplicate whole judgments (source order = preference; here id order)
    result = dedup([(r["id"], r["text"]) for r in records])
    if result.drop:
        log.info("dedup dropped %d near-duplicate judgments: %s",
                 result.n_duplicates, result.drop)
    kept = [r for r in records if r["id"] in set(result.keep)]

    all_chunks: list[Chunk] = []
    for rec in kept:
        chunks = record_to_chunks(rec)
        all_chunks.extend(chunks)
        log.info("[%s] %d chunks", rec["id"], len(chunks))

    out = settings.processed_dir / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(c.model_dump_json() + "\n")
    log.info("wrote %d judgment chunks from %d judgments -> %s",
             len(all_chunks), len(kept), out)


if __name__ == "__main__":
    main()
