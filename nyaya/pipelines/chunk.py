"""Splits extracted sections into fixed-size Chunk objects for embedding.

Reads data/interim/{id}.json (from extract.py), splits long sections into
overlapping windows, and writes data/processed/{id}.jsonl — one Chunk per line.

Chunking rules:
  - Sections under TARGET_CHARS are kept whole (one chunk = one section).
  - Longer sections are split on paragraph boundaries first, then by character
    window if a paragraph is still too long.
  - Overlap of OVERLAP_CHARS carries context across window boundaries.
  - Each chunk id: "{doc_id}:s{section}" or "{doc_id}:s{section}w{n}" for windows.

Usage:
    python -m nyaya.pipelines.chunk
    python -m nyaya.pipelines.chunk --only bns_2023 ipc_1860
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from nyaya.config import settings
from nyaya.pipelines.download import load_catalog
from nyaya.schema import Chunk, Era, License, Subject

log = logging.getLogger("nyaya.chunk")

TARGET_CHARS = 2_000   # ~512 tokens for bge-m3 (1 token ≈ 4 chars)
OVERLAP_CHARS = 200    # carry-over between windows


def _windows(text: str, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping windows, preferring paragraph breaks."""
    if len(text) <= target:
        return [text]

    # Guard against pathologically large sections (bad PDF extraction artefacts).
    # 200 KB of text in one section is impossible in a real statute — truncate it.
    max_section_chars = 200_000
    if len(text) > max_section_chars:
        text = text[:max_section_chars]

    # Try paragraph-level splits first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        # No paragraph breaks — sentence-split on ". " instead
        sentences = text.split(". ")
        if len(sentences) > 1:
            paragraphs = [s + "." for s in sentences if s]
        else:
            # Last resort: hard character windows (shouldn't happen for statute text)
            windows: list[str] = []
            start = 0
            while start < len(text):
                end = min(start + target, len(text))
                windows.append(text[start:end])
                start = end - overlap
            return windows

    # Accumulate paragraphs into target-sized windows
    windows = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > target and current:
            windows.append("\n\n".join(current))
            # Keep the last paragraph for overlap
            current = current[-1:]
            current_len = len(current[0])
        current.append(para)
        current_len += len(para)

    if current:
        windows.append("\n\n".join(current))

    return windows


def chunk_document(interim_path: Path, catalog_entry: object) -> list[Chunk]:
    with interim_path.open(encoding="utf-8") as fh:
        doc = json.load(fh)

    doc_id: str = doc["id"]
    era = Era(doc["era"])
    subject = Subject(doc["subject"])

    # act title comes from catalog; fall back to doc_id if not available
    act_title: str | None = getattr(catalog_entry, "title", None)

    chunks: list[Chunk] = []
    for sec in doc["sections"]:
        section_num: str = sec["section"]
        chapter: str | None = sec.get("chapter")
        text: str = sec["text"]

        try:
            windows = _windows(text)
        except (MemoryError, Exception) as exc:
            log.warning(
                "[%s] s%s: windowing failed (%s) — skipping section",
                doc_id, section_num, exc,
            )
            continue
        for w_idx, window in enumerate(windows):
            chunk_id = (
                f"{doc_id}:s{section_num}"
                if len(windows) == 1
                else f"{doc_id}:s{section_num}w{w_idx}"
            )
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=window,
                    doc_id=doc_id,
                    act=act_title,
                    section=section_num,
                    chapter=chapter,
                    era=era,
                    subject=subject,
                    source=getattr(catalog_entry, "source", "India Code"),
                    license=License.PUBLIC,
                    extraction_confidence=1.0,
                )
            )
    return chunks


def chunk_one(doc_id: str, interim_dir: Path, processed_dir: Path, catalog_entry: object) -> int:
    interim_path = interim_dir / f"{doc_id}.json"
    if not interim_path.exists():
        log.warning("[%s] interim file not found — run extract first", doc_id)
        return 0

    chunks = chunk_document(interim_path, catalog_entry)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{doc_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")

    log.info("[%s] %d chunks → %s", doc_id, len(chunks), out_path)
    return len(chunks)


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

    total = 0
    fail = 0
    for entry in entries:
        try:
            n = chunk_one(entry.id, settings.interim_dir, settings.processed_dir, entry)
            total += n
        except Exception as exc:
            log.error("[%s] chunking failed: %s", entry.id, exc, exc_info=True)
            fail += 1

    log.info("done: %d chunks total, %d acts failed", total, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
