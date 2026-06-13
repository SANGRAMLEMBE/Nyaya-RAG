"""Embeds chunks with BAAI/bge-m3 and indexes them into Qdrant + BM25.

Reads every data/processed/*.jsonl file, embeds each chunk's embed_text()
with bge-m3 (dense 1024-dim vectors), upserts into a local Qdrant collection,
and pickles a rank-bm25 index alongside for hybrid retrieval.

Run this on a machine with a GPU — bge-m3 is ~570 MB and runs fine on
the RTX 3050 (laptop) or A100 (CHAMP).

Usage:
    python -m nyaya.pipelines.index
    python -m nyaya.pipelines.index --batch-size 16   # smaller for low VRAM
    python -m nyaya.pipelines.index --collection nyaya_v1
    python -m nyaya.pipelines.index --reset           # drop and recreate collection
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import uuid
from pathlib import Path

from nyaya.config import settings
from nyaya.schema import Chunk

log = logging.getLogger("nyaya.index")

COLLECTION = "nyaya_v1"
VECTOR_DIM = 1024  # bge-m3 dense output dimension
BATCH_SIZE = 32


def _load_chunks(processed_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for jsonl in sorted(processed_dir.glob("*.jsonl")):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))
    log.info("loaded %d chunks from %s", len(chunks), processed_dir)
    return chunks


def _chunk_id_to_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _embed(
    model: object,
    texts: list[str],
    batch_size: int,
    checkpoint_path: Path,
    start_from: int = 0,
) -> list[list[float]]:
    """Return dense embeddings, saving a checkpoint after every 10 batches for resume."""
    import torch

    all_vecs: list[list[float]] = []

    # Load existing checkpoint if resuming
    if start_from > 0 and checkpoint_path.exists():
        import pickle
        with checkpoint_path.open("rb") as fh:
            all_vecs = pickle.load(fh)
        log.info("resumed from checkpoint: %d vectors already done", len(all_vecs))

    for i in range(start_from, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            vecs = model.encode(  # type: ignore[attr-defined]
                batch,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_vecs.extend(vecs.tolist())
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.warning("OOM at batch %d — retrying one-by-one", i // batch_size)
            for text in batch:
                try:
                    vec = model.encode(  # type: ignore[attr-defined]
                        [text], normalize_embeddings=True
                    )
                    all_vecs.extend(vec.tolist())
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log.error("OOM on single chunk at index %d — skipping", i)
                    all_vecs.append([0.0] * VECTOR_DIM)

        torch.cuda.empty_cache()

        batch_num = (i - start_from) // batch_size
        if batch_num % 10 == 0:
            log.info("embedded %d / %d chunks", min(i + batch_size, len(texts)), len(texts))
            # Save checkpoint
            import pickle
            with checkpoint_path.open("wb") as fh:
                pickle.dump(all_vecs, fh)

    return all_vecs


def build_qdrant(
    chunks: list[Chunk],
    vectors: list[list[float]],
    qdrant_path: Path,
    collection: str,
    reset: bool,
) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(path=str(qdrant_path))

    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        if reset:
            log.info("dropping existing collection %s", collection)
            client.delete_collection(collection)
        else:
            log.info("collection %s already exists — upserting (use --reset to recreate)", collection)

    if collection not in existing or reset:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        log.info("created collection %s (dim=%d, cosine)", collection, VECTOR_DIM)

    points = [
        PointStruct(
            id=_chunk_id_to_uuid(chunk.id),
            vector=vec,
            payload={
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "act": chunk.act,
                "section": chunk.section,
                "chapter": chunk.chapter,
                "era": chunk.era.value,
                "subject": chunk.subject.value,
                "source": chunk.source,
                "text": chunk.text,
            },
        )
        for chunk, vec in zip(chunks, vectors)
    ]

    batch = 256
    for i in range(0, len(points), batch):
        client.upsert(collection_name=collection, points=points[i : i + batch])
        log.info("upserted %d / %d points", min(i + batch, len(points)), len(points))

    info = client.get_collection(collection)
    log.info("collection %s: %d points indexed", collection, info.points_count)


def build_bm25(chunks: list[Chunk], out_dir: Path) -> None:
    from rank_bm25 import BM25Okapi

    tokenized = [chunk.embed_text().lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    index_data = {
        "bm25": bm25,
        "chunk_ids": [c.id for c in chunks],
        "texts": [c.embed_text() for c in chunks],
    }
    out_path = out_dir / "bm25.pkl"
    with out_path.open("wb") as fh:
        pickle.dump(index_data, fh)
    log.info("BM25 index written to %s (%d docs)", out_path, len(chunks))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)  # 8 fits safely in RTX 3050 4GB
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--reset", action="store_true", help="drop and recreate Qdrant collection")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,  # FlagEmbedding sets up its own handlers first; override them
    )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error(
            "sentence-transformers not installed. Run:\n"
            "  pip install 'sentence-transformers==3.0.1'"
        )
        return 1

    chunks = _load_chunks(settings.processed_dir)
    if not chunks:
        log.error("no chunks found in %s — run extract + chunk first", settings.processed_dir)
        return 1

    log.info("loading BAAI/bge-m3 (downloads ~570 MB on first run)…")
    import torch as _torch
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(settings.embedding_model, device=device)
    model.max_seq_length = 512  # truncate long sections to avoid OOM on 4GB VRAM

    texts = [c.embed_text() for c in chunks]
    checkpoint_path = settings.processed_dir / "embed_checkpoint.pkl"
    start_from = 0
    if checkpoint_path.exists() and not args.reset:
        import pickle
        with checkpoint_path.open("rb") as fh:
            existing = pickle.load(fh)
        start_from = len(existing)
        log.info("checkpoint found: resuming from chunk %d / %d", start_from, len(texts))

    log.info("embedding %d chunks with bge-m3…", len(texts))
    vectors = _embed(model, texts, args.batch_size, checkpoint_path, start_from)

    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    build_qdrant(chunks, vectors, settings.qdrant_path, args.collection, args.reset)
    build_bm25(chunks, settings.processed_dir)

    log.info("indexing complete — %d chunks in Qdrant + BM25", len(chunks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
