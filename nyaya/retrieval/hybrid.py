"""Hybrid retrieval: dense (bge-m3 + Qdrant) + sparse (BM25) fused via RRF.

Query flow:
  1. Embed query with bge-m3 → dense search in Qdrant (top-K)
  2. Tokenize query → BM25 keyword search (top-K)
  3. Apply era filter to both result sets
  4. Merge with Reciprocal Rank Fusion (no weight tuning needed)
  5. Return top rerank_k chunks

Era filtering:
  - old_code queries  → return old_code + neutral chunks
  - new_code queries  → return new_code + neutral chunks
  - neutral / unknown → return all chunks

Usage:
    from nyaya.retrieval.hybrid import HybridRetriever
    retriever = HybridRetriever()
    chunks = retriever.retrieve("What is the punishment for murder?", era="new_code")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from nyaya.config import settings
from nyaya.schema import Chunk, Era

log = logging.getLogger("nyaya.retrieval")


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a result at position `rank` (0-indexed)."""
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    """Loads Qdrant + BM25 once and serves multiple queries."""

    def __init__(
        self,
        qdrant_path: Path | None = None,
        bm25_path: Path | None = None,
        collection: str = "nyaya_v1",
        embedding_model: str | None = None,
    ) -> None:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        self._qdrant = QdrantClient(
            path=str(qdrant_path or settings.qdrant_path)
        )
        self._collection = collection

        log.info("loading embedding model for query encoding…")
        self._embedder = SentenceTransformer(
            embedding_model or settings.embedding_model,
            device="cuda" if self._cuda_available() else "cpu",
        )
        self._embedder.max_seq_length = 512

        bm25_file = bm25_path or (settings.processed_dir / "bm25.pkl")
        with bm25_file.open("rb") as fh:
            data = pickle.load(fh)  # noqa: S301 — our own BM25 index, trusted input
        self._bm25 = data["bm25"]
        self._bm25_ids: list[str] = data["chunk_ids"]

        self._reranker = None  # lazy — created on the first rerank=True call
        log.info("retriever ready (%d BM25 docs)", len(self._bm25_ids))

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _era_filter(self, era: Era | str | None) -> list[str] | None:
        """Return the list of era values a query should match, or None for no filter."""
        if era is None:
            return None
        era_val = era.value if isinstance(era, Era) else era
        if era_val == Era.OLD_CODE.value:
            return [Era.OLD_CODE.value, Era.NEUTRAL.value]
        if era_val == Era.NEW_CODE.value:
            return [Era.NEW_CODE.value, Era.NEUTRAL.value]
        return None  # neutral query → no filter

    def _dense_search(
        self, query_vec: list[float], top_k: int, era_values: list[str] | None
    ) -> list[str]:
        """Return ordered list of chunk_ids from Qdrant dense search."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        query_filter = None
        if era_values:
            query_filter = Filter(
                must=[FieldCondition(key="era", match=MatchAny(any=era_values))]
            )

        # qdrant-client >= 1.9: search() replaced by query_points()
        result = self._qdrant.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=top_k,
            query_filter=query_filter,
        )
        return [p.payload["chunk_id"] for p in result.points]  # type: ignore[index]

    def _bm25_search(
        self, query: str, top_k: int, era_values: list[str] | None
    ) -> list[str]:
        """Return ordered list of chunk_ids from BM25 keyword search."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Pair (chunk_id, score) and sort descending
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        results: list[str] = []
        for idx, score in ranked:
            if score <= 0:
                break
            chunk_id = self._bm25_ids[idx]
            # Era filtering for BM25: look up from Qdrant payload
            # (fast enough since we filter after top-K)
            results.append(chunk_id)
            if len(results) >= top_k * 3:  # over-fetch then filter
                break

        return results[:top_k]

    def _rrf_merge(
        self,
        dense_ids: list[str],
        bm25_ids: list[str],
        top_n: int,
    ) -> list[str]:
        """Combine two ranked lists with Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        for rank, cid in enumerate(dense_ids):
            scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank)
        for rank, cid in enumerate(bm25_ids):
            scores[cid] = scores.get(cid, 0.0) + _rrf_score(rank)

        return sorted(scores, key=lambda c: scores[c], reverse=True)[:top_n]

    def _fetch_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Retrieve full Chunk objects from Qdrant payloads."""
        import uuid

        uuids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, cid)) for cid in chunk_ids]
        points = self._qdrant.retrieve(
            collection_name=self._collection,
            ids=uuids,
            with_payload=True,
        )
        # Preserve RRF order
        by_chunk_id = {p.payload["chunk_id"]: p.payload for p in points}  # type: ignore[index]
        chunks: list[Chunk] = []
        for cid in chunk_ids:
            if cid not in by_chunk_id:
                continue
            pl = by_chunk_id[cid]
            chunks.append(
                Chunk(
                    id=pl["chunk_id"],
                    text=pl["text"],
                    doc_id=pl["doc_id"],
                    act=pl.get("act"),
                    section=pl.get("section"),
                    chapter=pl.get("chapter"),
                    era=Era(pl["era"]),
                    subject=pl["subject"],
                    source=pl["source"],
                )
            )
        return chunks

    def retrieve(
        self,
        query: str,
        era: Era | str | None = None,
        dense_k: int | None = None,
        bm25_k: int | None = None,
        final_k: int | None = None,
        rerank: bool = False,
    ) -> list[Chunk]:
        """Run hybrid retrieval and return up to `final_k` chunks.

        With ``rerank=False`` the RRF-fused order is returned. With
        ``rerank=True`` a larger candidate pool is fetched and re-scored by the
        cross-encoder (ADR-006) — recall from fusion, precision from the
        reranker.

        Args:
            query:   Natural language question.
            era:     "old_code", "new_code", "neutral", or None (no filter).
            dense_k: Qdrant candidates per query (default: settings.dense_top_k).
            bm25_k:  BM25 candidates per query (default: settings.bm25_top_k).
            final_k: Chunks returned (default: settings.rerank_top_k).
            rerank:  Re-score the fused pool with the cross-encoder.
        """
        dense_k = dense_k or settings.dense_top_k
        bm25_k = bm25_k or settings.bm25_top_k
        final_k = final_k or settings.rerank_top_k

        era_values = self._era_filter(era)

        query_vec: list[float] = self._embedder.encode(
            [query], normalize_embeddings=True
        ).tolist()[0]

        dense_ids = self._dense_search(query_vec, dense_k, era_values)
        bm25_ids = self._bm25_search(query, bm25_k, era_values)

        # Over-fetch to account for era filtering after BM25 (BM25 has no native
        # era filter); reranking wants an even deeper pool to re-score.
        pool_k = max(final_k * 3, settings.rerank_pool_k) if rerank else final_k * 3
        merged_ids = self._rrf_merge(dense_ids, bm25_ids, pool_k)
        chunks = self._fetch_chunks(merged_ids)
        if era_values:
            chunks = [c for c in chunks if c.era.value in era_values]

        if rerank and chunks:
            if self._reranker is None:
                from nyaya.retrieval.rerank import Reranker

                self._reranker = Reranker()
            return self._reranker.rerank(query, chunks, top_k=final_k)
        return chunks[:final_k]
