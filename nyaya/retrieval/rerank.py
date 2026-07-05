"""Cross-encoder reranker — the precision stage after hybrid retrieval (ADR-006).

Why this exists (measured, RESULTS.md section 1): the hybrid RRF merge has the
best recall pool (recall@10 = 0.953) but fusing BM25 dilutes ranking precision
(MRR 0.755 vs 0.920 for dense alone). A cross-encoder reads the query and the
chunk *together* and scores their relevance jointly, so it can restore ranking
precision over the high-recall pool — recall from fusion, precision from the
reranker.

Model: BAAI/bge-reranker-v2-m3 (ADR-006), loaded lazily via
sentence-transformers ``CrossEncoder``. On CHAMP the model must be pre-cached
on the login node (proxy set) before GPU jobs run with HF_HUB_OFFLINE=1::

    python -c "from huggingface_hub import snapshot_download; \
               snapshot_download('BAAI/bge-reranker-v2-m3')"

Chunks are scored on ``chunk.embed_text()`` (metadata header + text) so the
reranker sees act, section, and era — the same view the dense encoder indexed.

Usage::

    from nyaya.retrieval.rerank import Reranker
    reranker = Reranker()
    top = reranker.rerank("punishment for murder", chunks, top_k=8)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from nyaya.config import settings
from nyaya.schema import Chunk

log = logging.getLogger("nyaya.retrieval")

# A scorer maps (query, passage) pairs to relevance scores. Injectable so unit
# tests run without the model; production lazily loads the CrossEncoder.
Scorer = Callable[[list[tuple[str, str]]], Sequence[float]]


class Reranker:
    """Scores (query, chunk) pairs with a cross-encoder and reorders chunks."""

    def __init__(
        self,
        model_name: str | None = None,
        max_length: int = 1024,
        scorer: Scorer | None = None,
    ) -> None:
        """
        Args:
            model_name: HF id of the cross-encoder (default: settings.reranker_model).
            max_length: token cap per (query, passage) pair — statute chunks are
                ~500 tokens, so 1024 covers query + chunk without waste.
            scorer: optional scoring function replacing the model (tests only).
        """
        self._model_name = model_name or settings.reranker_model
        self._max_length = max_length
        self._scorer = scorer
        self._model = None  # lazy — loading costs ~2 GB; only pay when reranking

    def _score(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        if self._scorer is not None:
            return self._scorer(pairs)
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading reranker %s…", self._model_name)
            self._model = CrossEncoder(self._model_name, max_length=self._max_length)
        return self._model.predict(pairs)

    def rerank(
        self, query: str, chunks: list[Chunk], top_k: int | None = None
    ) -> list[Chunk]:
        """Reorder chunks by cross-encoder relevance to the query.

        Returns the top_k most relevant chunks (all, if top_k is None). Ties
        keep their incoming (RRF) order, so the fusion ranking acts as the
        tiebreak rather than arbitrary index order.
        """
        if not chunks:
            return []
        pairs = [(query, c.embed_text()) for c in chunks]
        scores = self._score(pairs)
        order = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))
        ranked = [chunks[i] for i in order]
        return ranked[:top_k] if top_k is not None else ranked
