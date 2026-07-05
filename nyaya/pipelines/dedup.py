"""Near-duplicate detection for the judgment corpus — bottom-k MinHash (PLAN M2).

Why: the same SC judgment appears on multiple police/court mirrors with minor
OCR and formatting differences. Indexing duplicates skews retrieval (the same
precedent crowds the top-k) and would contaminate future training data.

Design, deliberately boring:
  * word 5-gram shingles over normalized text
  * bottom-k sketch (k smallest blake2b hashes) — ONE deterministic hash pass
    per shingle; blake2b, not Python's salted hash(), so runs reproduce
  * candidate pairs via an inverted index on sketch values (near-duplicates
    necessarily share small hashes) — avoids O(n²) over the corpus
  * candidates verified with the standard bottom-k Jaccard estimator
  * pure stdlib — no numpy/datasketch; runs identically in CI and on CHAMP

Usage::

    from nyaya.pipelines.dedup import dedup
    result = dedup([("id1", text1), ("id2", text2), ...], threshold=0.85)
    result.keep   # ids to index (first occurrence of each group wins)
    result.drop   # {duplicate_id: kept_id_it_matches}
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

SHINGLE_WORDS = 5
SKETCH_K = 200

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def _shingles(text: str, n: int = SHINGLE_WORDS) -> set[str]:
    words = _normalize(text).split(" ")
    if len(words) < n:
        return {" ".join(words)} if words != [""] else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _h64(shingle: str) -> int:
    """Deterministic 64-bit hash (blake2b) — stable across processes and runs."""
    return int.from_bytes(
        hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big"
    )


def _sketch(text: str, k: int = SKETCH_K) -> list[int]:
    """The k smallest shingle hashes, ascending. Fewer if the doc is short."""
    hashes = sorted(_h64(s) for s in _shingles(text))
    return hashes[:k]


def _jaccard_bottom_k(a: list[int], b: list[int], k: int = SKETCH_K) -> float:
    """Standard bottom-k estimator: J ≈ |X ∩ A ∩ B| / |X|, X = bottom-k(A ∪ B)."""
    if not a or not b:
        return 0.0
    union_bottom = sorted(set(a) | set(b))[:k]
    x = set(union_bottom)
    inter = x & set(a) & set(b)
    return len(inter) / len(x) if x else 0.0


@dataclass
class DedupResult:
    keep: list[str] = field(default_factory=list)
    drop: dict[str, str] = field(default_factory=dict)  # duplicate id -> kept id

    @property
    def n_duplicates(self) -> int:
        return len(self.drop)


def dedup(
    docs: list[tuple[str, str]], threshold: float = 0.85, k: int = SKETCH_K
) -> DedupResult:
    """Drop near-duplicates, keeping the first occurrence (input order is rank).

    Args:
        docs: (doc_id, text) pairs; earlier entries win ties, so order the
              input by source preference (e.g. e-SCR before mirrors).
        threshold: bottom-k Jaccard estimate at or above which a later doc is
              treated as a duplicate of an earlier one.
        k: sketch size (bigger = finer estimate, slower).
    """
    result = DedupResult()
    sketches: dict[str, list[int]] = {}
    bucket: dict[int, list[str]] = {}  # sketch value -> kept doc ids sharing it

    for doc_id, text in docs:
        sketch = _sketch(text, k)
        # candidates: previously-kept docs sharing at least one sketch value
        candidates: list[str] = []
        seen: set[str] = set()
        for value in sketch:
            for other in bucket.get(value, ()):
                if other not in seen:
                    seen.add(other)
                    candidates.append(other)

        match: str | None = None
        best = threshold
        for other in candidates:
            sim = _jaccard_bottom_k(sketch, sketches[other], k)
            if sim >= best:
                best = sim
                match = other

        if match is not None:
            result.drop[doc_id] = match
            continue

        result.keep.append(doc_id)
        sketches[doc_id] = sketch
        for value in sketch:
            bucket.setdefault(value, []).append(doc_id)

    return result
