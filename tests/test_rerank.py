"""Tests for the cross-encoder reranker — injected scorer, no model, no GPU."""

from nyaya.retrieval.rerank import Reranker
from nyaya.schema import Chunk, Era, Subject


def _chunk(cid: str, text: str, section: str | None = None) -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        doc_id=cid.split(":")[0],
        act="Test Act, 2024",
        section=section,
        era=Era.NEW_CODE,
        subject=Subject.CRIMINAL,
        source="test",
    )


def _keyword_scorer(keyword: str):
    """Score 1.0 for passages containing the keyword, 0.0 otherwise."""

    def scorer(pairs):
        return [1.0 if keyword in passage else 0.0 for _, passage in pairs]

    return scorer


def test_rerank_orders_by_score():
    chunks = [
        _chunk("a:s1", "provisions about contracts"),
        _chunk("a:s2", "punishment for murder is severe"),
        _chunk("a:s3", "registration of vehicles"),
    ]
    r = Reranker(scorer=_keyword_scorer("murder"))
    ranked = r.rerank("what is the punishment for murder?", chunks)
    assert ranked[0].id == "a:s2"  # the relevant chunk moves to the top
    assert len(ranked) == 3


def test_top_k_truncates():
    chunks = [_chunk(f"a:s{i}", f"text {i}") for i in range(10)]
    r = Reranker(scorer=lambda pairs: list(range(len(pairs))))  # last scores highest
    ranked = r.rerank("q", chunks, top_k=3)
    assert len(ranked) == 3
    assert ranked[0].id == "a:s9"  # highest score first


def test_empty_chunks_returns_empty():
    r = Reranker(scorer=lambda pairs: [])
    assert r.rerank("q", []) == []


def test_no_top_k_returns_all():
    chunks = [_chunk(f"a:s{i}", f"text {i}") for i in range(5)]
    r = Reranker(scorer=lambda pairs: [0.0] * len(pairs))
    assert len(r.rerank("q", chunks)) == 5


def test_scorer_sees_embed_text_with_metadata_header():
    """The reranker must score the header+text view, so it sees act/section/era."""
    seen: list[str] = []

    def capture(pairs):
        seen.extend(p for _, p in pairs)
        return [0.0] * len(pairs)

    chunks = [_chunk("a:s103", "whoever commits murder…", section="103")]
    Reranker(scorer=capture).rerank("q", chunks)
    assert len(seen) == 1
    assert "Section 103" in seen[0]  # header present
    assert "whoever commits murder" in seen[0]  # body present


def test_ties_keep_incoming_order():
    """Equal scores must preserve the RRF order (stable sort), not shuffle."""
    chunks = [_chunk(f"a:s{i}", f"text {i}") for i in range(4)]
    r = Reranker(scorer=lambda pairs: [0.5] * len(pairs))
    ranked = r.rerank("q", chunks)
    assert [c.id for c in ranked] == [c.id for c in chunks]


def test_query_paired_with_every_chunk():
    captured: list[tuple[str, str]] = []

    def capture(pairs):
        captured.extend(pairs)
        return [0.0] * len(pairs)

    chunks = [_chunk("a:s1", "one"), _chunk("a:s2", "two")]
    Reranker(scorer=capture).rerank("my question", chunks)
    assert all(q == "my question" for q, _ in captured)
    assert len(captured) == 2
