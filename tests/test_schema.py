import json

from nyaya.schema import Chunk, Era, Subject


def _chunk() -> Chunk:
    return Chunk(
        id="bns_2023:s103",
        text="Whoever commits murder shall be punished with ...",
        doc_id="bns_2023",
        act="Bharatiya Nyaya Sanhita, 2023",
        section="103",
        era=Era.NEW_CODE,
        subject=Subject.CRIMINAL,
        source="India Code",
    )


def test_chunk_jsonl_roundtrip() -> None:
    c = _chunk()
    restored = Chunk.model_validate(json.loads(c.model_dump_json()))
    assert restored == c


def test_embed_text_carries_metadata_header() -> None:
    t = _chunk().embed_text()
    assert t.startswith("[Bharatiya Nyaya Sanhita, 2023 | Section 103 | new_code]")
    assert "murder" in t


def test_extraction_confidence_bounds() -> None:
    import pytest

    with pytest.raises(ValueError):
        Chunk(
            id="x:1", text="t", doc_id="x", era=Era.NEUTRAL,
            subject=Subject.CONSUMER, source="India Code",
            extraction_confidence=1.5,
        )
