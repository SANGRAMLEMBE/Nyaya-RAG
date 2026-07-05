"""Tests for bottom-k MinHash dedup — deterministic, stdlib-only."""

from nyaya.pipelines.dedup import _jaccard_bottom_k, _sketch, dedup

BASE = (
    "the appellant challenged the order of the tribunal on several grounds "
    "concerning the interpretation of the relevant provisions and the findings "
    "of fact recorded by the tribunal after examining the entire record "
) * 6


def test_exact_duplicate_dropped():
    r = dedup([("a", BASE), ("b", BASE)])
    assert r.keep == ["a"]
    assert r.drop == {"b": "a"}


def test_near_duplicate_dropped():
    # simulate an OCR/mirror variant: casing + a few token changes
    variant = BASE.upper().replace("TRIBUNAL", "TRIBUNAL BELOW", 1)
    r = dedup([("a", BASE), ("b", variant)], threshold=0.8)
    assert r.drop.get("b") == "a"


def test_different_docs_both_kept():
    other = (
        "registration of motor vehicles requires an application to the "
        "licensing authority in the prescribed form with the prescribed fee "
        "and evidence of the address of the applicant duly attested "
    ) * 6
    r = dedup([("a", BASE), ("b", other)])
    assert r.keep == ["a", "b"]
    assert r.drop == {}


def test_first_occurrence_wins_order_preference():
    # input order encodes source preference — the earlier id is kept
    r = dedup([("preferred_source", BASE), ("mirror_copy", BASE)])
    assert r.keep == ["preferred_source"]
    assert "mirror_copy" in r.drop


def test_three_way_group_maps_to_first():
    r = dedup([("a", BASE), ("b", BASE), ("c", BASE)])
    assert r.keep == ["a"]
    assert r.drop == {"b": "a", "c": "a"}
    assert r.n_duplicates == 2


def test_deterministic_across_runs():
    docs = [("a", BASE), ("b", BASE + " extra closing words"), ("c", BASE[: len(BASE) // 2])]
    r1 = dedup(docs)
    r2 = dedup(docs)
    assert r1.keep == r2.keep
    assert r1.drop == r2.drop


def test_sketch_is_stable_and_sorted():
    s1, s2 = _sketch(BASE), _sketch(BASE)
    assert s1 == s2
    assert s1 == sorted(s1)


def test_jaccard_estimator_bounds():
    a, b = _sketch(BASE), _sketch(BASE)
    assert _jaccard_bottom_k(a, b) == 1.0
    assert _jaccard_bottom_k(a, []) == 0.0


def test_short_docs_do_not_crash():
    r = dedup([("a", "short text"), ("b", "short text"), ("c", "another one")])
    assert "b" in r.drop
    assert "c" in r.keep
