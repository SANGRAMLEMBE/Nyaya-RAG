"""The catalog is the corpus contract — these tests gate every CI run."""

from pathlib import Path

from nyaya.pipelines.download import load_catalog
from nyaya.schema import Era

CATALOG = Path(__file__).resolve().parents[1] / "configs" / "acts_catalog.yaml"


def test_catalog_loads_and_validates() -> None:
    entries = load_catalog(CATALOG)
    assert len(entries) >= 20


def test_ids_unique() -> None:
    entries = load_catalog(CATALOG)
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids))


def test_2024_transition_pairs_present() -> None:
    """Both eras of the criminal-law transition MUST be in the corpus.
    Shipping only one side is the project's #1 dangerous failure mode."""
    ids = {e.id for e in load_catalog(CATALOG)}
    assert {"ipc_1860", "crpc_1973", "iea_1872"} <= ids
    assert {"bns_2023", "bnss_2023", "bsa_2023"} <= ids


def test_era_tags_consistent() -> None:
    by_id = {e.id: e for e in load_catalog(CATALOG)}
    assert by_id["ipc_1860"].era is Era.OLD_CODE
    assert by_id["bns_2023"].era is Era.NEW_CODE
    assert by_id["rti_2005"].era is Era.NEUTRAL


def test_priority1_covers_core_subjects() -> None:
    subjects = {e.subject.value for e in load_catalog(CATALOG) if e.priority == 1}
    for required in ("criminal", "consumer", "cyber", "family", "constitutional", "legal_aid"):
        assert required in subjects, f"gold-set subject {required!r} has no priority-1 act"
