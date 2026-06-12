"""Unit tests for the resumability layer — no network needed."""

from pathlib import Path

from nyaya.pipelines.download import Manifest


def test_manifest_resume_semantics(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.sqlite")
    assert m.status("ipc_1860") is None

    m.record("ipc_1860", "https://example.org/ipc.pdf", status="failed", error="HTTP 503")
    assert m.status("ipc_1860") == "failed"

    m.record("ipc_1860", "https://example.org/ipc.pdf", status="ok", sha256="ab" * 32, bytes=1234)
    assert m.status("ipc_1860") == "ok"

    row = m.conn.execute("SELECT attempts FROM downloads WHERE id='ipc_1860'").fetchone()
    assert row[0] == 2
