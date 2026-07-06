"""Download landmark judgment PDFs from the judgments catalog.

Reads configs/judgments_catalog.yaml, downloads each entry that has a verified
https url, and writes the PDF + a sidecar meta json to data/raw/judgments/.
Entries without a url are skipped and reported — nothing is fabricated
(CLAUDE.md); you fill urls from e-SCR first.

Safety, matching the acts downloader: https-only, content validated by PDF
magic bytes (not just content-type), sha256 recorded, polite delay between
requests. Re-running skips files already downloaded (idempotent).

Usage (from a machine with internet — laptop; then scp to CHAMP)::

    python -m nyaya.pipelines.download_judgments            # download all ready
    python -m nyaya.pipelines.download_judgments --list     # show fill status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from nyaya.config import settings
from nyaya.schema import JudgmentCatalogEntry

log = logging.getLogger("nyaya.pipelines")

CATALOG = "configs/judgments_catalog.yaml"
OUT_DIR = "data/raw/judgments"


def load_catalog(path: str | Path = CATALOG) -> list[JudgmentCatalogEntry]:
    """Parse + validate the catalog; raises on a malformed/duplicate entry."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = [JudgmentCatalogEntry.model_validate(e) for e in raw.get("judgments", [])]
    ids = [e.id for e in entries]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate judgment ids in catalog: {sorted(dupes)}")
    return entries


def _download_one(entry: JudgmentCatalogEntry, out_dir: Path, proxies: dict) -> bool:
    import requests

    pdf_path = out_dir / f"{entry.id}.pdf"
    if pdf_path.exists():
        log.info("[%s] already downloaded — skipping", entry.id)
        return True

    log.info("[%s] fetching %s", entry.id, entry.url)
    resp = requests.get(
        entry.url,
        timeout=settings.request_timeout_s,
        proxies=proxies,
        headers={"User-Agent": settings.user_agent},
    )
    resp.raise_for_status()
    data = resp.content
    if not data.startswith(b"%PDF"):
        log.error("[%s] not a PDF (magic bytes) — refusing to save", entry.id)
        return False

    pdf_path.write_bytes(data)
    meta = {
        "id": entry.id,
        "title": entry.title,
        "subject": entry.subject.value,
        "year": entry.year,
        "citation": entry.citation,
        "source": entry.source,
        "source_url": entry.url,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "fetch_date": datetime.now(UTC).isoformat(),
    }
    (out_dir / f"{entry.id}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("[%s] saved %.0f KB", entry.id, len(data) / 1024)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="show fill status, no download")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    entries = load_catalog()
    ready = [e for e in entries if e.url]
    pending = [e for e in entries if not e.url]

    if args.list or not ready:
        print(f"catalog: {len(entries)} judgments — {len(ready)} ready, "
              f"{len(pending)} awaiting a url")
        for e in pending:
            print(f"  [ ] {e.id}: {e.title}")
        if args.list:
            return
        if not ready:
            print("\nNothing to download yet — fill in url: fields from e-SCR.")
            return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import os

    proxies = {}
    if os.environ.get("https_proxy"):
        proxies = {"http": os.environ["https_proxy"], "https": os.environ["https_proxy"]}

    ok = 0
    for e in ready:
        try:
            if _download_one(e, out_dir, proxies):
                ok += 1
        except Exception as exc:  # noqa: BLE001 — report and continue the batch
            log.error("[%s] failed: %s", e.id, exc)
        time.sleep(settings.request_delay_s)
    log.info("done: %d/%d downloaded, %d still pending a url", ok, len(ready), len(pending))


if __name__ == "__main__":
    main()
