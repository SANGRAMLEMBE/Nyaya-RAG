"""Resumable downloader for the Nyaya-RAG raw corpus.

Reads configs/acts_catalog.yaml, fetches each verified PDF URL, writes
data/raw/<id>.pdf plus a JSON metadata sidecar data/raw/<id>.meta.json,
and records every attempt in a SQLite manifest so a crashed or
interrupted run resumes exactly where it stopped.

Politeness guarantees: respects robots.txt, configurable inter-request
delay, exponential-backoff retries, descriptive User-Agent.

Usage:
    python -m nyaya.pipelines.download --dry-run
    python -m nyaya.pipelines.download
    python -m nyaya.pipelines.download --only ipc_1860 bns_2023
    python -m nyaya.pipelines.download --priority 1
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sqlite3
import sys
import time
import urllib.robotparser
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from nyaya.config import settings
from nyaya.schema import CatalogEntry, DocumentMeta, License

log = logging.getLogger("nyaya.download")

PDF_MAGIC = b"%PDF"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS downloads (
    id          TEXT PRIMARY KEY,
    url         TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('pending','ok','failed','skipped')),
    sha256      TEXT,
    bytes       INTEGER,
    http_status INTEGER,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    fetched_at  TEXT
);
"""


def load_catalog(path: Path) -> list[CatalogEntry]:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    entries = [CatalogEntry.model_validate(item) for item in raw["acts"]]
    ids = [e.id for e in entries]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate catalog ids: {dupes}")
    return entries


class Manifest:
    """SQLite-backed progress ledger; makes the job resumable."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(SCHEMA_SQL)
        self.conn.commit()

    def status(self, doc_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT status FROM downloads WHERE id = ?", (doc_id,)
        ).fetchone()
        return row[0] if row else None

    def record(self, doc_id: str, url: str, **fields: object) -> None:
        self.conn.execute(
            "INSERT INTO downloads (id, url, status, attempts) VALUES (?, ?, 'pending', 0) "
            "ON CONFLICT(id) DO NOTHING",
            (doc_id, url),
        )
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE downloads SET {sets}, attempts = attempts + 1 WHERE id = ?",  # noqa: S608
            (*fields.values(), doc_id),
        )
        self.conn.commit()


class RobotsCache:
    """Per-host robots.txt check, fetched once per host."""

    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent
        self._parsers: dict[str, urllib.robotparser.RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        if host not in self._parsers:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"https://{host}/robots.txt")
            try:
                rp.read()
            except OSError:
                log.warning("could not read robots.txt for %s; assuming allowed", host)
                rp.allow_all = True  # type: ignore[attr-defined]
            self._parsers[host] = rp
        return self._parsers[host].can_fetch(self.user_agent, url)


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(multiplier=2, min=2, max=120),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def fetch(session: requests.Session, url: str) -> requests.Response:
    resp = session.get(url, timeout=settings.request_timeout_s)
    resp.raise_for_status()
    return resp


def download_one(
    entry: CatalogEntry,
    session: requests.Session,
    manifest: Manifest,
    robots: RobotsCache,
    out_dir: Path,
) -> bool:
    assert entry.url is not None  # filtered upstream
    if not robots.allowed(entry.url):
        log.error("[%s] robots.txt disallows %s — skipping", entry.id, entry.url)
        manifest.record(entry.id, entry.url, status="skipped", error="robots.txt disallow")
        return False

    resp = fetch(session, entry.url)
    content = resp.content
    if not content.startswith(PDF_MAGIC):
        msg = f"response is not a PDF (content-type={resp.headers.get('content-type')!r})"
        log.error("[%s] %s", entry.id, msg)
        manifest.record(
            entry.id, entry.url, status="failed",
            http_status=resp.status_code, error=msg,
        )
        return False

    sha = hashlib.sha256(content).hexdigest()
    pdf_path = out_dir / f"{entry.id}.pdf"
    pdf_path.write_bytes(content)

    meta = DocumentMeta(
        id=entry.id,
        title=entry.title,
        source_url=entry.url,
        source=entry.source,
        era=entry.era,
        subject=entry.subject,
        license=License.PUBLIC,
        fetch_date=datetime.now(UTC),
        sha256=sha,
        bytes=len(content),
        content_type=resp.headers.get("content-type", "application/pdf"),
        catalog_priority=entry.priority,
    )
    (out_dir / f"{entry.id}.meta.json").write_text(
        meta.model_dump_json(indent=2), encoding="utf-8"
    )
    manifest.record(
        entry.id, entry.url, status="ok",
        sha256=sha, bytes=len(content),
        http_status=resp.status_code,
        fetched_at=meta.fetch_date.isoformat(),
    )
    log.info("[%s] ok — %.1f KB, sha256=%s…", entry.id, len(content) / 1024, sha[:12])
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, fetch nothing")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these catalog ids")
    parser.add_argument("--priority", type=int, choices=[1, 2], default=None)
    parser.add_argument("--force", action="store_true", help="re-download even if manifest says ok")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    entries = load_catalog(settings.catalog_path)
    if args.only:
        entries = [e for e in entries if e.id in set(args.only)]
    if args.priority:
        entries = [e for e in entries if e.priority == args.priority]

    missing_url = [e.id for e in entries if e.url is None]
    ready = [e for e in entries if e.url is not None]

    if missing_url:
        log.warning(
            "%d catalog entries have no verified URL yet and will be skipped: %s",
            len(missing_url), ", ".join(missing_url),
        )
    if args.dry_run:
        for e in ready:
            print(f"WOULD FETCH  {e.id:<18} {e.url}")
        for e_id in missing_url:
            print(f"NO URL YET   {e_id}")
        return 0
    if not ready:
        log.error("nothing to download — fill `url:` fields in %s", settings.catalog_path)
        return 1

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(settings.manifest_db)
    robots = RobotsCache(settings.user_agent)
    session = requests.Session()
    session.headers["User-Agent"] = settings.user_agent

    ok = fail = skipped = 0
    for entry in ready:
        if not args.force and manifest.status(entry.id) == "ok":
            log.info("[%s] already downloaded — skipping (use --force to refetch)", entry.id)
            skipped += 1
            continue
        try:
            ok += int(download_one(entry, session, manifest, robots, settings.raw_dir))
        except requests.RequestException as exc:
            log.error("[%s] failed after retries: %s", entry.id, exc)
            manifest.record(entry.id, entry.url or "", status="failed", error=str(exc))
            fail += 1
        time.sleep(settings.request_delay_s)

    log.info("done: %d ok, %d failed, %d skipped, %d awaiting URLs",
             ok, fail, skipped, len(missing_url))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
