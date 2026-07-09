"""Fetch landmark judgments from the Indian Kanoon API (research phase).

Reads the case list from configs/judgments_catalog.yaml, searches Indian Kanoon
for each, picks the best Supreme Court match, and saves the full judgment text
to data/raw/judgments/<id>.json.

Auth: the API token is read from IK_API_TOKEN (env or .env) — NEVER hard-coded
or committed (ADR-008). Fetched judgment text is PRIVATE (data/raw is
gitignored, ADR-007): Indian Kanoon's terms restrict redistribution, and the
non-commercial free tier covers only the research phase. A commercial build
would re-source these from e-SCR (judgment text is public record) or a paid
IK plan — recorded as a known limitation.

Matching: a name search returns statutes + many cases, so we keep only
Supreme Court results and pick the most-cited one (landmark judgments are the
most cited — a reliable, non-fabricating signal). The chosen title + Indian
Kanoon doc id are printed so you can eyeball each pick.

Usage (laptop, token set)::

    python -m nyaya.pipelines.fetch_indiankanoon              # fetch all
    python -m nyaya.pipelines.fetch_indiankanoon --limit 3    # try a few first
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import time
from pathlib import Path

from nyaya.pipelines.download_judgments import load_catalog

log = logging.getLogger("nyaya.pipelines")

API = "https://api.indiankanoon.org"
OUT_DIR = "data/raw/judgments"
SC_SOURCE = "Supreme Court of India"
DELAY_S = 1.0


def load_token() -> str | None:
    tok = os.environ.get("IK_API_TOKEN")
    if tok:
        return tok.strip()
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("IK_API_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def search_query(catalog_title: str) -> str:
    """Turn 'X v. Y' into a plain-terms query — IK search dislikes the 'v.'."""
    return re.sub(r"\s+v\.?\s+|\s+versus\s+", " ", catalog_title, flags=re.IGNORECASE)


def _clean_title(title: str | None) -> str:
    """Lowercase + strip IK's <b> highlight tags for reliable matching."""
    return re.sub(r"</?b>", "", title or "").lower()


def key_tokens(catalog_title: str) -> list[str]:
    """Distinctive words from the petitioner name, e.g. 'Kesavananda', 'Bharati'.

    A full-text search returns any judgment mentioning the terms, so we anchor
    on the party name: the petitioner side (before 'v.') carries the
    identifying surname.
    """
    petitioner = re.split(r"\bv\.?\b|\bversus\b", catalog_title, flags=re.IGNORECASE)[0]
    return [w.lower() for w in re.findall(r"[A-Za-z]+", petitioner) if len(w) >= 4]


def pick_best_judgment(docs: list[dict], tokens: list[str]) -> dict | None:
    """Keep Supreme Court judgments whose TITLE contains the case's anchor word,
    then pick the most cited among them.

    If nothing matches the name, return None — we skip and flag for manual
    review rather than save the wrong case (grounding over coverage).
    Pure function (no network) so the policy is unit-tested.
    """
    sc = [d for d in docs if d.get("docsource") == SC_SOURCE]
    if tokens:
        anchor = max(tokens, key=len)  # the most distinctive (usually surname)
        sc = [d for d in sc if anchor in _clean_title(d.get("title"))]
    if not sc:
        return None
    return max(sc, key=lambda d: int(d.get("numcitedby") or 0))


def html_to_text(raw_html: str) -> str:
    """Indian Kanoon doc HTML → plain text, paragraph breaks preserved."""
    text = re.sub(r"(?i)</p\s*>", "\n\n", raw_html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([.,;:)])", r"\1", text)  # tag removal can leave " ." etc.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _post(path: str, token: str, params: dict | None = None) -> dict:
    import requests

    r = requests.post(
        f"{API}{path}",
        headers={"Authorization": f"Token {token}"},
        params=params or {},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def fetch_one(entry, token: str, out_dir: Path) -> bool:
    out = out_dir / f"{entry.id}.json"
    if out.exists():
        log.info("[%s] already fetched — skipping", entry.id)
        return True

    if entry.ik_tid is not None:  # pinned exact doc id — no search, no guessing
        tid = entry.ik_tid
        best = {}
        log.info("[%s] using pinned ik_tid=%s", entry.id, tid)
    else:
        search = _post(
            "/search/", token, {"formInput": search_query(entry.title), "pagenum": "0"}
        )
        best = pick_best_judgment(search.get("docs", []), key_tokens(entry.title))
        if best is None:
            log.warning(
                "[%s] no confident Supreme Court title-match for %r — SKIPPED "
                "(open it on indiankanoon.org and set ik_tid: <id> in the catalog)",
                entry.id, entry.title,
            )
            return False
        tid = best.get("tid")
        log.info("[%s] picked tid=%s  %r", entry.id, tid, (best.get("title") or "")[:80])
    doc = _post(f"/doc/{tid}/", token)
    text = html_to_text(doc.get("doc", ""))
    if len(text) < 500:
        log.warning("[%s] fetched text looks too short (%d chars)", entry.id, len(text))

    record = {
        "id": entry.id,
        "catalog_title": entry.title,
        "subject": entry.subject.value,
        "ik_tid": tid,
        "ik_title": doc.get("title") or best.get("title"),
        "publishdate": doc.get("publishdate") or best.get("publishdate"),
        "docsource": doc.get("docsource") or best.get("docsource"),
        "source": "Indian Kanoon (API)",
        "source_url": f"https://indiankanoon.org/doc/{tid}/",
        "text": text,
    }
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("[%s] saved %d chars", entry.id, len(text))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=None, help="fetch only the first N")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    token = load_token()
    if not token:
        print("No IK_API_TOKEN in env or .env — set it first.")
        raise SystemExit(1)

    entries = load_catalog()
    if args.limit:
        entries = entries[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for e in entries:
        try:
            if fetch_one(e, token, out_dir):
                ok += 1
        except Exception as exc:  # noqa: BLE001 — report and continue the batch
            log.error("[%s] failed: %s", e.id, exc)
        time.sleep(DELAY_S)
    log.info("done: %d/%d judgments fetched -> %s", ok, len(entries), out_dir)
    print("\nReview the picked titles above — if any is the wrong case, tell me "
          "and we'll refine that search.")


if __name__ == "__main__":
    main()
