#!/bin/bash
# Sync Qdrant index + BM25 pickle from laptop to CHAMP scratch.
# Run this from your laptop AFTER indexing is complete.
#
# Usage:
#   bash scripts/sync_to_champ.sh
#
# Prerequisites:
#   - You are on the CSIO network (or VPN)
#   - SSH key is set up for the gateway (14.139.134.235)
#   - Indexing is complete: data/qdrant/ and data/processed/bm25.pkl exist

set -e

GATEWAY="14.139.134.235"
CHAMP_HOST="champ"
REMOTE_SCRATCH="/scratch/srikanth/nyaya-rag"

# Files to sync (code + indexes only — raw PDFs stay on laptop)
LOCAL_DIRS=(
    "nyaya/"
    "configs/"
    "data/qdrant/"
    "data/processed/bm25.pkl"
    "data/mappings/"
    "pyproject.toml"
    "scripts/"
    "docs/SETUP_A100.md"
)

echo "=== Syncing nyaya-rag to CHAMP ==="
echo "Remote: ${GATEWAY} → ${CHAMP_HOST}:${REMOTE_SCRATCH}"
echo ""

# rsync through the SSH gateway using ProxyJump
# -avz  : archive mode (preserves permissions), verbose, compress
# --progress : show per-file progress
rsync -avz --progress \
    -e "ssh -J ${GATEWAY}" \
    "${LOCAL_DIRS[@]}" \
    "${CHAMP_HOST}:${REMOTE_SCRATCH}/"

echo ""
echo "=== Sync complete ==="
echo "Next steps on CHAMP:"
echo "  1. ssh ${GATEWAY}"
echo "  2. ssh ${CHAMP_HOST}"
echo "  3. cd ${REMOTE_SCRATCH}"
echo "  4. source .venv/bin/activate  (or create venv first — see docs/SETUP_A100.md)"
echo "  5. qsub scripts/serve_vllm.pbs"
echo "  6. uvicorn nyaya.api.app:app --host 0.0.0.0 --port 8080"
