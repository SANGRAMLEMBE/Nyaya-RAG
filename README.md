# Nyaya-RAG

**Era-aware, citation-verified retrieval-augmented generation over Indian
statutes — fully local, zero external APIs.**

> Research question: how much do structure-aware chunking, hybrid retrieval,
> reranking, and programmatic citation verification reduce legal hallucination
> versus a naive RAG baseline?

India replaced its three core criminal laws on 1 July 2024 (IPC→BNS,
CrPC→BNSS, Evidence Act→BSA). A legal assistant that ignores this gives
dangerously wrong answers. Nyaya-RAG treats the era as a first-class retrieval
dimension: every chunk is era-tagged, undated criminal queries trigger a
disambiguation question, and section-mapping tables translate citations
across eras.

⚠️ This system provides **legal information, not legal advice** (Advocates
Act, 1961). Every answer carries a disclaimer and an escalation path to free
legal aid (NALSA helpline **15100**).

## Architecture
PDF acts (India Code) → section-tree parser → era-tagged JSONL chunks →
BGE-M3 dense + BM25 sparse → RRF fusion → bge-reranker-v2-m3 →
Qwen2.5-14B-Instruct (local vLLM) → **citation verifier** → answer with
verified citations + disclaimer.

## Results
See [RESULTS.md](RESULTS.md) — retrieval ablations (recall@k, MRR, nDCG),
citation precision, pre/post-verifier hallucination rate, red-team outcomes.
*(Populated during eval, Days 5–7.)*

## Reproduce
See [docs/SETUP_A100.md](docs/SETUP_A100.md), then:
`make download && make test`. Every pipeline stage is a `python -m nyaya...`
module with a manifest; no manual steps.
