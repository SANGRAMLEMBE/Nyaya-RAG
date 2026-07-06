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
Evaluated on a **150-question gold set** (13 subjects, both eras), where every
answer key is grounded in a real corpus section (auto-validated — no fabricated
citations).

**Retrieval** (recall@k / MRR / nDCG@10 over the gold set):

| Config | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| dense (bge-m3) | 0.993 | 1.000 | 0.920 | 0.940 |
| BM25 | 0.727 | 0.800 | 0.608 | 0.654 |
| hybrid (RRF) | 0.873 | 0.953 | 0.755 | 0.803 |
| **hybrid + rerank (bge-reranker-v2-m3)** | **0.993** | **1.000** | **0.933** | **0.950** |

RRF fusion alone lowers ranking quality vs dense (BM25 noise dilutes the
order), but cross-encoder re-scoring of the fused pool recovers it and sets
the best overall config — recall from fusion, precision from the reranker.

**Citation verification (the headline):** across the 150 questions the base
model (Qwen2.5-14B) emitted 82 citations, **31.7% of them hallucinated**
(section numbers absent from the corpus). The post-generation verifier stripped
every unverifiable citation → **0% hallucinated-citation rate**, citation
precision **1.000**, and **perfect era-correctness (1.000)** on the criminal
subset.

See [RESULTS.md](RESULTS.md) for the full tables — retrieval ablations,
citation precision and pre/post-verifier hallucination rate, era-correctness,
and red-team outcomes.

## Reproduce
See [docs/SETUP_A100.md](docs/SETUP_A100.md), then:
`make download && make test`. Every pipeline stage is a `python -m nyaya...`
module with a manifest; no manual steps.
