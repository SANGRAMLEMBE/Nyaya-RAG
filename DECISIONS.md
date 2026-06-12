# Architecture Decision Records — Nyaya-RAG

## ADR-001: Fully local stack, zero external APIs
All inference (generation, embedding, reranking, eval-judging) runs on our own
hardware. vLLM serves Qwen2.5-14B-Instruct on localhost; the OpenAI Python
package is used only as a wire-protocol client to that local server. Rationale:
(a) the product promise is that user legal queries never leave the device;
(b) research depth — we own every layer; (c) reproducibility without billing keys.

## ADR-002: Corpus scope for v1 = Constitution + ~25 bare acts, no bulk judgments
Judgment scraping at scale (50k–200k docs, OCR, dedup) is a 6-week effort and
the #1 schedule risk. Statutes alone support a 150-question gold set across
all target subjects. Landmark SC judgments (~30, hand-picked from e-SCR) may be
added for demo richness if time permits. Revisit post-v1.

## ADR-003: Era handling is a first-class retrieval feature
Every chunk carries era ∈ {old_code, new_code, neutral}. Query pipeline asks
the incident-date question for undated criminal queries and applies an era
filter; IPC↔BNS / CrPC↔BNSS / IEA↔BSA section-mapping CSVs translate citations
across eras. A wrong-era answer is treated as a critical eval failure.

## ADR-004: One section = one chunk
Statutory sections are natural retrieval units with stable citations. Fixed
token-window chunking is kept only as an ablation baseline (RESULTS.md).

## ADR-005: Citations are verified, not trusted
A post-generation verifier parses every citation in the model output and
checks (1) the cited section exists in the corpus and (2) it appeared in the
retrieved context. Unverifiable citations are stripped and replaced with
"I could not verify this in my sources." Target: ~0% post-verifier
hallucinated-citation rate on the gold set; the pre-verifier rate is reported.

## ADR-006: Models
Generation Qwen2.5-14B-Instruct (bf16, vLLM) — stronger Hindi for future work,
fits A100 comfortably. Embeddings BAAI/bge-m3 (dense, multilingual, handles
section-number-heavy text). Reranker bge-reranker-v2-m3. Judge (eval only)
Qwen2.5-32B-Instruct-AWQ, served locally and never used in the product path.

## ADR-007: Open-core split (placement + product goals coexist)
Public in this repo: all pipeline/retrieval/eval code, IPC↔BNS mapping tables
(reproducible from official sources), the gold eval set (research credibility),
RESULTS.md. Private (gitignored, distributed later as a licensed "law pack"):
processed corpus at scale, synthetic training data, fine-tuned weights.
Patents are not pursued for v1: Section 3(k) Patents Act excludes software per
se, and the moat is data quality + update cadence + trust, not claims.

## ADR-008: Commercial license hygiene
Only Apache-2.0/MIT/BSD components in the product path. Generation/training
models: Qwen2.5 7B/14B/32B (Apache 2.0 — verify each model card before first
download; Qwen licenses vary BY SIZE). Llama-family excluded from the product
path (community-license conditions). Every corpus document already carries a
`license` field; restricted sources never enter the corpus. Third-party API
terms (e.g. Indian Kanoon) reviewed for commercial use BEFORE purchase.
