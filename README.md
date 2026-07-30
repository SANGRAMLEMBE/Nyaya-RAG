# Nyaya-RAG

**Era-aware, citation-verified retrieval-augmented generation over Indian
statutes and case law — fully local, zero external APIs.**

> Research question: how much do structure-aware chunking, hybrid retrieval,
> cross-encoder reranking and programmatic citation verification reduce legal
> hallucination versus a naive RAG baseline?

On 1 July 2024 India replaced the three statutes at the centre of its criminal
justice system (IPC→BNS, CrPC→BNSS, Evidence Act→BSA). For decades to come both
bodies of law remain live: offences committed before the cutover are still tried
under the repealed codes. A legal assistant that is unaware of this transition
does not merely lose accuracy — it returns confidently worded answers under the
wrong statute. Compounding that, large language models fabricate legal citations
at high rates, and a fabricated section number is indistinguishable in form from
a real one.

Nyaya-RAG addresses both failure modes with mechanism rather than prompt
engineering:

1. **Era is a first-class retrieval dimension.** Every chunk carries an era
   label, queries are routed through a deterministic classifier that resolves
   era *and records why*, and hand-verified mapping tables translate citations
   across the 2024 boundary.
2. **Citations are verified, not trusted.** A post-generation verifier parses
   every citation the model emits, checks it against both the corpus and the
   specific retrieved context, and strips whatever it cannot confirm.

⚠️ This system provides **legal information, not legal advice** (Advocates
Act, 1961). Every answer carries a disclaimer and an escalation path to free
legal aid (NALSA helpline **15100**).

## Architecture

```
acts + judgments → provenance-tracked ingestion → section-tree / paragraph
  chunking → era-tagged JSONL
    → BGE-M3 dense (Qdrant, era filter in-query) ∥ BM25 sparse
    → RRF fusion → bge-reranker-v2-m3 → top-8
    → Qwen2.5-14B-Instruct (local vLLM)
    → CITATION VERIFIER → answer + verified citations + disclaimer
```

**Corpus:** 8,932 chunks across 34 documents — 17 statutes (4,059 chunks over
3,647 unique sections) and 17 landmark Supreme Court judgments (4,873
paragraph-window chunks). Judgments are tagged era-neutral: precedent persists
across the transition. 1,090 hand-verified rows map IPC↔BNS, CrPC↔BNSS and
IEA↔BSA sections.

## Results

Evaluated on a **150-question gold set** where every answer key is grounded in a
real corpus section (auto-validated — no fabricated citations). The validated
gold set now stands at 300 questions; re-running the harnesses at n=300 is in
progress.

**Retrieval** (recall@k / MRR / nDCG@10):

| Config | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| dense (BGE-M3) | 0.993 | 1.000 | 0.920 | 0.940 |
| BM25 | 0.727 | 0.800 | 0.608 | 0.654 |
| hybrid (RRF) | 0.873 | 0.953 | 0.755 | 0.803 |
| **hybrid + rerank (bge-reranker-v2-m3)** | **0.993** | **1.000** | **0.933** | **0.950** |

An ablation finding that contradicts a common default: **RRF fusion with BM25
alone *degrades* ranking** relative to dense retrieval (MRR 0.920 → 0.755). The
gold questions are semantic paraphrases of section headings — precisely where a
bi-encoder is strong and bag-of-words matching is weak. Cross-encoder re-scoring
of the fused pool repairs this and produces the best configuration: recall from
fusion, precision from joint (query, section) scoring.

**Citation verification.** Across the 150 questions the base model emitted 82
citations, of which **31.7% named sections absent from the corpus** — outright
fabrication, not misattribution (0 citations were real-but-unretrieved). The
verifier removed all of them, so post-verification citation precision is
**1.000** and the hallucinated-citation rate is **0.000**.

Read that second number correctly: it is a **guarantee by construction**, not an
independent measurement — the verifier strips every citation it cannot confirm,
so the surviving set is exactly the verified set. The empirically meaningful
figure is the **31.7% base rate**, which quantifies the risk the mechanism
eliminates. Era-correctness on the criminal subset is **1.000**: the era filter
never admitted a wrong-era chunk.

**Verifier-gated training chain** (fully local, no human annotation):
the same verifier was reused as an automated quality gate on synthetic training
data. It rejected **3,934 of 16,191** generated answers (≈24%) for carrying no
citation or an unverifiable one, yielding **12,257 grounded pairs** plus 400
refusal examples (**12,657** total) used for 4-bit QLoRA supervised fine-tuning
of Qwen2.5-14B (validation loss 0.238). Of
604 answers subsequently harvested from the tuned model, only **20 (3.3%)** still
contained an unverifiable citation.

Note that 3.3% and 31.7% are **not** like-for-like — different denominators
(answers vs citations), different question sets and different models. The
apples-to-apples evaluation of the tuned adapter on the same 150-question gold
set is implemented and pending a GPU slot; no fine-tuning improvement is claimed
until it runs.

See [RESULTS.md](RESULTS.md) for the full tables.

## Stack

Python 3.11 · PyTorch · vLLM (Qwen2.5-14B-Instruct) · BGE-M3 · Qdrant · BM25 ·
bge-reranker-v2-m3 · TRL/PEFT (QLoRA, DPO) · FastAPI · Streamlit · pytest ·
ruff · GitHub Actions. Every model runs on owned hardware; there are no external
API calls anywhere in the product path. Evaluation runs as self-contained PBS
batch jobs on an A100.

## Reproduce

See [docs/SETUP_A100.md](docs/SETUP_A100.md), then:

```bash
make download && make test
```

Every pipeline stage is a `python -m nyaya...` module driven by a declarative
catalogue and a SQLite manifest; there are no manual steps. Architecture
decisions are recorded as ADRs in [DECISIONS.md](DECISIONS.md) and treated as
binding.

## Status

| Milestone | Scope | State |
|---|---|---|
| M1 | Corpus, era layer, hybrid retrieval, verifier, gold set, ablations, API + UI | complete (`v0.1`) |
| M2 | Case law indexed, query router, cross-encoder reranker, 300-q gold set | code complete; re-evaluation pending |
| M3 | Verifier-gated synthetic data, QLoRA SFT | trained; gold-set evaluation pending |
| M4 | DPO, red-team suite, CPU quantisation | harvest done; training and red-team pending |

## Licence

Apache-2.0. Code, mapping tables and the gold evaluation set are public;
corpora, model weights and synthetic training data are not redistributed
(judgment text is used under a non-commercial research tier).
