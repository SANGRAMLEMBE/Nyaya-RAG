# Results

> Populated by `nyaya/eval/run_ablations.py`. Do not edit tables by hand.

## 1. Retrieval ablation (gold set, n=150)
| Config | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| dense (bge-m3) | 0.993 | 1.000 | 0.920 | 0.940 |
| BM25 | 0.727 | 0.800 | 0.608 | 0.654 |
| hybrid (RRF) | 0.873 | 0.953 | 0.755 | 0.803 |
| hybrid + rerank | – | – | – | – |
| naive baseline (old repo: recursive split + FAISS) | – | – | – | – |

**Finding:** dense (bge-m3) alone is the strongest config on this gold set —
RRF fusion with BM25 *lowers* MRR and nDCG (0.920→0.755, 0.940→0.803). The gold
questions are semantic paraphrases of section headings, which the dense encoder
captures well, while lexical BM25 introduces lower-ranked noise that RRF mixes
in. Reranking (pending) and the naive baseline are still to be run.

## 2. End-to-end answer quality (gold set, n=150)
| Metric | pre-verifier | post-verifier |
|---|---|---|
| citation precision | 0.683 | 1.000 |
| hallucinated-citation rate | 0.317 | 0.000 |
| faithfulness (local judge) | – | – |
| era-correctness (criminal subset) | 1.000 | 1.000 |

**Headline:** across 150 questions the base model (Qwen2.5-14B) emitted 82
citations, of which **31.7% were hallucinated** (section numbers absent from
the corpus). The post-generation verifier (ADR-005) stripped every
unverifiable citation, taking the hallucinated-citation rate to **0%** and
citation precision to **1.000**. Era-correctness on the criminal subset is
**perfect (1.000)** — the era filter never admitted a wrong-era citation.
Faithfulness via the local judge model is a later pass.

## 3. Red-team (n=25)
| Category | pass rate |
|---|---|
| fabricated-citation pressure | – |
| guaranteed-outcome promises | – |
| illegal-act assistance | – |
| advocate impersonation | – |
