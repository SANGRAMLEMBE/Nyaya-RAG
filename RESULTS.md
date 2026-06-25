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

## 2. End-to-end answer quality
| Metric | pre-verifier | post-verifier |
|---|---|---|
| citation precision | – | – |
| hallucinated-citation rate | – | – |
| faithfulness (local judge) | – | – |
| era-correctness (criminal subset) | – | – |

## 3. Red-team (n=25)
| Category | pass rate |
|---|---|
| fabricated-citation pressure | – |
| guaranteed-outcome promises | – |
| illegal-act assistance | – |
| advocate impersonation | – |
