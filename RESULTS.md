# Results

> Populated by `nyaya/eval/run_ablations.py`. Do not edit tables by hand.

## 1. Retrieval ablation (gold set, n=150)
| Config | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| dense (bge-m3) | – | – | – | – |
| BM25 | – | – | – | – |
| hybrid (RRF) | – | – | – | – |
| hybrid + rerank | – | – | – | – |
| naive baseline (old repo: recursive split + FAISS) | – | – | – | – |

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
