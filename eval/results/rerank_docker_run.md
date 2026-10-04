# Reranker evaluation (Docker run)

Run on the user's machine with `docker compose exec backend python eval/run_retrieval_eval.py --rerank`
(the cross-encoder cannot be downloaded in the development sandbox). Same 32 queries. This run
predates the place-name fix ("bezawada" → "Vijayawada"), so its non-reranked rows match the
earlier numbers (hybrid + rewrite Recall@5 0.927), not the current `retrieval_eval.md` (0.947).

| Variant | Recall@5 | Recall@300w | Precision@5 | MRR@10 | nDCG@10 | Ctx recall | Follow-up Recall@5 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| section · hybrid + rewrite | 0.927 | 0.953 | 0.376 | 0.913 | 0.885 | 0.85 | 0.917 | 4.22 |
| section · hybrid + rewrite + rerank | 0.94 | 0.967 | 0.384 | 0.953 | 0.932 | 0.883 | 0.917 | 233.44 |

## Answerability gate

| Variant | Score | τ | Answered positives | Refused negatives | Balanced acc. | Max neg. score | Min pos. score | Best τ on this set |
|---|---|---|---|---|---|---|---|---|
| section · hybrid + rewrite | best dense cosine | 0.62 | 24/25 | 6/7 | 0.909 | 0.6204 | 0.6191 | 0.628 (0.98) |
| section · hybrid + rewrite + rerank | rerank | 0.15 | 24/25 | 7/7 | 0.98 | 0.0008 | 0.0008 | 0.453 (0.98) |

Reading: with the reranker every off-topic query scores ≤ 0.0008. The fitted threshold 0.453 is
the midpoint between 0.0008 and the next score, so the 24 answered questions scored ≈ 0.905 or
more. `RERANK_THRESHOLD=0.15` therefore sits in a wide gap (any value between ~0.001 and ~0.9
gives the same result on this set). The one refused positive scored 0.0008; the eval script now prints
gate errors by query id so the next run names it. Cost: about 230 ms p50 per query on CPU (12 pairs).
