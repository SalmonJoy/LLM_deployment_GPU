# Performance — Debugging by Hypothesis Pruning: A Method for Strange System Behavior

**Slug:** `c04-debugging-hypothesis-pruning`
**Batch:** parallel-20 (concepts)
**Started:** 2026-05-20T17:42:39
**Finished:** 2026-05-20T17:44:45

## Request

- Endpoint: `http://ollama.gravity.ind.in:11434/api/chat`
- Model: `openai/gpt-oss-120b_vllm` (routed through wrapper to local vLLM)
- Stream: `false`
- num_predict cap: 9000
- temperature: 0.6, top_p: 0.92
- Concurrency: this article ran alongside 19 others in parallel

## Result

| Metric | Value |
|---|---:|
| Wall time (s) | 126.41 |
| Prompt tokens | 858 |
| Output tokens (eval_count) | 5290 |
| Output words | 3328 |
| Output characters | 20478 |
| Tokens / second (per-request, shared GPU) | 41.8 |
| Words / token (efficiency) | 0.629 |

## Notes

- This article was generated as part of a 20-way parallel batch — per-request tok/s is lower than the sequential baseline (116 tok/s) because all 20 requests shared the GPU's decode capacity.
- Aggregate batch throughput (sum of tokens / batch wall time) is the more meaningful number for parallel runs — see `_summary_concepts.md`.
