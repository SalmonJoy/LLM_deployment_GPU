# Performance — Idempotent Operations as a Design Principle for Infrastructure

**Slug:** `c15-idempotent-ops`
**Batch:** parallel-20 (concepts)
**Started:** 2026-05-20T17:42:39
**Finished:** 2026-05-20T17:44:52

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
| Wall time (s) | 132.82 |
| Prompt tokens | 912 |
| Output tokens (eval_count) | 6651 |
| Output words | 3771 |
| Output characters | 26521 |
| Tokens / second (per-request, shared GPU) | 50.1 |
| Words / token (efficiency) | 0.567 |

## Notes

- This article was generated as part of a 20-way parallel batch — per-request tok/s is lower than the sequential baseline (116 tok/s) because all 20 requests shared the GPU's decode capacity.
- Aggregate batch throughput (sum of tokens / batch wall time) is the more meaningful number for parallel runs — see `_summary_concepts.md`.
