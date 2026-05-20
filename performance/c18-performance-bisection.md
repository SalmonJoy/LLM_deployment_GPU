# Performance — Performance Bisection: Finding the Slow Layer in a Stack

**Slug:** `c18-performance-bisection`
**Batch:** parallel-20 (concepts)
**Started:** 2026-05-20T17:42:39
**Finished:** 2026-05-20T17:44:43

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
| Wall time (s) | 123.90 |
| Prompt tokens | 917 |
| Output tokens (eval_count) | 6161 |
| Output words | 3461 |
| Output characters | 22352 |
| Tokens / second (per-request, shared GPU) | 49.7 |
| Words / token (efficiency) | 0.562 |

## Notes

- This article was generated as part of a 20-way parallel batch — per-request tok/s is lower than the sequential baseline (116 tok/s) because all 20 requests shared the GPU's decode capacity.
- Aggregate batch throughput (sum of tokens / batch wall time) is the more meaningful number for parallel runs — see `_summary_concepts.md`.
