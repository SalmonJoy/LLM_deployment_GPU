# Performance — Why Default Configurations Are Almost Always Wrong for LLM Serving

**Slug:** `c12-defaults-are-wrong`
**Batch:** parallel-20 (concepts)
**Started:** 2026-05-20T17:42:39
**Finished:** 2026-05-20T17:45:43

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
| Wall time (s) | 184.10 |
| Prompt tokens | 891 |
| Output tokens (eval_count) | 5648 |
| Output words | 3255 |
| Output characters | 20392 |
| Tokens / second (per-request, shared GPU) | 30.7 |
| Words / token (efficiency) | 0.576 |

## Notes

- This article was generated as part of a 20-way parallel batch — per-request tok/s is lower than the sequential baseline (116 tok/s) because all 20 requests shared the GPU's decode capacity.
- Aggregate batch throughput (sum of tokens / batch wall time) is the more meaningful number for parallel runs — see `_summary_concepts.md`.
