# Performance — The Network Boundary as the Hidden Bug Source

**Slug:** `c11-network-boundary-bugs`
**Batch:** parallel-20 (concepts)
**Started:** 2026-05-20T17:46:50
**Finished:** 2026-05-20T17:47:42

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
| Wall time (s) | 51.88 |
| Prompt tokens | 887 |
| Output tokens (eval_count) | 6022 |
| Output words | 3299 |
| Output characters | 21221 |
| Tokens / second (per-request, shared GPU) | 116.1 |
| Words / token (efficiency) | 0.548 |

## Notes

- This article was generated as part of a 20-way parallel batch — per-request tok/s is lower than the sequential baseline (116 tok/s) because all 20 requests shared the GPU's decode capacity.
- Aggregate batch throughput (sum of tokens / batch wall time) is the more meaningful number for parallel runs — see `_summary_concepts.md`.
