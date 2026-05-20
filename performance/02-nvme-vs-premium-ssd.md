# Performance — From Premium SSD to NVMe: A 62× Cold-Load Speedup for Large Language Models

**Slug:** `02-nvme-vs-premium-ssd`
**Started:** 2026-05-20T16:24:10
**Finished:** 2026-05-20T16:24:53

## Request

- Endpoint: `http://ollama.gravity.ind.in:11434/api/chat`
- Model: `openai/gpt-oss-120b_vllm` (routed through wrapper to local vLLM)
- Stream: `false`
- num_predict cap: 8000
- temperature: 0.6, top_p: 0.92

## Result

| Metric | Value |
|---|---:|
| Wall time (s) | 43.02 |
| Prompt tokens | 736 |
| Output tokens (eval_count) | 5029 |
| Output words | 2613 |
| Output characters | 16689 |
| Tokens / second (decode) | 116.9 |
| Words / token (efficiency) | 0.520 |

## Notes

- Single-request (sequential), no concurrent load.
- Wall time = client-side total including network RTT.
- Cloud and Ollama-local routes were NOT exercised in this run; only the local vLLM path via the `_vllm` suffix.
