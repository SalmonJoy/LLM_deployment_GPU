# Performance — Ollama vs vLLM: Choosing the Right Inference Engine for Your Stack

**Slug:** `10-ollama-vs-vllm`
**Started:** 2026-05-20T16:35:01
**Finished:** 2026-05-20T16:35:54

## Request

- Endpoint: `http://ollama.gravity.ind.in:11434/api/chat`
- Model: `openai/gpt-oss-120b_vllm` (routed through wrapper to local vLLM)
- Stream: `false`
- num_predict cap: 8000
- temperature: 0.6, top_p: 0.92

## Result

| Metric | Value |
|---|---:|
| Wall time (s) | 52.87 |
| Prompt tokens | 878 |
| Output tokens (eval_count) | 6148 |
| Output words | 3157 |
| Output characters | 21694 |
| Tokens / second (decode) | 116.3 |
| Words / token (efficiency) | 0.514 |

## Notes

- Single-request (sequential), no concurrent load.
- Wall time = client-side total including network RTT.
- Cloud and Ollama-local routes were NOT exercised in this run; only the local vLLM path via the `_vllm` suffix.
