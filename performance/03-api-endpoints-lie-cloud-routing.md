# Performance — When Your LLM Endpoint Lies: Detecting Hidden Cloud Routing in API Proxies

**Slug:** `03-api-endpoints-lie-cloud-routing`
**Started:** 2026-05-20T16:24:53
**Finished:** 2026-05-20T16:25:44

## Request

- Endpoint: `http://ollama.gravity.ind.in:11434/api/chat`
- Model: `openai/gpt-oss-120b_vllm` (routed through wrapper to local vLLM)
- Stream: `false`
- num_predict cap: 8000
- temperature: 0.6, top_p: 0.92

## Result

| Metric | Value |
|---|---:|
| Wall time (s) | 51.14 |
| Prompt tokens | 777 |
| Output tokens (eval_count) | 5962 |
| Output words | 3170 |
| Output characters | 21921 |
| Tokens / second (decode) | 116.6 |
| Words / token (efficiency) | 0.532 |

## Notes

- Single-request (sequential), no concurrent load.
- Wall time = client-side total including network RTT.
- Cloud and Ollama-local routes were NOT exercised in this run; only the local vLLM path via the `_vllm` suffix.
