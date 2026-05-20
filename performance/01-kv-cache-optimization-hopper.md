# Performance — Optimizing KV Cache for Hopper GPUs: Flash Attention and q8_0 Quantization

**Slug:** `01-kv-cache-optimization-hopper`
**Started:** 2026-05-20T16:23:13
**Finished:** 2026-05-20T16:24:10

## Request

- Endpoint: `http://ollama.gravity.ind.in:11434/api/chat`
- Model: `openai/gpt-oss-120b_vllm` (routed through wrapper to local vLLM)
- Stream: `false`
- num_predict cap: 8000
- temperature: 0.6, top_p: 0.92

## Result

| Metric | Value |
|---|---:|
| Wall time (s) | 56.50 |
| Prompt tokens | 740 |
| Output tokens (eval_count) | 6599 |
| Output words | 3370 |
| Output characters | 21269 |
| Tokens / second (decode) | 116.8 |
| Words / token (efficiency) | 0.511 |

## Notes

- Single-request (sequential), no concurrent load.
- Wall time = client-side total including network RTT.
- Cloud and Ollama-local routes were NOT exercised in this run; only the local vLLM path via the `_vllm` suffix.
