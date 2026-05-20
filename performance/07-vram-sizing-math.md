# Performance — VRAM Sizing Math for LLM Serving: The Four-Knob KV Cache Formula

**Slug:** `07-vram-sizing-math`
**Started:** 2026-05-20T16:29:03
**Finished:** 2026-05-20T16:30:02

## Request

- Endpoint: `http://ollama.gravity.ind.in:11434/api/chat`
- Model: `openai/gpt-oss-120b_vllm` (routed through wrapper to local vLLM)
- Stream: `false`
- num_predict cap: 8000
- temperature: 0.6, top_p: 0.92

## Result

| Metric | Value |
|---|---:|
| Wall time (s) | 58.65 |
| Prompt tokens | 814 |
| Output tokens (eval_count) | 6810 |
| Output words | 3469 |
| Output characters | 20510 |
| Tokens / second (decode) | 116.1 |
| Words / token (efficiency) | 0.509 |

## Notes

- Single-request (sequential), no concurrent load.
- Wall time = client-side total including network RTT.
- Cloud and Ollama-local routes were NOT exercised in this run; only the local vLLM path via the `_vllm` suffix.
