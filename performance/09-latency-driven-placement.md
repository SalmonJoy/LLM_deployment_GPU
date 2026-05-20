# Performance — Latency-Driven Placement: When to Split a GPU Host into Multiple VMs

**Slug:** `09-latency-driven-placement`
**Started:** 2026-05-20T16:30:51
**Finished:** 2026-05-20T16:31:46

## Request

- Endpoint: `http://ollama.gravity.ind.in:11434/api/chat`
- Model: `openai/gpt-oss-120b_vllm` (routed through wrapper to local vLLM)
- Stream: `false`
- num_predict cap: 8000
- temperature: 0.6, top_p: 0.92

## Result

| Metric | Value |
|---|---:|
| Wall time (s) | 55.42 |
| Prompt tokens | 797 |
| Output tokens (eval_count) | 6448 |
| Output words | 3358 |
| Output characters | 22175 |
| Tokens / second (decode) | 116.3 |
| Words / token (efficiency) | 0.521 |

## Notes

- Single-request (sequential), no concurrent load.
- Wall time = client-side total including network RTT.
- Cloud and Ollama-local routes were NOT exercised in this run; only the local vLLM path via the `_vllm` suffix.
