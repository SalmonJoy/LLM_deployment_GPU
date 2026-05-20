# Performance — Surviving Azure's Daily Deallocate Cycle with Ephemeral NVMe Storage

**Slug:** `05-azure-daily-deallocate-cycle`
**Started:** 2026-05-20T16:26:53
**Finished:** 2026-05-20T16:27:59

## Request

- Endpoint: `http://ollama.gravity.ind.in:11434/api/chat`
- Model: `openai/gpt-oss-120b_vllm` (routed through wrapper to local vLLM)
- Stream: `false`
- num_predict cap: 8000
- temperature: 0.6, top_p: 0.92

## Result

| Metric | Value |
|---|---:|
| Wall time (s) | 66.60 |
| Prompt tokens | 761 |
| Output tokens (eval_count) | 7769 |
| Output words | 4086 |
| Output characters | 28107 |
| Tokens / second (decode) | 116.6 |
| Words / token (efficiency) | 0.526 |

## Notes

- Single-request (sequential), no concurrent load.
- Wall time = client-side total including network RTT.
- Cloud and Ollama-local routes were NOT exercised in this run; only the local vLLM path via the `_vllm` suffix.
