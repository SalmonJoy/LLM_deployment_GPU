# Performance — Storage Tiering for Self-Hosted LLM Stacks: Ephemeral vs Persistent

**Slug:** `08-storage-tiering`
**Started:** 2026-05-20T16:30:02
**Finished:** 2026-05-20T16:30:51

## Request

- Endpoint: `http://ollama.gravity.ind.in:11434/api/chat`
- Model: `openai/gpt-oss-120b_vllm` (routed through wrapper to local vLLM)
- Stream: `false`
- num_predict cap: 8000
- temperature: 0.6, top_p: 0.92

## Result

| Metric | Value |
|---|---:|
| Wall time (s) | 48.68 |
| Prompt tokens | 765 |
| Output tokens (eval_count) | 5652 |
| Output words | 2649 |
| Output characters | 18526 |
| Tokens / second (decode) | 116.1 |
| Words / token (efficiency) | 0.469 |

## Notes

- Single-request (sequential), no concurrent load.
- Wall time = client-side total including network RTT.
- Cloud and Ollama-local routes were NOT exercised in this run; only the local vLLM path via the `_vllm` suffix.
