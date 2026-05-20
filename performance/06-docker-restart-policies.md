# Performance — Docker Restart Policies and Recreate Scripts for Self-Hosted Inference

**Slug:** `06-docker-restart-policies`
**Started:** 2026-05-20T16:27:59
**Finished:** 2026-05-20T16:29:03

## Request

- Endpoint: `http://ollama.gravity.ind.in:11434/api/chat`
- Model: `openai/gpt-oss-120b_vllm` (routed through wrapper to local vLLM)
- Stream: `false`
- num_predict cap: 8000
- temperature: 0.6, top_p: 0.92

## Result

| Metric | Value |
|---|---:|
| Wall time (s) | 64.00 |
| Prompt tokens | 755 |
| Output tokens (eval_count) | 7440 |
| Output words | 4109 |
| Output characters | 28967 |
| Tokens / second (decode) | 116.2 |
| Words / token (efficiency) | 0.552 |

## Notes

- Single-request (sequential), no concurrent load.
- Wall time = client-side total including network RTT.
- Cloud and Ollama-local routes were NOT exercised in this run; only the local vLLM path via the `_vllm` suffix.
