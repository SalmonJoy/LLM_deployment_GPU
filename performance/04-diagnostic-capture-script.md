# Performance — Building a Diagnostic Capture Script for Inherited LLM Infrastructure

**Slug:** `04-diagnostic-capture-script`
**Started:** 2026-05-20T16:33:55
**Finished:** 2026-05-20T16:35:01

## Request

- Endpoint: `http://ollama.gravity.ind.in:11434/api/chat`
- Model: `openai/gpt-oss-120b_vllm` (routed through wrapper to local vLLM)
- Stream: `false`
- num_predict cap: 8000
- temperature: 0.6, top_p: 0.92

## Result

| Metric | Value |
|---|---:|
| Wall time (s) | 66.65 |
| Prompt tokens | 783 |
| Output tokens (eval_count) | 7751 |
| Output words | 3771 |
| Output characters | 26591 |
| Tokens / second (decode) | 116.3 |
| Words / token (efficiency) | 0.487 |

## Notes

- Single-request (sequential), no concurrent load.
- Wall time = client-side total including network RTT.
- Cloud and Ollama-local routes were NOT exercised in this run; only the local vLLM path via the `_vllm` suffix.
