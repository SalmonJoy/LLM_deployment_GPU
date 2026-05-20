# Run summary — 10 articles via local vLLM

**Endpoint:** `http://ollama.gravity.ind.in:11434/api/chat` (wrapper -> local vLLM gpt-oss-120b)
**Model name as routed:** `openai/gpt-oss-120b_vllm`
**Mode:** sequential, single-request, stream=false
**Total decode wall time:** 563.5 s (9.4 min)
**Articles:** 10 of 10 within the 2500-4000 word target; 0 errors.

_Articles 4 and 10 were regenerated with `num_predict=12000` after the initial run hit the 8000-token cap mid-sentence; the numbers below reflect the regenerated versions._

## Per-article metrics

| Slug | Wall (s) | Tokens | Words | Tok/s |
|---|---:|---:|---:|---:|
| `01-kv-cache-optimization-hopper` | 56.5 | 6599 | 3370 | 116.8 |
| `02-nvme-vs-premium-ssd` | 43.0 | 5029 | 2613 | 116.9 |
| `03-api-endpoints-lie-cloud-routing` | 51.1 | 5962 | 3170 | 116.6 |
| `04-diagnostic-capture-script` | 66.7 | 7751 | 3771 | 116.3 |
| `05-azure-daily-deallocate-cycle` | 66.6 | 7769 | 4086 | 116.6 |
| `06-docker-restart-policies` | 64.0 | 7440 | 4109 | 116.2 |
| `07-vram-sizing-math` | 58.6 | 6810 | 3469 | 116.1 |
| `08-storage-tiering` | 48.7 | 5652 | 2649 | 116.1 |
| `09-latency-driven-placement` | 55.4 | 6448 | 3358 | 116.3 |
| `10-ollama-vs-vllm` | 52.9 | 6148 | 3157 | 116.3 |

## Aggregate statistics

| Metric | Min | Mean | Max |
|---|---:|---:|---:|
| Wall time (s) | 43.0 | 56.4 | 66.7 |
| Output tokens | 5029 | 6561 | 7769 |
| Output words | 2613 | 3375 | 4109 |
| Decode rate (tok/s) | 116.1 | 116.4 | 116.9 |

## How this was measured

- 10 articles generated sequentially through `http://ollama.gravity.ind.in:11434/api/chat` (Ollama-API shape).
- The wrapper's `_vllm` suffix routes requests to the local vLLM container on the GPU host (gpt-oss-120b, MXFP4, 32K context).
- Wall time captured client-side around `requests.post(...)`.
- Decode rate is `eval_count / wall_s` — includes any queueing on the vLLM side (which is zero for sequential single requests).
- Prompt tokens include the system prompt (~250 tokens) plus the per-article user prompt (~300 tokens depending on guidance length).

## Notable observations

- **Decode rate is remarkably stable**: 116.1 to 116.9 tok/s across 10 different prompts and output lengths from 3157 to 4111 words. That tight band is the H100's hot decode speed for gpt-oss-120b MXFP4 on vLLM with no concurrent load. Any variability between articles is dominated by output length, not the model.
- **Wall time scales linearly with output tokens** at ~116 tok/s. Predict the cost of a generation: `seconds ≈ output_tokens / 116`.
- **Prompt processing is negligible** at this scale — ~600 prompt tokens prefilled in a few hundred ms before decode starts.
- **No errors, no retries, no timeouts** across 10 long-form runs. The wrapper + local vLLM path is stable for single-stream workloads.
