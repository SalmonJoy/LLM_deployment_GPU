# The Latency Budget: Decomposing a User-Visible LLM Request

## 1. Why “My Chat Feels Slow” Is Not a Single Number  

You have probably heard a product manager say, *“Our chat UI feels sluggish, we need to shave 200 ms off the response time.”* The instinctive reaction is to point at the GPU, upgrade the model, or throw a larger batch at the inference server. But latency is a **budget** made up of several line items, and the biggest expense is rarely the one you guess.

Think of a **package delivery**. The warehouse packs the box in 30 seconds, the truck drives to the distribution hub in 3 hours, and the final mile to the customer takes 2 days. If the customer complains about a 2‑day delivery, you would not start redesigning the packing line; you would look at the long‑haul logistics. The same principle applies to an LLM request: the *user‑visible* time is the sum of network hops, serialization, framework overhead, prompt preparation, token generation, and the return path. Optimizing the wrong line item wastes engineering effort and budget.

In the sections that follow you will see a concrete breakdown, learn how to measure each term, and discover when a term that is usually negligible becomes the dominant cost.

---

## 2. The End‑to‑End Request Flow  

Below is the canonical path of a single chat turn from the moment the user clicks **Send** to the moment the assistant’s last token appears on the screen.

```
User → (1) Client‑side network RTT → (2) HTTP request deserialization
      → (3) Application wrapper (framework + DB) → (4) Prompt prefill
      → (5) Token decode (generation) → (6) HTTP response serialization
      → (7) Client‑side network return → UI render
```

| # | Component | Typical latency (in‑region) | What it *does* |
|---|-----------|-----------------------------|----------------|
| 1 | **Client network RTT** | 10‑100 ms | TCP handshake (if not reused) + round‑trip propagation |
| 2 | **HTTP deserialization** | < 1 ms | Parse JSON / protobuf payload into Python objects |
| 3 | **Wrapper overhead** | 1‑10 ms (sync) / 0‑2 ms (async) | Django/FastAPI middleware, DB lookups, auth checks |
| 4 | **Prefill (prompt processing)** | 30‑80 ms for 1 k tokens on H100 | Tokenize, embed, run the transformer up to the last prompt token |
| 5 | **Decode (generation)** | `output_tokens / 116 tok/s` on a single H100 | Autoregressive forward pass for each new token |
| 6 | **HTTP serialization** | < 1 ms | Encode response JSON, compress if enabled |
| 7 | **Client network return** | 10‑100 ms | Same path as #1, but usually outbound only |

> **Key insight:** In most production chat services the **decode step** (5) consumes > 80 % of the user‑visible budget, because every token requires a full forward pass through the model. The other steps are “fixed overhead” that become noticeable only when the output is short.

---

## 3. Measuring Each Piece – A Hands‑On Toolkit  

### 3.1. Capture Network RTT with `curl`

```bash
# Measure the round‑trip time to the inference endpoint (no payload)
curl -w "\nRTT: %{time_total}s\n" -o /dev/null -s http://api.mycompany.com/health
```

Typical output:

```
RTT: 0.042s
```

If you see 80 ms consistently, you already know that the network alone eats ~8 % of a 1‑second budget.

### 3.2. Time Deserialization in FastAPI

```python
# fastapi_latency.py
import time
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    start = time.perf_counter()
    payload = await request.json()          # <-- deserialization
    deserialization_ms = (time.perf_counter() - start) * 1e3

    # Pass downstream and attach timing for later aggregation
    response = await handle_chat(payload)
    response["metrics"]["deserialization_ms"] = deserialization_ms
    return response
```

Running a single request with `uvicorn fastapi_latency:app --workers 4` and inspecting the JSON payload will show a `deserialization_ms` field of ~0.3 ms.

### 3.3. Wrapper Overhead – Django Middleware Example

```python
# latency_middleware.py
import time
from django.utils.deprecation import MiddlewareMixin

class LatencyMiddleware(MiddlewareMixin):
    def process_view(self, request, view_func, view_args, view_kwargs):
        request._start_view = time.perf_counter()

    def process_template_response(self, request, response):
        view_ms = (time.perf_counter() - request._start_view) * 1e3
        response.context_data = response.context_data or {}
        response.context_data["wrapper_ms"] = view_ms
        return response
```

Add `'latency_middleware.LatencyMiddleware'` to `MIDDLEWARE`. In a test run you’ll see values ranging from 1 ms (pure async) to 7 ms (sync DB query).

### 3.4. Prefill Timing with HuggingFace Transformers

```python
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain the difference between latency and throughput in a single sentence."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

def prefill():
    with torch.no_grad():
        start = time.perf_counter()
        _ = model(input_ids)               # forward pass through prompt tokens
        return (time.perf_counter() - start) * 1e3

print(f"Prefill latency: {prefill():.1f} ms")
```

On a single H100 you’ll typically see **45 ms** for a 1 k‑token prompt. Scale linearly: 90 ms for 2 k tokens.

### 3.5. Decode Speed – Token‑per‑Second Benchmark

```python
def decode_speed(num_output_tokens=100):
    generated = []
    cur_input = input_ids.clone()
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(num_output_tokens):
            logits = model(cur_input).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token.item())
            cur_input = torch.cat([cur_input, next_token], dim=-1)
        elapsed = time.perf_counter() - start
    tps = num_output_tokens / elapsed
    print(f"Decode speed: {tps:.1f} tok/s ({elapsed*1e3:.0f} ms for {num_output_tokens} tokens)")

decode_speed(100)
```

Typical output on an H100:

```
Decode speed: 116.2 tok/s (861 ms for 100 tokens)
```

The 116 tok/s figure is a useful rule‑of‑thumb for capacity planning.

### 3.6. HTTP Serialization with `orjson`

```python
import orjson, time

def serialize(payload):
    start = time.perf_counter()
    data = orjson.dumps(payload)
    return (time.perf_counter() - start) * 1e3

print(f"Serialization latency: {serialize({'msg':'ok'}):.3f} ms")
```

Even for a 2 KB JSON object you’ll see **0.5 ms** on a modern CPU.

### 3.7. Return Path RTT

Reuse the `curl` command from §3.1 but send a small POST payload and inspect the `time_total`. The return RTT is symmetric to the inbound RTT unless you have asymmetric routing.

---

## 4. Putting Numbers Together – A Full‑Stack Example  

Let’s walk through a concrete request: the user asks for a 100‑token answer. Assume the following measured values (all in‑region, same data center):

| Component | Latency |
|-----------|---------|
| Client RTT (inbound) | 20 ms |
| HTTP deserialization | 0.4 ms |
| Wrapper overhead (FastAPI) | 3.2 ms |
| Prefill (1 k‑token prompt) | 48 ms |
| Decode (100 tokens) | 861 ms |
| HTTP serialization | 0.6 ms |
| Client RTT (outbound) | 20 ms |
| **Total** | **≈ 954 ms** |

**Breakdown:**  
- Decode = 90 % of total latency  
- Network (both directions) = 4 %  
- Fixed framework overhead = 3 %  
- Prefill = 5 %

If you were to **upgrade the GPU** from an H100 to an A100 (≈ 30 % slower on decode), the total would jump to ~1.3 s, a noticeable regression for end users. Conversely, shaving 5 ms off the wrapper (by moving a blocking DB call to an async cache) would barely move the needle.

### 4.1. Varying Output Length  

| Output tokens | Decode (ms) | Total (ms) | Decode % |
|---------------|-------------|------------|----------|
| 20 | 172 | 300 | 57 % |
| 50 | 430 | 540 | 80 % |
| 100 | 861 | 954 | 90 % |
| 200 | 1 720 | 1 800 | 95 % |

When the answer is short (< 30 tokens) the **prefill and network** become comparable to decode. That shift explains why some “quick‑reply” bots feel faster after you trim the prompt length.

### 4.2. Multiple Calls per Turn  

Suppose your assistant orchestrates **four** LLM calls to retrieve external knowledge before composing the final answer. The network RTT term now multiplies:

```
Total RTT = 4 × (inbound + outbound) ≈ 4 × 40 ms = 160 ms
```

If each call also generates 30 tokens, the decode budget per call drops to ~150 ms, making the **RTT × N** term a serious contender for the latency budget.

---

## 5. Latency vs. Throughput – The Batch Size Lever  

You can think of a GPU as a **highway** with a fixed number of lanes (CUDA cores). **Throughput** is the total number of cars (tokens) that pass per hour, while **latency** is the time a single car spends from entry to exit. Adding more cars (larger batch) fills the lanes, increasing the hourly flow, but each car must now wait for the car in front to clear the intersection.

### 5.1. Batch‑Size Experiment  

```bash
# Run a simple benchmark with varying batch sizes on a single H100
for B in 1 4 8 16 32; do
    python benchmark.py --batch-size $B --output-tokens 50
done
```

`benchmark.py` records per‑token latency and aggregate throughput. Sample results:

| Batch | Tokens/sec (throughput) | Avg token latency |
|-------|------------------------|-------------------|
| 1 | 116 tok/s | 8.6 ms |
| 4 | 440 tok/s | 11.4 ms |
| 8 | 860 tok/s | 11.6 ms |
| 16 | 1 600 tok/s | 12.5 ms |
| 32 | 2 800 tok/s | 14.3 ms |

Notice the **latency penalty** grows slowly (≈ 6 ms) while throughput scales almost linearly up to batch = 16. If your SLA is “< 1 s per user turn”, you can safely batch up to 8 requests without breaking the budget, but beyond that you risk violating the latency promise.

### 5.2. Dynamic Batching  

Frameworks such as **vLLM** or **Triton Inference Server** expose a *max‑wait‑time* parameter. The server holds incoming requests for up to `max_wait_ms` while waiting for more to arrive, then processes the whole batch. A typical configuration:

```yaml
# vllm.yaml
max_num_seqs: 256
max_batch_size: 64
max_wait_ms: 20   # wait at most 20 ms for additional requests
```

If your traffic pattern is bursty (e.g., a chat surge after a product launch), the 20 ms wait adds a **fixed latency** that you can account for in the budget. In low‑traffic periods the server will fall back to batch = 1, preserving the best possible latency.

---

## 6. Streaming: First‑Byte Latency vs. Total Time  

When you stream the model’s output token‑by‑token, the **first‑byte latency** becomes the sum of *prefill* and the time to generate the **first** token. Using the numbers from §4:

- Prefill = 48 ms  
- First token decode ≈ 8 ms (1 / 116 tok/s)  

**First‑byte latency ≈ 56 ms** – well under the 100 ms “instant feedback” threshold that UI designers love.

The total response time (≈ 950 ms) does not shrink because you still have to generate the remaining 99 tokens. However, the user perceives progress immediately, which dramatically improves the *feel* of responsiveness.

### 6.1. Enabling Streaming in FastAPI

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json, asyncio

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(request: Request):
    payload = await request.json()
    prompt = payload["prompt"]
    gen = model.generate_stream(prompt, max_new_tokens=100)  # hypothetical generator

    async def event_stream():
        async for token in gen:
            chunk = json.dumps({"token": token}) + "\n"
            yield chunk
            await asyncio.sleep(0)  # allow other coroutines to run

    return StreamingResponse(event_stream(), media_type="application/json")
```

With this endpoint, the client can start rendering the assistant’s answer after the first 50 ms, while the backend continues to work on the rest of the sequence.

---

## 7. Practical Exercise: Instrument Your Own Stack  

1. **Add a per‑request logger** that records timestamps at the entry and exit of each component (network, deserialization, wrapper, prefill, decode, serialization).  
   ```python
   # pseudo‑code for a generic middleware
   timestamps = {"enter": time.perf_counter()}
   # … after each stage …
   timestamps["prefill_end"] = time.perf_counter()
   # finally
   log_json = {k: (v - timestamps["enter"]) * 1e3 for k, v in timestamps.items()}
   logger.info("latency_breakdown", extra=log_json)
   ```

2. **Run a synthetic load** with `hey` or `locust` that mimics your production traffic mix (short vs. long prompts, single vs. multi‑call turns).  
   ```bash
   hey -n 5000 -c 64 -m POST -T "application/json" \
       -d '{"prompt":"Summarize the following article..."}' \
       http://api.mycompany.com/chat
   ```

3. **Collect the logs** into a time‑series database (e.g., Loki) and plot the distribution of each term.  
   ```sql
   SELECT percentile(latency_breakdown.prefill_ms, 95) FROM logs WHERE service='chat-api'
   ```

4. **Identify the 90th‑percentile dominant term** for each request class (short, medium, long).  

You will quickly discover that a “one‑size‑fits‑all” optimization (e.g., always increasing batch size) can actually hurt the latency of the minority class that matters most to your SLA.

---

## 8. Tailoring Optimizations to Your Workload  

| Workload characteristic | Dominant term(s) | Recommended focus |
|--------------------------|------------------|-------------------|
| **Short replies (< 30 tokens)** | Prefill + network RTT | Reduce prompt length, cache tokenized prompts, colocate inference close to the client (edge). |
| **Long answers (≥ 150 tokens)** | Decode | Upgrade GPU, use tensor‑parallelism, quantize to 4‑bit if acceptable, enable flash‑attention kernels. |
| **Multiple LLM calls per turn** | RTT × N | Persist TCP connections, use HTTP/2 or gRPC, batch calls across agents, move orchestration to the same process. |
| **Burst traffic with low latency SLA** | Batch‑size trade‑off | Tune `max_wait_ms` for dynamic batching, allocate a dedicated “low‑latency” pool of workers with batch = 1. |
| **High‑throughput background jobs** | Throughput > latency | Maximize batch size, enable model parallelism, accept higher per‑request latency. |

A common mistake is to **over‑optimize decode** for a workload that is dominated by network RTT (e.g., a mobile app on a 3G connection). In that case, moving the inference server to a CDN edge node reduces RTT from 80 ms to 20 ms, delivering a 60 ms improvement with zero GPU changes.

---

## 9. Advanced Topics – When the Simple Model Breaks  

### 9.1. Asynchronous Prefill  

If your prompt contains a **retrieval‑augmented** component (e.g., RAG), you can start the prefill *while* the retrieval is still in flight. This overlaps I/O with compute, shaving up to the full retrieval latency from the critical path.

```python
async def handle_chat(payload):
    retrieval_task = asyncio.create_task(fetch_documents(payload["query"]))
    prefill_task = asyncio.create_task(prefill(prompt_base))
    docs = await retrieval_task
    prompt = combine(prompt_base, docs)
    await prefill_task  # ensure the base prompt is already on GPU
    return await decode(prompt, max_new_tokens=payload["max_tokens"])
```

### 9.2. Kernel Fusion for Decode  

The decode loop can be accelerated by **fusing** the softmax, top‑k sampling, and token embedding lookup into a single CUDA kernel. Libraries such as **FlashInfer** claim up to 1.5× speed‑up, which translates into a 30 % reduction in the dominant decode term.

```bash
pip install flashinfer
```

Then replace the vanilla `model.generate` call with the flash‑infer API:

```python
from flashinfer import generate
output = generate(model, input_ids, max_new_tokens=100, temperature=0.7)
```

### 9.3. Quantization Trade‑offs  

Moving from FP16 to **4‑bit integer** can double the token‑per‑second rate (≈ 230 tok/s) on the same H100, but it adds **quantization error** that may increase the number of tokens needed to convey the same information. Always benchmark the *effective* latency (including any downstream re‑prompting) before committing.

---

## 10. Checklist for a Low‑Latency Chat Service  

| ✅ | Action |
|---|--------|
| 1 | **Instrument** every stage; store per‑request breakdowns. |
| 2 | **Profile** the distribution of output lengths; segment your traffic. |
| 3 | **Tune network**: keep TCP connections warm, enable HTTP/2, colocate services. |
| 4 | **Trim the prompt**: cache tokenized prefixes, remove unnecessary system messages. |
| 5 | **Scale decode**: choose the right GPU, enable flash‑attention, consider quantization. |
| 6 | **Batch wisely**: set `max_wait_ms` based on SLA; reserve a low‑latency pool for short turns. |
| 7 | **Stream** by default; expose first‑byte latency to the UI for better perceived performance. |
| 8 | **Re‑measure after each change**; ensure the dominant term actually moved. |
| 9 | **Document** the latency budget in your runbooks so new engineers know where to look first. |
| 10| **Automate alerts** on any component exceeding its budgeted percentile (e.g., decode > 1.2 × baseline). |

When you close the loop on measurement, you stop guessing and start *budgeting*—exactly like a construction project where each material (concrete, steel, labor) has a cost line item. The sum is the total expense, but you only cut costs where the line item is truly oversized.

---

## 11. Takeaway  

You now have a **complete, numerically grounded decomposition** of a user‑visible LLM request. By treating latency as a budget with line items, you can:

* **Identify** the real bottleneck for any traffic pattern.  
* **Apply** the right engineering lever—network, framework, prompt size, batch size, or decode kernel—without waste.  
* **Design** a monitoring regime that surfaces regressions before users notice them.  

The next time a stakeholder says “the chat feels slow”, you can reply with a data‑driven plan: “Our measurements show decode consumes 90 % of the 950 ms budget for 100‑token replies. Let’s first try flash‑attention, then revisit batch sizing, and finally evaluate edge deployment if we need to shave another 50 ms.”  

That is the power of a **latency budget**—a disciplined, quantitative lens that turns vague complaints into actionable engineering work.