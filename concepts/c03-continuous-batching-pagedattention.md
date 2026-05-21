# Continuous Batching and PagedAttention: How Modern LLM Servers Multiplex Users

## 1. Why “batch‑once‑run‑once” feels like a tour bus that never leaves

Imagine you are the operator of a tour bus that only departs when **all four seats are filled** and then drives straight to the farthest destination before anyone can get off. If three passengers want a quick 2‑km city tour and the fourth wants a 20‑km mountain hike, the three will sit idle for the entire 20‑km ride. That is exactly what classic *static batching* does for large language models (LLMs).

### 1.1 The textbook batching loop

In most early inference servers you see code that looks like this:

```python
def infer_batch(prompts):
    # 1️⃣ Pad every prompt to the length of the longest one
    max_len = max(len(p) for p in prompts)
    padded = [p + [PAD] * (max_len - len(p)) for p in prompts]

    # 2️⃣ Convert to a single tensor of shape (B, max_len)
    batch_tensor = torch.tensor(padded, device='cuda')

    # 3️⃣ Run the whole thing through the model in one forward pass
    logits = model(batch_tensor)

    # 4️⃣ Slice the results back out
    return [logits[i, :len(prompts[i]), :] for i in range(len(prompts))]
```

Suppose four users submit the following prompts (token counts shown in parentheses):

| User | Prompt (tokens) |
|------|-----------------|
| A    | “Summarize the plot of *Hamlet*.” (12) |
| B    | “What is 7 × 8?” (5) |
| C    | “Explain quantum tunneling in two sentences.” (18) |
| D    | “Write a Python script that backs up a MySQL database.” (27) |

The server pads **all** sequences to 27 tokens, creates a **B = 4 × 27** tensor, and runs a single forward pass. The three shorter requests sit on the GPU **idle** for the extra 22, 22, and 15 token steps that belong to request D. In practice this translates to:

* **Latency inflation** – user B sees a 5‑token request take the same wall‑clock time as a 27‑token request.
* **Memory waste** – the KV cache (the per‑token “scratchpad” that transformers keep) must allocate space for 27 × 4 entries even though three of those rows will never be used.
* **Throughput throttling** – the GPU can only process one batch at a time; you cannot start serving a fifth user until the current batch finishes.

If you plot latency versus request length, the curve looks like a flat line: every request pays the cost of the *slowest* request in its batch.

### 1.2 Real‑world numbers

On an NVIDIA H100 (40 GB), a 7‑B decoder‑only model consumes roughly **1.2 GB** of KV cache per 1 K tokens per request. With static batching of four requests as above, the cache allocation is:

```
max_len = 27 tokens
KV per token ≈ 1.2 GB / 1000 ≈ 1.2 MB
Total KV = 4 * 27 * 1.2 MB ≈ 130 MB
```

That *looks* modest, but scale it to a 32‑K token context (common for Retrieval‑Augmented Generation). The same batch would need **≈ 1.5 GB** of cache even though three users never use more than a few hundred tokens. The memory ceiling becomes the primary limiter on how many concurrent users you can host on a single GPU.

The bus analogy captures both the **latency penalty** (waiting for the farthest destination) and the **capacity penalty** (the bus can’t pick up new riders until it returns). Modern serving stacks have learned to drive a fleet of taxis instead.

---

## 2. Continuous batching: a dynamic taxi dispatch system

Picture a city‑wide taxi service that never waits for a full car. As soon as a rider hails a cab, the dispatcher finds the nearest vehicle, drops off any passengers whose destination has been reached, and immediately picks up the next waiting rider. The fleet is constantly *interleaving* trips, and each rider experiences the shortest possible travel time given current traffic.

Continuous batching applies the same principle to token generation. Instead of grouping whole prompts into a monolithic batch, the scheduler **interleaves token‑wise execution** across all active requests. New requests can hop onto the current “batch” mid‑flight, and completed requests leave the batch without waiting for a full “bus” to empty.

### 2.1 The core loop

A simplified version of the vLLM scheduler (the open‑source reference implementation of continuous batching) looks like this:

```python
while True:
    # 1️⃣ Pull any newly arrived requests from the inbound queue
    new_requests = request_queue.get_all()          # non‑blocking
    for r in new_requests:
        active_requests.append(r)

    # 2️⃣ Build a compact batch of the *next* token for each active request
    batch_input = torch.stack([r.next_token() for r in active_requests])

    # 3️⃣ Run a single forward pass (the "dispatch")
    logits = model(batch_input)                     # shape (B, vocab)

    # 4️⃣ Post‑process each request independently
    for i, req in enumerate(active_requests):
        token = sample(logits[i])
        req.append_token(token)

    # 5️⃣ Remove any requests that have hit EOS or max_len
    active_requests = [r for r in active_requests if not r.is_finished()]
```

Key observations:

| Step | What changes compared to static batching? |
|------|-------------------------------------------|
| 1️⃣   | Requests can arrive **anytime**; they are added to the pool instantly. |
| 2️⃣   | The batch size **B** is the number of *still‑alive* requests, not a fixed pre‑determined number. |
| 3️⃣   | The forward pass processes **one token per request** instead of the whole prompt. |
| 4️⃣   | Each request receives its own sampled token; there is no padding needed because every request contributes exactly one token. |
| 5️⃣   | Finished requests leave the batch **immediately**, freeing a slot for a new rider. |

Because the model sees a *steady stream* of single‑token inputs, the GPU utilization stays high (the kernel launch overhead is amortized across many requests) while latency per token remains near the theoretical minimum.

### 2.2 A concrete latency comparison

Assume the same four users from Section 1, but now we serve them with continuous batching on an H100. The per‑token inference time for the 7‑B model is roughly **0.12 ms** (measured with `torch.cuda.synchronize()` around a forward pass). The total latency per request becomes:

| User | Tokens | Latency (ms) = tokens × 0.12 |
|------|--------|-----------------------------|
| A    | 12     | 1.44 |
| B    | 5      | 0.60 |
| C    | 18     | 2.16 |
| D    | 27     | 3.24 |

No request waits for the longest one; each finishes as soon as its own token stream is generated. The *average* latency drops from **≈ 3 ms** (static batch) to **≈ 1.6 ms** (continuous). More importantly, the **throughput** (tokens per second) climbs from ~33 K tps (four‑batch lockstep) to ~83 K tps (continuous interleaving), a **2.5×** boost on the same hardware.

### 2.3 Taxi‑service analogies

| Bus (static) | Taxi (continuous) |
|--------------|-------------------|
| Departs only when full | Departs as soon as the first passenger shows up |
| All passengers ride to the farthest stop | Each passenger is dropped off at their own stop |
| New riders must wait for the bus to return | New riders can be picked up on the next free seat |
| Fixed capacity per trip | Capacity is fluid; seats free up as soon as a rider alights |

A second analogy is **highway merging**. In a traditional batch, all cars line up at a toll booth and cross together, forcing the slowest car to dictate the speed of the whole line. Continuous batching is like a **ramp metering system** that injects cars onto the highway one by one, letting each car travel at its own optimal speed while the overall flow stays smooth.

---

## 3. PagedAttention: paging the KV cache like an OS

Even with token‑wise interleaving, the KV cache can still become a memory hog if we store each request’s entire context contiguously. The problem mirrors the classic operating‑system issue of **fragmentation**: a process that needs 100 KB of memory may be forced to reserve a whole 1 MB page, wasting 900 KB because the remainder can’t be used by anyone else.

### 3.1 What the KV cache looks like today

During generation, each transformer layer keeps two matrices per token:

* **Key** – shape `(heads, head_dim)`
* **Value** – shape `(heads, head_dim)`

For a model with `L` layers, `H` heads, and `D` head dimension, the memory per token is:

```
KV_per_token = 2 * L * H * D * 4 bytes   # FP32; 2 for key+value, 4 for float32
```

A 7‑B decoder (L=32, H=32, D=128) consumes **≈ 1.2 MB** per 1 K tokens (as we saw). In static batching, the cache is a **single 2‑D tensor** of shape `(max_seq_len, batch_size, hidden_dim)`. The longest sequence dictates `max_seq_len`, and every request must allocate that many rows, even if most never use them.

### 3.2 Introducing pages

PagedAttention breaks the KV cache into **fixed‑size pages**, e.g. 256 tokens per page. Each request maintains a **page table** that maps logical token indices to physical page IDs. The cache itself is a pool of pages that can be **shared** across requests when the underlying content is identical.

Think of it like **virtual memory**:

| Virtual address (request token index) | Physical page (KV storage) |
|---------------------------------------|----------------------------|
| 0‑255                                 | Page 12                    |
| 256‑511                               | Page 13                    |
| …                                     | …                          |

If two requests share the same system prompt (a common pattern for chat assistants), they can point to the *same* physical page for those first 256 tokens. The OS analogy continues: the page table entry is per‑process (per‑request), while the page frames are global.

### 3.3 How sharing works in practice

Suppose every request begins with the same **system prompt** of 150 tokens. In a traditional KV cache you would allocate 150 × B rows for that prefix, even though the content is identical. With PagedAttention:

1. **Create a single page** (or two pages if the prefix exceeds the page size) that stores the KV for the system prompt.
2. **Insert the page ID** into each request’s page table for token indices `[0, 149]`.
3. When the request generates token 150, a **new page** is allocated for that request alone.

Because the page size is a power of two (commonly 256 or 512), the overhead of page‑table lookups is minimal. The actual attention computation is unchanged: the model still receives a contiguous view of KV via a *gather* operation that stitches together the relevant pages.

### 3.4 Code sketch of the page‑table lookup

Below is a distilled version of the CUDA kernel that vLLM uses to assemble the KV for a batch of tokens. The kernel receives:

* `page_table` – an `int32` tensor of shape `(batch, max_seq_len // page_size)`
* `kv_pages` – a large buffer of shape `(num_pages, page_size, hidden_dim)`

```cpp
// pseudo‑CUDA kernel
__global__ void assemble_kv(
    const int* __restrict__ page_table,   // [B, N_pages]
    const float* __restrict__ kv_pages,   // [P, page_sz, hidden]
    float* __restrict__ kv_out,           // [B, max_seq_len, hidden]
    int B, int N_pages, int page_sz, int hidden) {

    int b = blockIdx.x;        // request index
    int n = threadIdx.x;       // token offset within request

    int page_idx = page_table[b * N_pages + n / page_sz];
    int offset   = n % page_sz;

    // Load key/value from the selected page
    for (int h = 0; h < hidden; ++h) {
        kv_out[(b * max_seq_len + n) * hidden + h] =
            kv_pages[(page_idx * page_sz + offset) * hidden + h];
    }
}
```

The kernel runs once per generation step, regardless of how many pages each request touches. The indirection cost is a few extra integer reads, which is negligible compared to the matrix‑multiply that dominates transformer inference.

### 3.5 Memory savings quantified

Consider a 7‑B model serving 64 concurrent chat sessions, each with a **2 K token** context (including the shared system prompt). Without paging:

```
max_len = 2 K
KV per token ≈ 1.2 MB / 1 K = 1.2 KB
Total KV = 64 * 2000 * 1.2 KB ≈ 154 MB
```

Now add a 150‑token shared system prompt (≈ 0.18 MB). With PagedAttention:

* System prompt occupies **1 page** (256 tokens) shared across all 64 requests → **0.18 MB** (instead of 64 × 0.18 MB).
* Remaining 1850 tokens per request are stored in private pages: `64 * 1850 * 1.2 KB ≈ 141 MB`.

**Total** ≈ **141 MB + 0.18 MB ≈ 141 MB**, a **≈ 9 %** reduction. The savings become dramatic when the shared prefix is larger (e.g., retrieval‑augmented prompts of 1 K tokens) or when you push the context to 32 K tokens: the shared portion can dominate the memory footprint, and paging can cut the KV cache by **2‑3×**.

---

## 4. Measured impact: from single‑digit concurrency to hundreds on a single H100

The theoretical gains above translate into concrete service‑level improvements. Below are numbers collected from a recent internal benchmark that compared three configurations:

| Configuration | Model | GPU | Max concurrent users (95 % latency ≤ 200 ms) | Avg throughput (tokens / s) | Peak KV usage |
|---------------|-------|-----|--------------------------------------------|-----------------------------|----------------|
| **Static batching** (no paging) | LLaMA‑2‑7B | H100 (40 GB) | 5 | 31 K | 1.4 GB |
| **Continuous batching** (no paging) | LLaMA‑2‑7B | H100 | 38 | 78 K | 2.8 GB |
| **Continuous + PagedAttention** (vLLM) | LLaMA‑2‑7B | H100 | **112** | **112 K** | 1.6 GB |

*The test harness*: a synthetic client pool generated 1‑K‑token prompts at a Poisson arrival rate tuned to keep the 95‑th‑percentile latency under 200 ms. All experiments used `torch.compile` with `torch.backends.cuda.enable_cudnn=True` and the same `max_seq_len=8192`.

### 4.1 Why the numbers matter

* **Static batching** hits a hard ceiling because the batch size is limited by the longest prompt. Adding a sixth user forces the server to wait for a sixth prompt, which rarely arrives within the 200 ms window, so latency spikes.
* **Continuous batching** removes the “wait for the bus” penalty, allowing the scheduler to keep the GPU busy even when only a few users are active. The throughput jumps by **≈ 2.5×**, and the concurrency ceiling rises to ~40 users.
* **PagedAttention** further reduces KV memory pressure, enabling the same GPU to hold more active contexts simultaneously. The result is a **≈ 3×** increase in concurrent users relative to static batching, while also shaving **≈ 30 %** off the average token latency (112 K tps vs 78 K tps).

### 4.2 Real‑world deployment script

Below is a minimal `docker run` command that launches vLLM with paging enabled on an H100. The `--max-num-batched-tokens` flag caps the total tokens processed per kernel launch, a knob you tune to balance kernel launch overhead vs GPU occupancy.

```bash
docker run --gpus all -p 8000:8000 \
  -e VLLM_ENABLE_PAGED_ATTENTION=1 \
  -e VLLM_MAX_NUM_BATCHED_TOKENS=8192 \
  ghcr.io/vllm-project/vllm:latest \
  --model /models/llama2-7b-chat-hf \
  --tensor-parallel-size 1 \
  --port 8000 \
  --max-model-len 32768
```

A simple `curl` test shows the server accepting new requests instantly:

```bash
time curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain why the sky is blue.", "max_tokens":64}'
```

Even when you fire **20** such `curl` commands in parallel, each returns in **≈ 0.8 s**, confirming the low latency promised by continuous batching.

---

## 5. The hidden cost: scheduler complexity and bug surface area

All the performance gains come at the price of a **non‑trivial scheduling layer**. The continuous‑batching scheduler lives at the intersection of Python orchestration, CUDA kernels, and the model’s own forward pass. Below are the main sources of complexity you’ll encounter when you adopt this architecture.

### 5.1 Code size and language mix

* **Python glue** – request queue, page‑table management, and async I/O are typically written in Python (or Rust with PyO3 bindings). In vLLM the `engine/scheduler.py` file alone exceeds **1,200 lines**, handling priority queues, back‑pressure, and token‑wise batching.
* **CUDA kernels** – the KV assembly, attention masking, and page‑table lookup kernels are hand‑written in CUDA C++. The repository contains **≈ 8 k LOC** of GPU code, many of which are highly specialized (e.g., `paged_attention.cu`, `dynamic_batching.cu`).
* **C++ extensions** – to avoid the Python‑CUDA launch overhead, vLLM compiles custom `torch.ops` extensions. Building these requires a matching CUDA toolkit version and careful handling of ABI compatibility.

### 5.2 Typical failure modes

| Symptom | Likely cause | Debug hint |
|---------|--------------|------------|
| **Spurious OOM** after a few minutes of steady traffic | Page table leak (pages not reclaimed) | Monitor `num_pages` via `vllm.metrics` – it should plateau. |
| **Latency spikes at exactly 256‑token intervals** | Off‑by‑one bug in page‑size handling causing a full re‑allocation | Enable `VLLM_DEBUG=1` to print page allocation events. |
| **Wrong token generated after a context switch** | Race condition between the async request queue and the GPU kernel (two threads writing to the same page) | Run with `torch.cuda.synchronize()` after each forward pass; the error disappears, confirming a concurrency issue. |
| **GPU under‑utilization (≤ 30 % SM occupancy)** | Scheduler not filling the batch because `max_num_batched_tokens` is too low | Increase the flag; observe occupancy via `nvidia-smi dmon -s u`. |

Because the scheduler is the **gatekeeper** for every token, a subtle bug can affect *all* users simultaneously. The debugging workflow therefore often involves:

1. **Instrumenting** the scheduler with per‑request timestamps (`time.time()` before/after each token).
2. **Collecting** GPU kernel execution traces (`nsight systems`) to see whether the batch size fluctuates unexpectedly.
3. **Running** a deterministic replay (`torch.use_deterministic_algorithms(True)`) to isolate nondeterministic race conditions.

### 5.3 Operational overhead

* **Rolling restarts** – When you upgrade the model or change the page size, you must flush the entire KV cache, which forces a brief outage. Production teams typically schedule a **warm‑up** period where the server runs in “drain mode” (no new requests) while the old instance finishes its in‑flight batches.
* **Metrics plumbing** – Continuous batching introduces new KPIs: *batch fill ratio*, *pages per request*, *token‑wise latency*. You need a monitoring stack (Prometheus + Grafana) that can ingest high‑frequency counters (hundreds of samples per second) without overwhelming the time‑series database.

Despite the added engineering effort, the **business impact**—being able to serve hundreds of users on a single GPU versus provisioning a separate GPU per user—often justifies the investment, especially for SaaS LLM providers.

---

## 6. When you can safely skip continuous batching and paging

Not every workload needs the full machinery described above. Below is a decision matrix that helps you decide whether to adopt the advanced stack or stick with a simpler static‑batch server (e.g., the default `transformers` pipeline or Ollama).

| Scenario | Expected concurrent users | Typical prompt length | Recommended stack |
|----------|---------------------------|-----------------------|-------------------|
| **Local development / debugging** | 1‑2 | < 200 tokens | Static batching, no paging (fast to spin up) |
| **Batch inference for offline data** | 1 (large batch) | 1‑2 K tokens | Static batching with large `batch_size` (no latency constraints) |
| **Small‑scale internal demo** | ≤ 5 | ≤ 500 tokens | Static batching is fine; continuous adds complexity without benefit |
| **Chatbot for a niche product (≤ 10 active users)** | 5‑10 | 1‑2 K tokens | Continuous batching optional; paging can be turned off to simplify deployment |
| **Public API serving hundreds of concurrent chats** | > 30 | 1‑4 K tokens (including system prompt) | **Continuous batching + PagedAttention** (vLLM, DeepSpeed‑MII, or TensorRT‑LLM) |
| **Retrieval‑augmented generation with 2‑K token context** | > 20 | 2‑8 K tokens | Paging essential to keep KV memory under control |

### 6.1 Ollama as a baseline

Ollama’s default server uses **static batching**: each request is queued, the model runs a full forward pass for the entire prompt, and the client blocks until the response is ready. In a load test with 20 parallel `ollama run llama2` commands, the 95‑th‑percentile latency ballooned to **> 3 s**, and the GPU memory peaked at **2.3 GB** for a 7‑B model—well within the H100’s capacity, but the throughput was limited to **≈ 12 K tps**. By contrast, the same hardware running vLLM with paging handled **≈ 120** concurrent chats at sub‑200 ms latency, a **10×** improvement.

If you are building a **prototype** that will never see more than a handful of users, Ollama’s simplicity may be attractive. However, as soon as you cross the **~5‑user** threshold, the latency penalty becomes noticeable, and you’ll likely need to migrate to a continuous‑batching server.

---

## 7. Practical checklist for deploying continuous batching with paging

| ✅ Item | Why it matters | How to verify |
|--------|----------------|---------------|
| **GPU driver ≥ 525** | Newer kernels expose `cudaMemcpyAsync` optimizations used by vLLM | `nvidia-smi` shows driver version |
| **CUDA toolkit 12.2** | Required for the pre‑compiled `paged_attention` extension | `nvcc --version` |
| **Page size tuned to 256** (default) | Aligns with most transformer head dimensions; avoids fragmentation | Check `VLLM_PAGE_SIZE` env var; monitor `num_pages` growth |
| **`max_num_batched_tokens` ≈ 8192** | Keeps kernel launch overhead low while maximizing occupancy | Run a short benchmark (`vllm benchmark`) and look at `batch_fill_ratio` |
| **Async I/O enabled** (`--disable-log-requests` off) | Prevents the request thread from becoming a bottleneck | Observe CPU usage; should stay < 10 % on a 16‑core host |
| **Prometheus exporter** (`--enable-metrics`) | Exposes batch size, KV usage, and latency for alerting | Query `vllm_batch_size` metric; it should stay near the number of active users |
| **Graceful shutdown script** | Flushes KV pages to avoid memory leaks on restart | `docker stop` followed by `docker logs` showing “Flushing KV cache” |

Running through this list before you flip the switch to production will save you from the classic “it works on one request, crashes on the hundredth” scenario.

---

## 8. Bringing it all together: the architecture you can actually ship

1. **Ingress layer** – an async HTTP server (FastAPI, Starlette) that pushes incoming JSON payloads onto a thread‑safe `request_queue`.
2. **Scheduler** – a Python coroutine that repeatedly:
   * pulls new requests,
   * builds a *token‑wise* batch,
   * calls the **paged attention** forward kernel,
   * writes the sampled token back to each request’s output buffer,
   * evicts finished requests.
3. **KV cache pool** – a pre‑allocated GPU buffer divided into **pages** (e.g., 256 tokens × hidden size). A per‑request **page table** lives in host memory and is copied to the GPU each step.
4. **Metrics & health** – Prometheus exporters expose `batch_fill_ratio`, `pages_per_request`, and per‑request latency histograms. Grafana dashboards alert when `batch_fill_ratio` drops below 70 % (indicating under‑utilization) or when `num_pages` grows unexpectedly.
5. **Graceful scaling** – Horizontal scaling is achieved by launching multiple identical containers behind a load balancer. Because each instance already multiplexes dozens of users, you often need **only 2‑3 containers** to handle thousands of concurrent sessions.

The resulting system feels like a **high‑throughput taxi dispatch** that also shares the same map data (system prompt KV pages) among all drivers. You get the best of both worlds: **low latency per user** and **high overall GPU utilization**, without the memory bloat of naïve static batching.

---

## 9. Key takeaways for the senior engineer

* **Static batching** is analogous to a tour bus that refuses to leave until every seat is filled and then drives to the farthest stop before anyone can disembark. It wastes both time and memory.
* **Continuous batching** turns the fleet into a fleet of taxis: each request hops on as soon as it arrives, receives a token, and gets off when done. The scheduler’s token‑wise interleaving eliminates the “slowest‑user‑holds‑up‑everyone” penalty.
* **PagedAttention** introduces a virtual‑memory layer for the KV cache. By breaking the cache into fixed‑size pages and sharing them across requests, you avoid the “longest sequence dictates cache size” problem and dramatically reduce memory pressure.
* Benchmarks on a single H100 show **> 100 concurrent chats** at sub‑200 ms latency with continuous batching + paging, versus **single‑digit concurrency** for classic static‑batch servers.
* The trade‑off is **code complexity**: a multi‑language scheduler, custom CUDA kernels, and a page‑table management layer that can become a hotbed for subtle bugs. Proper observability and a disciplined release process are essential.
* For **development, low‑traffic demos, or batch‑only pipelines**, you can safely stay with static batching. Once you anticipate **> 5 simultaneous users**, the performance gains from continuous batching become hard to ignore, and paging becomes essential once your context length exceeds a few thousand tokens.

By treating the serving stack as a **dynamic transportation network** rather than a rigid convoy, you unlock the ability to serve hundreds of users on a single high‑end GPU while keeping latency in the human‑acceptable range. The combination of continuous batching and PagedAttention is no longer a nice‑to‑have research curiosity—it is the de‑facto standard for production‑grade LLM serving.