# The 80/20 of LLM Optimization: Which Knobs Pay Off

---

## 1. Why “Pareto” Is Not Just a Buzzword in LLM Serving  

If you’ve ever watched a Formula 1 pit crew, you know that **80 % of the lap time comes from a handful of adjustments**—tire pressure, wing angle, fuel load. The rest of the crew is busy polishing the dashboard lights, swapping out the spare‑wheel lug nuts, and polishing the driver’s helmet. The same Pareto principle applies to large‑language‑model (LLM) serving: a tiny set of configuration knobs delivers the bulk of latency and throughput gains, while the rest are decorative.

Imagine you’re trying to squeeze a 2‑hour inference job into a 30‑second window. You could spend a week rewriting the transformer kernel in CUDA, or you could move the model files from a cold SATA disk to a hot NVMe SSD and enable flash‑attention. The first effort might shave off a few milliseconds; the second can cut **cold‑load time from 60 seconds to 1 second**—a 60× improvement for almost no engineering effort.

The danger of “tuning the wrong knobs” is not just wasted time; it’s also a hidden cost in cloud spend. A 5 % latency gain on a request that already spends 90 % of its wall‑clock time waiting for the KV cache to be populated is essentially a **zero‑sum game**. You’ll be paying for extra GPU minutes without ever seeing a measurable ROI.

In the sections that follow you’ll get:

* A **ranked, data‑driven list** of the knobs that actually move the needle.
* A **counter‑list of over‑hyped levers** that sound impressive but rarely pay off.
* A **step‑by‑step method** to profile your own stack, map the bottlenecks, and pick the right lever.
* A **meta‑rule** that helps you decide whether a claimed 2× win is credible.
* A **real‑world compounding example** that shows how three 1‑line changes can give you > 100× concurrency.

You’ll finish with a concrete checklist you can run against any LLM deployment—whether you’re on an on‑prem GPU farm, a managed inference endpoint, or a hobbyist laptop.

---

## 2. The LLM Serving Pipeline in Plain English  

Before we start pulling knobs, let’s demystify the **four major cost centers** that dominate inference latency and cost:

| Stage | What Happens | Typical % of Wall‑Clock Time* |
|-------|--------------|-------------------------------|
| **Cold Load** | Model weights read from storage → deserialized → copied into GPU memory | 30–70 % (depends on storage tier) |
| **KV Cache Allocation** | Allocate and optionally quantize the key‑value (KV) cache for the requested context length | 10–25 % |
| **Attention Compute** | Matrix multiplications for each transformer layer (dominates compute‑bound workloads) | 20–35 % |
| **Batch Management** | Packing multiple requests into a single forward pass (or falling back to single‑request mode) | 5–15 % |

\*Numbers are from a 70 B model on an NVIDIA H100, measured with `torch.profiler` on a 32 K token request. Your exact percentages will differ, but the shape is universal.

Think of the pipeline as a **kitchen**:

1. **Cold Load** is the time you spend fetching ingredients from the pantry and chopping them. If the pantry is a basement freezer (slow HDD), you’ll waste minutes before you can even start cooking.
2. **KV Cache Allocation** is like pre‑heating the oven and setting the right rack positions. If you use a tiny oven (low‑precision cache) you may have to run the dish multiple times.
3. **Attention Compute** is the actual cooking—stirring, sautéing, blending. This is where the chef’s skill (GPU compute) matters most.
4. **Batch Management** is the service line: how many orders you can fulfill at once without sending the chef back to the prep table for each plate.

Just as a restaurant can double its throughput by moving the pantry to the kitchen door, a model server can double its request rate by moving the model files to a fast NVMe tier. The analogy also shows why **over‑optimizing the garnish (e.g., tweaking a log level) never feeds more customers**.

---

## 3. The High‑ROI Knobs, Ranked  

Below is the **Pareto‑ordered list** of levers that give you the biggest bang for the buck. For each knob we give:

* **What it does** (in plain language)
* **Measured ROI** (speedup, memory reduction, or cost reduction)
* **Implementation effort** (roughly how many minutes, lines, or environment variables)
* **Concrete commands / config snippets** you can copy‑paste

| Rank | Knob | ROI (Typical) | Effort |
|------|------|---------------|--------|
| 1 | **Disk tier for model storage** (Premium SSD → local NVMe) | 60× cold‑load reduction (e.g., 60 s → 1 s) | 10 min (mount + symlink) |
| 2 | **KV cache quantization** (fp16 → q8_0) | 50 % KV size, 87 % smaller with flash‑attention | 2 env vars |
| 3 | **Context cap** (128 K → 32 K) | 4× KV memory reduction, 0 % latency impact if not needed | 1 config flag |
| 4 | **Continuous batching engine** (vLLM vs Ollama) | 50–100× throughput for concurrent workloads | Migration (≈1 day) |
| 5 | **Flash attention ON** (Hopper+) | 1.5–2× attention speed, lower memory pressure | 1 env var |
| 6 | **Restart policy `unless‑stopped`** | Eliminates downtime from crashes, saves up to 20 % GPU‑hour waste | 1 flag |
| 7 | **Pin model to VRAM** (`OLLAMA_KEEP_ALIVE=-1`) | Removes warm‑load latency on every request (≈0.5 s saved) | 1 env var |

### 3.1 Disk Tier for Model Storage  

**Why it matters** – The model weights for a 70 B parameter LLM are roughly **140 GB** in fp16. Loading that from a **standard SATA SSD** (≈500 MB/s) takes ~ 280 seconds. A **local NVMe** (≈3 GB/s) shaves that to ~ 45 seconds, and a **PCIe 4.0 NVMe** can hit 5 GB/s, bringing the load down to ~ 30 seconds. In practice, when you combine **direct‑IO** and **pre‑warming**, you can hit **1 second** cold‑load on a 40 GB checkpoint.

**Real‑world numbers** (AWS `i3en.large` vs `m5.large`):

| Instance | Storage | Cold‑load (70 B) | Speedup |
|----------|---------|------------------|---------|
| `m5.large` (EBS gp3, 125 MB/s) | 60 s | 1× |
| `i3en.large` (NVMe 3 GB/s) | 1.2 s | 50× |
| `i4i.large` (NVMe 5 GB/s) | 0.8 s | 75× |

**Implementation steps** (Linux, Docker‑based deployment):

```bash
# 1. Identify a fast NVMe mount point, e.g. /mnt/nvme0
sudo mkdir -p /mnt/nvme0/models
sudo chown $(whoami):$(whoami) /mnt/nvme0/models

# 2. Copy the model checkpoint (once)
rsync -avh /data/models/llama-2-70b/ /mnt/nvme0/models/llama-2-70b/

# 3. Update your container to point at the new location
docker run -d \
  -e OLLAMA_MODEL_PATH=/mnt/nvme0/models \
  -v /mnt/nvme0/models:/models \
  --name ollama \
  ollama/serve
```

That’s it. The container now reads from the NVMe device on every cold start. The **cost** is a single `rsync` (≈ 5 GB/min on a 10 Gbps network) and a couple of minutes to edit the launch script.

### 3.2 KV Cache Quantization  

The **KV cache** holds the key and value vectors for every token that has already been processed. For a 70 B model with 32 K context, the KV cache in fp16 occupies:

```
tokens × layers × hidden_dim × 2 (K+V) × 2 bytes
= 32,000 × 80 × 4096 × 2 × 2 ≈ 41 GB
```

That’s a **memory hog** that forces you to split the model across multiple GPUs or to truncate the context.

**Quantizing to `q8_0`** (8‑bit integer with per‑tensor scale) reduces the cache to **≈ 5 GB**, a **~ 8× reduction**. In practice you’ll see **~ 50 % less memory pressure** because the runtime still allocates some fp16 buffers for the attention matmul.

**How to enable** (vLLM example):

```bash
export VLLM_KV_CACHE_QUANTIZATION=q8_0   # or q4_0 for even smaller
export VLLM_ATTENTION_TYPE=flash_attn   # flash‑attention works best with quantized cache
docker run -d \
  -e VLLM_KV_CACHE_QUANTIZATION=$VLLM_KV_CACHE_QUANTIZATION \
  -e VLLM_ATTENTION_TYPE=$VLLM_ATTENTION_TYPE \
  -p 8000:8000 \
  vllm/vllm:latest \
  --model /models/llama-2-70b
```

Only **two environment variables** are required. The **runtime** automatically selects a quantized cache implementation if the hardware supports it (most modern GPUs do). The **speed impact** is neutral or slightly positive because the smaller cache fits into L2, reducing memory traffic.

### 3.3 Context Cap  

Many production workloads request **128 K tokens** just to be safe, but the **average useful context** is often under **8 K**. Reducing the maximum context length from 128 K to 32 K cuts the KV cache size **linearly**:

```
KV size ∝ context_length
```

So a 4× reduction in context reduces KV memory by 4× *without* touching the model weights.

**Configuration** (Ollama example):

```yaml
# ollama.yaml
model:
  name: llama-2-70b
  max_context_length: 32768   # 32K tokens
```

Or via CLI:

```bash
ollama serve --max-context 32768
```

If you have a **multi‑tenant API**, you can expose the `max_context` as a per‑client quota, ensuring that nobody accidentally triggers a 128 K allocation.

**Result**: For a 70 B model, the KV cache drops from ~ 41 GB to ~ 10 GB, freeing up **~ 30 GB** of VRAM that can be used for **larger batch sizes** or **additional models**.

### 3.4 Continuous Batching Engine (vLLM vs Ollama)  

**Batching** is the difference between a single‑threaded kitchen line and a **conveyor‑belt**. Ollama’s default server processes each request in isolation, which is fine for a single user but collapses under concurrency. **vLLM** implements *continuous batching*: it keeps a pool of ready‑to‑run tensors and dynamically packs incoming prompts into the same forward pass.

**Benchmark** (H100, 70 B, 32 K context, 8 concurrent users):

| Engine | Throughput (req/s) | Latency (p99) | Speedup |
|--------|-------------------|---------------|---------|
| Ollama (single‑request) | 0.2 | 5 s | 1× |
| vLLM (continuous) | 12 | 0.4 s | 60× |

Even a **conservative 50×** improvement is common when the workload has any concurrency. The **effort** is a one‑day migration:

1. Pull the vLLM image.
2. Point it at the same model directory.
3. Update your client to call the new endpoint.

```bash
docker run -d \
  -p 8000:8000 \
  -v /mnt/nvme0/models:/models \
  vllm/vllm:latest \
  --model /models/llama-2-70b \
  --max-model-len 32768 \
  --tensor-parallel-size 1
```

### 3.5 Flash Attention ON  

Flash attention replaces the naïve `O(N²)` attention kernel with a **tiling algorithm** that streams data through the GPU’s shared memory, reducing both memory bandwidth and latency. On NVIDIA **Hopper** (H100) the speedup is **1.5–2×** for long contexts; on Ampere (A100) you’ll see **1.2–1.4×**.

**Enable it** (vLLM or Transformers):

```bash
export VLLM_ATTENTION_TYPE=flash_attn
# For HuggingFace Transformers (>=4.34)
export TORCH_COMPILE=1
export FLASH_ATTENTION=1
```

No code changes required; the runtime swaps the kernel at import time. The **cost** is a single environment variable and a restart.

### 3.6 Restart Policy `unless‑stopped`  

When a GPU process crashes (e.g., OOM, driver reset), Docker’s default `no` policy leaves the container in a stopped state. A **restart policy** of `unless-stopped` automatically brings the server back up, preserving the **pinned model** in VRAM (if you also set `OLLAMA_KEEP_ALIVE=-1`).

```bash
docker run -d \
  --restart unless-stopped \
  -e OLLAMA_KEEP_ALIVE=-1 \
  -p 11434:11434 \
  ollama/serve
```

The **ROI** is not a pure speedup but a **reduction in lost GPU‑hour**. In a production environment with a 5 % crash rate, you can reclaim **~ 20 %** of your allocated GPU budget.

### 3.7 Pin Model to VRAM (`OLLAMA_KEEP_ALIVE=-1`)  

By default, Ollama unloads the model after a short idle period (≈ 30 s). The next request incurs a **warm‑load** of ~ 0.5 s to copy weights from system RAM to GPU VRAM. Setting `OLLAMA_KEEP_ALIVE=-1` tells the server to **keep the model resident** indefinitely.

```bash
export OLLAMA_KEEP_ALIVE=-1
docker run -d \
  -e OLLAMA_KEEP_ALIVE=$OLLAMA_KEEP_ALIVE \
  -p 11434:11434 \
  ollama/serve
```

The **cost** is a single env var. The **benefit** is a **constant 0.5 s latency reduction** per request, which adds up quickly when you have high QPS.

---

## 4. The Over‑Hyped Knobs That Rarely Move the Needle  

Below is a list of levers that sound impressive on paper but, in most real‑world deployments, deliver **negligible ROI** relative to effort.

| Knob | Why It Looks Good | Why It Usually Doesn’t Pay Off |
|------|-------------------|--------------------------------|
| **Bigger GPU (more TFLOPs)** | More raw compute → lower per‑token time. | Most serving workloads are **memory‑bound** (KV cache) or **I/O‑bound** (cold load). Only *compute‑bound* workloads (e.g., dense embeddings, heavy sampling) benefit. |
| **More VRAM** | Allows larger context or multi‑model hosting. | Latency is dominated by KV cache size, not by the extra VRAM itself. Adding 64 GB to a 80 GB card rarely reduces per‑request latency unless you also shrink the cache (see §3.3). |
| **Lower Temperature** | “Cleaner” outputs, sometimes faster decoding. | Temperature only affects the **sampling distribution**, not the number of tokens generated. The runtime cost is identical; you might even increase latency due to more beam search steps if you lower it too far. |
| **Disabling Safety Filters** | Removes a few extra conditional checks. | The safety filter adds **≈ 1 ms** per token on a 70 B model—well within measurement noise. The risk (toxic output, policy violation) far outweighs the tiny gain. |
| **Speculative Decoding** | Theoretically 2× speedup by predicting next token. | Requires a **draft model** and complex orchestration. Gains are model‑dependent; for LLaMA‑2 70 B you typically see **≤ 1.2×** and you add **≈ 0.5 GB** of extra VRAM for the draft. Integration effort is high (custom inference loop, extra Docker image). |

**Rule of thumb**: If a vendor or blog post claims “2× speedup by adding a bigger GPU,” ask for a **baseline measurement** that isolates the GPU compute component (e.g., run a pure matrix‑multiply benchmark). If the baseline already shows **> 90 % memory traffic**, the claim is likely **noise**.

---

## 5. A Systematic Method to Find Your “High‑ROI” Lever  

### 5.1 Profile, Don’t Guess  

1. **Instrument the server** with a low‑overhead profiler. For PyTorch‑based servers, `torch.profiler` with `schedule=trace` works well. For vLLM, enable `--log-level=debug` and parse the CSV output.
2. **Collect three key metrics** for a representative request:
   * **Cold‑load time** (`t_load`)
   * **KV cache allocation time** (`t_kv`)
   * **Attention compute time** (`t_attn`)
   * **Batch wait time** (`t_batch`)
3. **Compute percentages**:

```python
total = t_load + t_kv + t_attn + t_batch
print(f"Cold load: {t_load/total:.1%}")
print(f"KV cache: {t_kv/total:.1%}")
print(f"Attention: {t_attn/total:.1%}")
print(f"Batching: {t_batch/total:.1%}")
```

If `t_load` > 30 %, you’ve found a **cold‑load bottleneck** → apply the **NVMe** knob first. If `t_kv` dominates, look at **KV cache quantization** and **context cap**. If `t_attn` is the biggest slice, enable **flash attention**. If `t_batch` is high, you need **continuous batching**.

### 5.2 Map the Bottleneck to the Ranked List  

| Dominant % | Recommended High‑ROI Knob(s) |
|------------|------------------------------|
| > 30 % cold load | Disk tier (NVMe) |
| 20–30 % KV cache | KV cache quantization, context cap |
| 15–25 % attention | Flash attention, speculative decoding (if you’re willing to code) |
| > 10 % batching | Continuous batching engine |
| < 5 % (all low) | Consider over‑hyped knobs only after confirming compute‑bound nature |

### 5.3 Validate the Change  

After applying a knob, **re‑run the same profiling script**. The percentages should shift, and you can iterate. A typical workflow:

```bash
# 1. Baseline
python profile.py --model llama-2-70b --prompt "Explain quantum tunneling."

# 2. Apply NVMe
# (mount, symlink, restart)

# 3. Re‑profile
python profile.py ...

# 4. Apply KV quantization
export VLLM_KV_CACHE_QUANTIZATION=q8_0
docker restart vllm

# 5. Re‑profile again
python profile.py ...
```

If you see **diminishing returns** (e.g., after NVMe and KV quant you’re now at 5 % cold load, 60 % attention), stop and move to the next tier.

### 5.4 Cost‑Benefit Spreadsheet  

Create a tiny spreadsheet with columns:

| Knob | Implementation Cost (min) | Expected Speedup | Estimated Savings ($/mo) |
|------|----------------------------|------------------|--------------------------|
| NVMe | 10 | 60× cold‑load | 0.5 % of GPU‑hour |
| KV q8_0 | 2 | 2× memory | 5 % of GPU‑hour |
| Flash Attn | 1 | 1.7× | 2 % |
| vLLM | 480 | 70× throughput | 30 % |

The **ROI** is `Savings / Cost`. The top‑ranked knobs will have **ROI > 1000** (tiny effort, huge savings).

---

## 6. The Meta‑Rule: “Do You Have a 2× Claim From a Comparable Workload?”  

Before you invest time in any optimization, ask yourself:

> **“Is there a published benchmark on a model, hardware, and workload that is within 10 % of mine, showing at least a 2× improvement for this exact knob?”**

If the answer is **no**, treat the claim as **unverified** and **low priority**. This rule prevents you from chasing **shiny objects** like speculative decoding or custom CUDA kernels that only work on a specific transformer variant.

**How to verify quickly:**

1. Search the **GitHub issues** of the runtime (e.g., `vllm/vllm#1234`).
2. Look for **MLPerf** or **HuggingFace** inference benchmark tables that include the knob.
3. Check the **hardware** column; if it’s a newer GPU generation, factor in the compute delta.

If you find a **peer‑reviewed or production‑grade case** (e.g., “Flash attention on H100 reduces latency from 12 ms to 7 ms on 32 K context”), you can safely allocate the 1‑minute effort.

---

## 7. Compounding Wins: A Real‑World Walkthrough  

Let’s walk through a **hypothetical SaaS** that serves a 70 B LLM to 200 concurrent users, each sending an average of 2 K tokens per request. The baseline architecture is:

* **Instance**: `g5.12xlarge` (4 × H100, 80 GB VRAM each)
* **Storage**: EBS gp3 (125 MB/s)
* **Server**: Ollama single‑request mode
* **Cold‑load**: 45 s (model already in GPU memory after warm‑up)
* **KV cache**: fp16, 32 K context → 41 GB per GPU (requires model sharding)
* **Throughput**: 0.3 req/s per GPU (≈ 0.12 req/s overall)

### Step‑by‑Step Optimizations

| Step | Change | Immediate Effect | New Throughput |
|------|--------|-------------------|----------------|
| 0 | Baseline | 0.12 req/s | 0.12 |
| 1 | Move model to local NVMe (Tier 1) | Cold‑load ↓ 45 s → 1 s (60×) | No change in steady‑state QPS, but **startup time** drops from 5 min to 4 s. |
| 2 | KV cache quantization (`q8_0`) + context cap 32 K | KV memory ↓ 8× → 5 GB per GPU; frees 36 GB VRAM | **Batch size** can increase from 1 → 7 per GPU (7×). |
| 3 | Switch to vLLM continuous batching | Throughput ↑ 70× (concurrency) | **Overall QPS** = 0.12 × 70 × 7 ≈ **58 req/s** |
| 4 | Enable flash attention | Per‑token latency ↓ 1.7× | QPS ↑ additional 1.3× → **≈ 75 req/s** |
| 5 | Set `OLLAMA_KEEP_ALIVE=-1` + restart policy | Eliminates 0.5 s warm‑load per request (≈ 1 % latency) | Negligible QPS change, but **99.9 % uptime**. |

**Result**: From a **0.12 req/s** baseline to **≈ 75 req/s**—a **> 600×** improvement. The first three knobs together account for **> 99 %** of the gain, each requiring **≤ 5 minutes** of work (NVMe mount, two env vars, Docker image swap). The remaining two knobs add a **modest but measurable** extra boost.

**Cost comparison**:

| Resource | Baseline Monthly GPU‑hour | Optimized Monthly GPU‑hour |
|----------|---------------------------|----------------------------|
| 4 × H100 | 4 × 720 h × $3.00 = $8,640 | Same GPUs, but **75× higher utilization**, so you can **downsize** to 1 × H100 and stay under $2,200. |

The **financial impact** is a **~ 75 % reduction in cloud spend** while delivering a **100× better user experience**.

---

## 8. Checklist: Turn the Knobs On, One‑Line at a Time  

| ✅ | Action | Command / Config |
|----|--------|------------------|
| 1 | Mount fast NVMe and point model path | `ln -s /mnt/nvme0/models /models` |
| 2 | Enable KV cache quantization | `export VLLM_KV_CACHE_QUANTIZATION=q8_0` |
| 3 | Reduce max context length | `ollama serve --max-context 32768` |
| 4 | Deploy vLLM continuous batching | `docker run -p 8000:8000 -v /models:/models vllm/vllm:latest --model /models/llama-2-70b --max-model-len 32768` |
| 5 | Turn on flash attention | `export VLLM_ATTENTION_TYPE=flash_attn` |
| 6 | Set restart policy | `docker run --restart unless-stopped …` |
| 7 | Keep model in VRAM | `export OLLAMA_KEEP_ALIVE=-1` |

Apply them **in order of ROI** (NVMe → KV → Context → Batching → Flash → Restart → Keep‑Alive). After each step, **run the profiling script** to confirm the expected shift in percentages.

---

## 9. The Real Challenge: Execution, Not Knowledge  

You now have a **catalog** of the knobs that actually matter, a **methodology** to discover which one hurts you the most, and a **meta‑rule** to filter out hype. The remaining obstacle is **execution**:

* **Automation** – bake the env‑var changes into your CI/CD pipeline so every new model version inherits the same settings.
* **Observability** – ship the profiling data to a Grafana dashboard; set alerts when `t_load` exceeds 5 seconds.
* **Governance** – enforce the “2× claim” rule in pull‑request reviews; require a benchmark link before merging any performance‑related change.

When you embed these practices, the **Pareto principle becomes a habit**, not a one‑off checklist. You’ll spend most of your engineering time on the **few levers that truly matter**, and you’ll stop chasing the glitter of bigger GPUs, extra VRAM, or speculative decoding that never materializes into real savings.

In practice, the **fastest path to a production‑grade LLM service** is:

1. **Put the model on NVMe** (minutes).  
2. **Quantize the KV cache** (two env vars).  
3. **Cap the context** to what your API actually needs.  
4. **Swap to a continuous batching runtime** (vLLM).  
5. **Enable flash attention** (one env var).  

If you can get those five steps into your deployment pipeline, you’ll already be **orders of magnitude ahead** of the “default‑out‑of‑the‑box” experience. The rest—restart policies, keep‑alive flags, and safety‑filter tweaks—are polish that protects you from downtime and ensures a smooth user experience.

Now go ahead, open a terminal, and start turning those high‑ROI knobs. The only thing standing between you and a 100× throughput boost is a **few lines of configuration**.