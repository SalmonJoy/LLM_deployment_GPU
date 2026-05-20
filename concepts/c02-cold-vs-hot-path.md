# Cold-Path vs Hot-Path Optimization in LLM Serving

## 1. Why the distinction matters before you write a single line of code  

You have probably seen a ticket that says *“Our LLM is too slow”* and you instinctively start digging into GPU kernels, quantization tricks, or batch‑size gymnastics. In many organizations the first thing you do is fire up a profiler, look at FLOPs, and then spend weeks tuning the *hot* decoding loop.  

What you often miss is that the latency a user actually experiences is the sum of two very different phases:

| Phase | What happens | Where the time is spent | Typical latency range |
|-------|--------------|--------------------------|-----------------------|
| **Cold‑Path** | Load model weights from persistent storage into GPU memory (VRAM) | Disk I/O, OS page‑cache, PCIe transfer | 0.5 s – 10 min (depends on model size & storage) |
| **Hot‑Path** | Generate tokens using the weights already resident in VRAM | HBM bandwidth, compute kernels, KV‑cache traffic | 5 ms – 300 ms per token |

If you treat the whole service as a monolith, you will keep “optimizing the hot path” while the real user‑visible latency is dominated by the cold start. The opposite can happen too: you may over‑provision NVMe arrays for a workload that never warms up more than a handful of times per day, wasting dollars on a problem that never surfaces.

The rest of this article walks you through the *foundations* of each path, the *physics* that bound them, and the *decision tree* you can use to know which lever to pull. By the end you will have a concrete checklist, real commands you can copy‑paste, and a mental model that lets you diagnose a “slow LLM” ticket in under five minutes.

---

## 2. The serving pipeline in plain English  

Think of an LLM service as a two‑stage kitchen:

1. **Prep stage (cold‑path)** – You unpack a giant box of ingredients (the model weights) from the pantry (disk) and lay them out on the countertop (GPU memory). This is a one‑time cost per model version, per node, per warm‑up.
2. **Cooking stage (hot‑path)** – You repeatedly add a pinch of spice (the next token) to the pot, stir, and serve the dish. The ingredients are already on the countertop, so the bottleneck is how fast you can move the spoon (memory bandwidth) and how quickly the stove (GPU compute) can heat the pot.

If the pantry is a dusty back‑room with a slow door, the prep stage will dominate. If the pantry is right next to the kitchen and you have a high‑speed door, the cooking stage becomes the limiting factor.

In technical terms:

* **Cold‑Path** = loading *N* GB of checkpoint files (often 30 – 80 GB for 70 B‑parameter models) from persistent storage, mapping them into the process address space, and copying them over PCIe into VRAM.
* **Hot‑Path** = the token‑generation loop: matrix multiplications, attention kernels, KV‑cache reads/writes, and any post‑processing (detokenization, sampling).

Both paths share the same code base (e.g., `transformers` + `accelerate`), but they stress completely different hardware subsystems.

---

## 3. Analogy deep‑dive: cold mornings vs. highway cruising  

| Analogy | Cold‑Path | Hot‑Path |
|---------|-----------|----------|
| **Car** | Starting a cold engine on a frosty morning. The engine oil is thick, the battery is low, and you need to crank the starter motor long enough to get the pistons moving. | Driving at a steady 65 mph on a highway. The engine is warm, the transmission is in the optimal gear, and the limiting factor is aerodynamic drag and fuel efficiency. |
| **Plumbing** | Opening a valve that has been closed for weeks. Air bubbles and sediment must be flushed before water flows. | Maintaining a constant flow through a pipe that is already full of water; the limiting factor is pipe diameter (bandwidth) and pressure (latency). |
| **Kitchen** | Unpacking a 60‑lb turkey from the freezer, thawing it, and seasoning it before you can start cooking. | Stirring a simmering sauce; the heat transfer and spoon speed dictate how fast the sauce thickens. |

These analogies help you ask the right question when you see a latency spike:

* “Is the engine still cranking?” → Check if the model is being loaded.
* “Is the car stuck in first gear?” → Look at compute/ bandwidth utilization.

---

## 4. Cold‑Path is disk‑bound – numbers that matter  

### 4.1 Raw I/O throughput vs. effective throughput  

A 70 B‑parameter model stored in 65 GB of fp16 weights looks innocuous on paper, but the *effective* transfer speed is heavily throttled by the storage stack:

| Storage type | Nominal read speed | Real‑world effective speed (Linux `dd` with 1 MiB block) | Time to load 65 GB |
|--------------|-------------------|-----------------------------------------------------------|--------------------|
| Premium SSD (Azure) | ~250 MB/s | ~170 MB/s (≈ 30 % overhead from TLS, metadata) | 380 s (≈ 6 min) |
| Local NVMe (PCIe 3.0 x4) | 2 GB/s | 1.5 GB/s (OS page‑cache, readahead) | 43 s |
| Direct‑mapped NVMe (PCIe 4.0 x8, `mmap` + `readahead`) | 5 GB/s | 5 GB/s (near‑line) | 13 s |
| RAM‑disk (tmpfs) | 20 GB/s | 18 GB/s (CPU‑to‑CPU copy) | 3.6 s |

The *62×* speedup from a Premium SSD to a well‑tuned NVMe configuration is not a theoretical curiosity; it is a real lever you can pull by moving the checkpoint onto a local NVMe drive or by using a RAM‑disk for hot‑swap models.

### 4.2 Real command to measure your own load time  

```bash
# 1. Warm up the page cache (optional)
sudo dd if=/dev/zero of=/dev/null bs=1M count=0 seek=65536

# 2. Time a raw copy from storage to /dev/null (simulates load)
time dd if=/mnt/models/70b_fp16.bin of=/dev/null bs=4M iflag=direct
```

On a machine with a 2 TB NVMe, you should see something like:

```
real    0m13.2s
user    0m0.0s
sys     0m0.1s
```

If you get > 30 s, you are likely hitting a Premium SSD or a mis‑configured mount (e.g., `noatime,nodiratime` missing).

### 4.3 Mapping vs. copying  

Most serving frameworks (e.g., `vLLM`, `torchserve`) use `torch.load` which internally does a *copy* from the host buffer to GPU memory via PCIe. You can shave a few seconds by using `mmap` to let the OS page‑cache handle the transfer and by enabling `readahead`:

```python
import os, mmap, torch

# Pre‑advise the kernel about the upcoming sequential reads
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)

with open("/mnt/models/70b_fp16.bin", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    # Directly create a torch tensor from the memoryview (zero‑copy)
    weight_tensor = torch.frombuffer(mm, dtype=torch.float16).reshape(-1, 4096)
    # Pin to GPU (asynchronous copy)
    weight_tensor = weight_tensor.to("cuda", non_blocking=True)
```

The `POSIX_FADV_SEQUENTIAL` hint reduces page‑fault latency by allowing the kernel to pre‑fetch up to 1 GB ahead of the current read pointer. In practice you can see the load time drop from 13 s to **9 s** on the same NVMe.

### 4.4 Model pinning and lazy loading  

If you have a fleet of identical GPUs, you can *pin* the most popular model in VRAM at node start‑up:

```bash
# systemd service that runs before your inference server
/usr/bin/python -m torchrun \
    --nproc_per_node=8 \
    preload_model.py \
    --model-dir /mnt/models/70b_fp16.bin \
    --device cuda:0-7
```

`preload_model.py` simply loads the checkpoint and calls `torch.cuda.empty_cache()` after the copy, leaving the weights resident for the lifetime of the process. The cost is a few GB of VRAM per model, but the latency for the first request drops to **< 0.5 s** (GPU PCIe transfer only).

Lazy loading is the opposite extreme: you keep the model on a fast SSD and only copy the *shards* that are needed for the current request. This works for *Mixture‑of‑Experts* (MoE) models where only a subset of experts is active per token. The code pattern looks like:

```python
def get_expert_weights(expert_id):
    path = f"/mnt/models/moe_expert_{expert_id}.pt"
    # Load on‑demand, keep in LRU cache of size 4
    return torch.load(path, map_location="cuda")
```

The trade‑off is a small per‑token latency increase (≈ 2 ms) for a huge reduction in VRAM pressure.

---

## 5. Hot‑Path is HBM‑bound – why memory bandwidth is king  

### 5.1 The HBM bandwidth ceiling  

High‑Bandwidth Memory (HBM) on modern GPUs offers ~ 900 GB/s (A100) to ~ 2 TB/s (H100). That sounds huge, but the *attention* kernel in a decoder layer streams **K** and **V** matrices for every token. For a 70 B model with 128 k‑dimensional hidden state, a single token requires:

* **Q**: 128 k × 128 k = 16 M float16 multiply‑accumulate → 32 MB read, 32 MB write
* **K**, **V** (cached): 2 × 128 k × 128 k = 64 MB read
* **KV‑cache update**: 128 k × 128 k = 16 MB write

Total per‑token memory traffic ≈ **144 MB**. At 20 tokens/s (a typical decode rate for a 70 B model on a single A100), the required bandwidth is:

```
144 MB/token * 20 token/s = 2.88 GB/s
```

That looks tiny compared to 900 GB/s, but the *effective* bandwidth is limited by:

* **Cache line granularity** – each access is 32 B, causing many small reads.
* **Bank conflicts** – multiple threads hitting the same HBM bank.
* **KV‑cache fragmentation** – as the sequence length grows, the cache becomes less contiguous.

When you scale to a batch size of 8, the per‑token bandwidth requirement multiplies by 8, quickly approaching the HBM ceiling.

### 5.2 Profiling the hot path with `nsight`  

```bash
nsight-systems profile -o hotpath.nsys \
    python -m vllm.entrypoint --model 70b \
    --max-batch-size 8 --max-seq-len 2048
```

Open `hotpath.nsys` and look for the *Top 10 kernels* table. You will typically see:

| Kernel | % Time | HBM Read (GB) | HBM Write (GB) |
|--------|--------|---------------|----------------|
| `flash_attn_fwd` | 38 % | 1.9 | 0.9 |
| `layer_norm` | 12 % | 0.5 | 0.5 |
| `mlp_gelu` | 10 % | 0.8 | 0.4 |

If `flash_attn_fwd` dominates, you are bandwidth‑bound. If `layer_norm` dominates, you may be compute‑bound (unlikely for large models).

### 5.3 Hot‑path fixes that actually move the needle  

| Fix | Mechanism | Expected gain (typical) |
|-----|-----------|--------------------------|
| **KV‑cache quantization (FP8 / INT8)** | Reduce cache size → half the reads/writes | 1.5 × throughput |
| **Chunked attention (paged KV)** | Load only the most recent N tokens into fast HBM, keep older chunks in DDR | 1.2 × throughput for long sequences |
| **Tensor‑parallelism + GPU scaling** | Split the weight matrix across GPUs, each sees a smaller HBM load | Near‑linear scaling up to 8 GPUs |
| **Batch‑size tuning** | Larger batch → more compute per memory transaction, better utilization | 1.3 × throughput (but higher latency per request) |
| **Kernel fusion (flash‑attention v2)** | Combine QKV matmuls and softmax into a single kernel, reducing intermediate writes | 1.2 × throughput |

A concrete example: enabling FP8 KV‑cache in `vLLM` (v0.3+):

```bash
export VLLM_KV_CACHE_DTYPE=fp8
python -m vllm.entrypoint --model 70b --max-batch-size 8
```

On an H100, the per‑token latency drops from **45 ms** to **30 ms**, a **33 %** improvement without any hardware change.

---

## 6. The trap: mixing cold‑start and decode complaints  

Imagine you receive two tickets:

1. **Ticket A** – “The first request after a model swap takes 8 seconds.”
2. **Ticket B** – “Our chat UI feels laggy; each token appears after 200 ms.”

If you treat both as *the same problem*, you might:

* Deploy a new quantization scheme (hot‑path) that reduces token latency to 150 ms.
* Still see Ticket A users waiting 8 seconds because the model still loads from a remote SSD.

Conversely, you could:

* Move the checkpoint to a local NVMe (cold‑path) and see Ticket A drop to 0.8 s.
* Leave Ticket B untouched, still suffering 200 ms per token.

The key is to **measure** which phase dominates the *user‑visible* latency. A simple instrumentation snippet can give you the split:

```python
import time, torch

def load_and_warmup(model_path):
    t0 = time.time()
    model = torch.load(model_path, map_location="cuda")
    torch.cuda.synchronize()
    t1 = time.time()
    # Warm up a single forward pass
    dummy = torch.randn(1, 1, 4096, device="cuda")
    _ = model(dummy)
    torch.cuda.synchronize()
    t2 = time.time()
    return t1 - t0, t2 - t1   # (cold, hot)

cold, hot = load_and_warmup("/mnt/models/70b_fp16.pt")
print(f"Cold‑path: {cold:.2f}s, Hot‑path (first token): {hot*1000:.1f}ms")
```

If `cold` > 2 s, you are in a *cold‑dominant* regime. If `hot` > 150 ms, you are in a *hot‑dominant* regime.

---

## 7. Decision rules – when to pull which lever  

Below is a concise decision matrix you can keep on a wiki page. Fill in the columns with numbers from your own monitoring system (Prometheus, Grafana, or simple logs).

| Metric | Threshold | Action (Cold‑Path) | Action (Hot‑Path) |
|--------|-----------|--------------------|-------------------|
| **Cold‑start latency (p95)** | > 2 s | • Move checkpoint to local NVMe or RAM‑disk  <br>• Enable `mmap` + `readahead` <br>• Pin popular models at service start | – |
| **Cold‑start frequency** (cold starts / hour) | > 30 | • Reduce model‑swap frequency (cache warm models) <br>• Use a model‑registry that pre‑loads next‑most‑used model | – |
| **Per‑token latency (p95)** | > 150 ms | – | • Enable KV‑cache quantization (FP8/INT8) <br>• Increase batch size (if latency budget permits) <br>• Upgrade GPU or add tensor‑parallel nodes |
| **GPU HBM utilization** (avg %) | > 85 % | – | • Apply kernel fusion (flash‑attention v2) <br>• Split KV‑cache across DDR (paged KV) |
| **Sequence length > 2048** | Frequent | – | • Use chunked attention / sliding‑window <br>• Offload older KV‑cache to system RAM |
| **Model swap rate** (per day) | > 5 | • Consolidate models (e.g., merge LoRA adapters) <br>• Use a “warm pool” of GPUs dedicated to hot models | – |

**Rule of thumb:** if the *cold‑start latency* contributes **> 30 %** of the end‑to‑end request latency *and* you see more than **10 cold starts per hour**, prioritize cold‑path fixes. Otherwise, focus on hot‑path optimizations.

---

## 8. Anti‑pattern case study: weeks spent on hot‑path while cold‑start was the culprit  

**Background** – A SaaS provider runs a 13 B‑parameter model on a fleet of A100‑40GB GPUs. Users report “slow responses” after a weekend deployment. The engineering team spent **four weeks** rewriting the inference loop to use a custom fused attention kernel, expecting a 20 % speed‑up.

**What they measured**  

| Phase | Avg latency (ms) | % of total |
|-------|------------------|------------|
| Disk load (first request) | 8 800 ms | 78 % |
| Token decode (subsequent) | 2 500 ms | 22 % |

The custom kernel reduced the decode latency from 2 500 ms to 2 200 ms (12 % gain), but the overall user‑visible latency only improved from **11.3 s** to **11.0 s** – a negligible change.

**Root cause** – The model checkpoint lived on a **Premium SSD** mounted with default `noatime` options, resulting in 170 MB/s effective read speed. The team had ignored the *cold‑path* metric entirely.

**What they should have done**  

1. **Move the checkpoint to local NVMe** – `dd if=/dev/zero of=/mnt/nvme/70b.bin bs=1M count=65536` to pre‑allocate and then `mv` the file.
2. **Enable `readahead`** – `sudo blockdev --setra 4096 /dev/nvme0n1`.
3. **Pin the model** – Add a systemd service that loads the model at boot (see Section 4.3).

Result after the fix:

| Phase | Avg latency (ms) | % of total |
|-------|------------------|------------|
| Disk load (first request) | **820 ms** | 8 % |
| Token decode (subsequent) | 2 200 ms | 92 % |

Now the hot‑path truly dominates, and the team’s later effort on kernel fusion gave a **30 %** end‑to‑end improvement (down to **2.9 s** per request). The lesson: **measure first, then optimize**.

---

## 9. Advanced cold‑path tricks you can apply today  

### 9.1 Asynchronous pre‑fetching  

If you know the next model version will be needed (e.g., a nightly rollout), start loading it *while* you are still serving the current model.

```python
import asyncio, torch

async def async_preload(path, device="cuda"):
    loop = asyncio.get_event_loop()
    # Offload I/O to a thread pool
    weight = await loop.run_in_executor(None, torch.load, path, "cpu")
    # Async copy to GPU
    await loop.run_in_executor(None, weight.to, device, True)
    return weight

# In your request handler:
if request.model_version != current_version:
    asyncio.create_task(async_preload(request.model_path))
```

The first request after the switch will see the model already resident, shaving off the entire cold‑path latency.

### 9.2 Multi‑stage checkpoint sharding  

Instead of a monolithic 65 GB file, split the checkpoint into **layer groups** (e.g., 8 GB per 8 layers). Load the first group synchronously (required for the first token) and stream the remaining groups in the background. This reduces the *perceived* cold start to the time needed for the first group:

```bash
# Split with torch.save
python - <<'PY'
import torch, os
ckpt = torch.load("70b_full.pt", map_location="cpu")
layer_groups = [ckpt[i*8:(i+1)*8] for i in range(8)]
os.makedirs("shards", exist_ok=True)
for i, grp in enumerate(layer_groups):
    torch.save(grp, f"shards/group_{i}.pt")
PY
```

During serving, load `group_0.pt` before the first forward pass, then launch `torch.load` for `group_1..7` in separate threads. The user sees a **cold‑path** of ~ 2 seconds instead of 13 seconds.

### 9.3 Leveraging cloud‑native block storage  

If you run on Kubernetes, you can request a **local persistent volume** (PV) that maps a node‑local NVMe directly into the pod:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: local-nvme-pv
spec:
  capacity:
    storage: 2Ti
  accessModes:
    - ReadWriteOnce
  storageClassName: local-nvme
  local:
    path: /dev/nvme0n1
  nodeAffinity:
    required:
      nodeSelectorTerms:
        - matchExpressions:
            - key: kubernetes.io/hostname
              operator: In
              values: ["gpu-node-01"]
```

Mount this PV into `/mnt/models` and you get the NVMe performance without any network hop. The cold‑path latency drops to the numbers shown in Section 4.2.

---

## 10. Hot‑path refinements for the seasoned engineer  

### 10.1 KV‑cache layout optimization  

The default layout stores **K** and **V** interleaved per layer, which leads to *stride* accesses when the sequence grows. Re‑ordering the cache to **row‑major** per token reduces cache‑line misses.

```cpp
// Pseudo‑C++ for custom KV layout
struct KVCache {
    // [batch, seq_len, heads, head_dim]
    float16* K;   // contiguous in seq_len dimension
    float16* V;   // same layout
};
```

A benchmark on an H100 shows a **5 %** per‑token latency reduction for sequences > 4096 tokens.

### 10.2 Dynamic batch sizing  

Instead of a static `max_batch_size`, implement a *batch‑size scheduler* that groups requests arriving within a 2 ms window. This technique, often called **adaptive batching**, can raise GPU utilization from 45 % to 78 % without increasing tail latency beyond 10 ms.

```python
from collections import deque
import time, threading

batch_queue = deque()
def request_handler(req):
    batch_queue.append(req)
    # Return a future; actual response will be filled later

def batch_worker():
    while True:
        start = time.time()
        while len(batch_queue) < 1 and (time.time() - start) < 0.002:
            time.sleep(0.0005)
        batch = [batch_queue.popleft() for _ in range(min(len(batch_queue), 8))]
        # Run model on batch
        outputs = model.forward(batch)
        for req, out in zip(batch, outputs):
            req.future.set_result(out)

threading.Thread(target=batch_worker, daemon=True).start()
```

In production this pattern is used by `vLLM` and `tgi` (Text Generation Inference) under the flag `--max-batch-size` combined with `--batch-schedule=dynamic`.

### 10.3 Mixed‑precision inference beyond FP16  

The newest GPUs support **FP8** for both compute and memory. Switching the *compute* dtype to FP8 while keeping the *weights* in FP16 yields a **1.8 ×** speed‑up for the hot path with < 0.1 % perplexity loss.

```bash
export VLLM_COMPUTE_DTYPE=fp8
python -m vllm.entrypoint --model 70b
```

Make sure your CUDA driver is ≥ 12.2; otherwise the kernel fallback will silently revert to FP16.

### 10.4 Kernel‑level profiling for micro‑optimizations  

When you have already exhausted the high‑level knobs, dive into kernel‑level metrics:

```bash
nsight-cu-cli -k flash_attn_fwd -o flash.txt \
    python -c "import vllm; vllm.run(...)" 
```

Look for:

* **SM occupancy** < 60 % → consider increasing block size.
* **L2 hit rate** < 70 % → try `torch.backends.cudnn.benchmark = True` to let cuDNN pick better algorithms.
* **Shared memory usage** > 48 KB per block → may cause spill to global memory.

Tuning these low‑level parameters can shave another **5‑10 %** off the hot‑path latency, but the ROI diminishes quickly.

---

## 11. Putting it all together – a step‑by‑step workflow  

1. **Instrument** – Add the timing snippet from Section 6 to every request path. Log `cold_ms` and `hot_ms` separately.
2. **Collect** – Run a 24‑hour experiment under typical load. Export the data to a CSV and compute percentiles.
3. **Classify** – If `cold_ms_p95` > 2 s **or** `cold_start_rate` > 10 %/hour, you are cold‑dominant. Otherwise you are hot‑dominant.
4. **Apply the matrix** – Use the decision table in Section 7 to pick the first set of actions.
5. **Validate** – Re‑run the experiment, compare the new percentiles. If the target latency is still not met, iterate on the other path.
6. **Automate** – Encode the chosen actions in your CI/CD pipeline:
   * **Cold‑path** – `preload_model.service`, `nvme‑mount.yaml`, `model‑sharding.py`.
   * **Hot‑path** – `vllm` flags (`--kv-cache-dtype=fp8`, `--tensor-parallel-size=4`), `nsight` baseline check.

By making the measurement step mandatory, you avoid the classic *“spend weeks on hot‑path while cold‑start is the real problem”* anti‑pattern.

---

## 12. Checklist for the senior engineer on call  

| ✅ Item | Description | Command / Config |
|--------|-------------|------------------|
| **Cold‑path latency** | Verify load time < 1 s for 65 GB model on NVMe | `time dd if=/mnt/models/70b.bin of=/dev/null bs=4M iflag=direct` |
| **Model pinning** | Keep hot models resident across requests | Systemd service `preload_model.py` |
| **Lazy loading / sharding** | Load only required layers on demand | `torch.load(..., map_location="cpu")` + `mmap` |
| **KV‑cache dtype** | Use FP8 or INT8 to halve bandwidth | `export VLLM_KV_CACHE_DTYPE=fp8` |
| **Batch scheduler** | Enable dynamic batching (≤ 2 ms window) | `--batch-schedule=dynamic` |
| **Kernel version** | Flash‑attention v2 or later | `pip install flash-attn==2.0.0` |
| **HBM utilization** | Keep < 80 % to avoid saturation | `nsight-systems profile …` |
| **Monitoring** | Export `cold_ms`, `hot_ms`, `seq_len` to Prometheus | `pushgateway` or `statsd` integration |
| **Rollback plan** | Ability to revert to previous checkpoint instantly | Keep a symlink `current_model -> /mnt/models/70b_fp16.pt` |

Running through this list after each deployment guarantees you are not inadvertently re‑introducing a cold‑path regression while chasing hot‑path gains.

---

## 13. Final thoughts – a mental model you can carry everywhere  

When you think about LLM serving, picture **two independent pipelines** that converge only at the moment a request finishes:

1. **The loading conveyor belt** – moves massive weight files from disk to GPU memory. Its speed is dictated by the *type of floor* (SSD vs. NVMe) and the *conveyor motor* (mmap, readahead, pinning).  
2. **The cooking stove** – repeatedly stirs a pot of attention. Its speed is limited by *how fast you can move the spoon* (HBM bandwidth) and *how hot the flame is* (GPU compute power, kernel efficiency).

If the conveyor belt is jammed, no amount of stove‑tuning will make dinner arrive faster. If the belt is smooth but the stove is under‑powered, you’ll still wait forever for the next bite.

By **measuring**, **classifying**, and then **applying the right lever**, you turn a vague “LLM is slow” complaint into a concrete engineering story with numbers, commands, and a clear path to improvement. The next time a stakeholder asks for “better performance,” you’ll be able to say:

> “We’ve profiled the cold‑path and it’s taking 9 seconds because the model lives on a remote SSD. Moving it to local NVMe and enabling async pre‑fetch will cut that to sub‑second. After that, we’ll look at KV‑cache quantization to shave another 30 % off token latency.”

Armed with the distinctions, analogies, and concrete knobs laid out in this article, you can now diagnose and optimize LLM serving pipelines with surgical precision. Happy serving!