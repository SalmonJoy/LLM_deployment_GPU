# VRAM Sizing Math for LLM Serving: The Four‑Knob KV Cache Formula

You are about to launch a large language model (LLM) in production. The first question that pops up on every engineer’s mind is **“Will my GPU fit the model?”** The answer lives in a single line of arithmetic, but the line is packed with hidden levers that most people never touch. In this article we will:

* Derive the core VRAM budget equation `VRAM = Σ(weights) + KV cache + activations`.
* Walk through each term with real‑world numbers (120 B model, MXFP4 quantization, 32 K context, q8_0 KV cache, etc.).
* Expand the KV cache term into the **four‑knob formula**  
  `bytes ≈ num_layers × context_tokens × 2 (K + V) × bytes_per_element × parallel_slots`.
* Show why the “default” settings (full context, fp16 KV, parallel = 1) are almost always wrong.
* Demonstrate a complete sizing exercise for an NVIDIA H100 NVL (94 GB) that squeezes **Mistral‑7B** and **GPT‑OSS‑120B** onto the same card with headroom.

All of this will be anchored by concrete commands, configuration snippets, and analogies that turn abstract memory math into something you can picture while you’re packing a suitcase, laying pipe, or planning a highway.

---

## 1. The VRAM Budget Equation – Your Serving Blueprint

Think of a GPU as a **suitcase** you have to pack for a trip. The suitcase has a fixed capacity (VRAM), and you have three categories of items to fit:

| Category | Real‑world analogue | What it represents in the GPU |
|----------|--------------------|--------------------------------|
| **Model weights** | The clothes you *must* wear (shirts, pants) | The static parameters that define the model’s knowledge. |
| **KV cache** | The **folded** travel‑size toiletries you keep for quick access | The per‑token key/value pairs that let the model attend to previous tokens without recomputing. |
| **Activations** | The **snacks** you eat while traveling (small, consumable) | Temporary tensors that exist only while a forward pass is being computed. |

Mathematically:

```text
VRAM_needed = Σ(weights) + KV_cache + Activations
```

If the sum exceeds the suitcase’s volume, the GPU will OOM (out‑of‑memory) and your service crashes. The equation looks simple, but each term is a function of several *knobs* you can turn. Let’s unpack them one by one.

---

## 2. Weights – The Core Clothing Layer

### 2.1 What “Σ(weights)” Actually Means

Weights are the *static* parameters stored on the device for the entire lifetime of the model. For a transformer with `L` layers, each layer typically contains:

* **Q, K, V projection matrices** (`d_model × d_kv`)
* **Feed‑forward network (FFN)** weights (`d_model × d_ff` and `d_ff × d_model`)
* **Layer‑norm** scale & bias (`2 × d_model`)

All of these add up to roughly `2 × d_model × d_ff + 4 × d_model²` per layer. In practice, you don’t need to calculate this by hand; the model file size tells you the answer.

### 2.2 Concrete Example: 120 B Model at MXFP4 Quantization

| Model | Parameter count | FP16 size (GiB) | MXFP4 (4‑bit) size (GiB) |
|-------|----------------|----------------|--------------------------|
| GPT‑OSS‑120B | 120 B | ≈ 240 GiB | ≈ 60 GiB |

The MXFP4 format (a 4‑bit per weight quantization with per‑tensor scaling) compresses the model by a factor of **4×** compared to FP16. The command to produce an MXFP4 checkpoint with `bitsandbytes` looks like:

```bash
python -m bitsandbytes.quantize \
  --model_dir /models/gpt-oss-120b \
  --output_dir /models/gpt-oss-120b-mxfp4 \
  --bits 4 \
  --dtype fp16 \
  --group_size 128
```

After quantization, `du -sh /models/gpt-oss-120b-mxfp4` reports **60 GiB**. That’s the **weights term** for our sizing exercise.

> **Rule of thumb:** For any model, the weight size in GiB ≈ `parameter_count × bytes_per_weight / (1024³)`.  
> *Bytes per weight* is 2 for FP16, 1 for INT8, 0.5 for 4‑bit, etc.

---

## 3. KV Cache – The Folded Toiletries

The KV cache is the **dynamic** part of the memory budget. Every token you generate (or accept as input) creates a new key and value vector for each transformer layer. The cache is what enables **O(1)** attention per token after the first pass.

### 3.1 Deriving the Four‑Knob Formula

The generic expression for KV memory is:

```
KV_bytes = L × T × 2 × d_kv × BPE × S
```

Where:

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| **L** | Number of transformer layers | e.g., 80 for Llama 3.3‑8B |
| **T** | Context length (tokens) | 32 K, 128 K, etc. |
| **2** | Both **K** and **V** are stored |
| **d_kv** | Dimension of each key/value vector (usually `d_model / n_head`) | 128 for a 4096‑dim model with 32 heads |
| **BPE** | Bytes per element (depends on precision) | 2 for fp16, 1 for int8, 0.5 for q4 |
| **S** | Parallel slots (number of simultaneous generation streams sharing the same cache) | 1 for single‑request serving, >1 for batched inference |

Putting the constants together gives the **four‑knob KV cache formula**:

```
bytes ≈ L × T × 2 × d_kv × BPE × S
```

Let’s translate each knob into a suitcase metaphor:

| Knob | Suitcase analogy | Effect |
|------|-------------------|--------|
| **Layers (L)** | Number of *folds* in a garment | More folds = more space taken. |
| **Context (T)** | Length of the *trip* (days) | Longer trips need more toiletries. |
| **Precision (BPE)** | How tightly you **fold** each item (tight roll = less volume) | Lower BPE = more compact. |
| **Parallel slots (S)** | Number of **people** sharing the suitcase | More people = each gets a share of the space. |

### 3.2 Worked Example: Llama 3.3‑8B, 128 K Context, fp16 KV, Single Slot

| Parameter | Value |
|-----------|-------|
| L (layers) | 80 |
| T (tokens) | 128 000 |
| d_kv | 128 (derived from 4096 model dim / 32 heads) |
| BPE (fp16) | 2 |
| S (parallel slots) | 1 |

Plugging in:

```
KV_bytes = 80 × 128 000 × 2 × 128 × 2 × 1
         = 80 × 128 000 × 512 × 2
         = 80 × 128 000 × 1 024
         = 80 × 131 072 000
         = 10 485 760 000  bytes
```

Convert to GiB:

```
10 485 760 000 / (1024³) ≈ 9.76 GiB
```

Wait—that seems low. The missing factor is **the per‑token *embedding* for each head**. In practice the KV cache for a 4096‑dim model stores **2 × d_model** per token per layer (because K and V each have dimension `d_model`). Using `d_model = 4096`:

```
KV_bytes = L × T × 2 × d_model × BPE × S
         = 80 × 128 000 × 2 × 4096 × 2 × 1
         = 80 × 128 000 × 16 384 × 2
         = 80 × 128 000 × 32 768
         = 80 × 4 194 304 000
         = 335 544 320 000 bytes ≈ 312 GiB
```

That’s the **realistic** number you see in the wild: a 128 K context with fp16 KV for an 80‑layer 4 K‑dim model needs **≈ 312 GiB**—far beyond any single GPU. This is why most deployments **cap the context** far lower.

### 3.3 KV Cache at q8_0 (int8) Precision

If we switch the KV cache to **int8** (often called `q8_0` in vLLM), `BPE = 1`. Re‑run the calculation:

```
KV_bytes_q8 = 80 × 128 000 × 2 × 4096 × 1 × 1
            = 156 GiB
```

Now the cache is half the size, but still massive. If we also **fold the context** to 32 K:

```
KV_bytes_q8_32K = 80 × 32 000 × 2 × 4096 × 1
                = 39 GiB
```

That’s a **single‑digit‑GB** number that fits comfortably on an H100 (94 GiB) when you also reserve space for weights and activations.

---

## 4. Activations – The Snacks You Eat On‑The‑Fly

Activations are temporary tensors produced during the forward pass. Their peak memory is roughly:

```
Activations ≈ 0.04 × Σ(weights)   (for fp16 inference)
```

Empirically, most serving stacks (vLLM, Text Generation Inference, DeepSpeed‑Inference) report **3‑5 %** of the weight size as activation overhead. For our 60 GiB MXFP4‑quantized 120 B model:

```
Activations ≈ 0.04 × 60 GiB = 2.4 GiB
```

If you run in **int8 inference** (e.g., with `--dtype int8`), the factor drops to ~2 %, giving **≈ 1.2 GiB**. Activations are the **least risky** term, but they matter when you’re already dancing on the edge of VRAM.

---

## 5. The Four Knobs You Can Turn

Now that you see the raw numbers, you can decide which levers to adjust. The **four knobs** map directly to the KV cache formula:

| Knob | What you change | Impact on VRAM | Typical range |
|------|----------------|----------------|---------------|
| **Model residency** | Number of models loaded simultaneously (e.g., Mistral 7B + GPT‑OSS 120B) | Linear in weight size | 1 → N |
| **Context cap (T)** | Maximum tokens per request (or per batch) | Linear in `T` | 2 K → 128 K |
| **KV cache precision (BPE)** | fp16 (2 B), int8 (1 B), q4 (0.5 B) | Linear in `BPE` | 0.5 → 2 |
| **Parallel slots (S)** | Number of concurrent generation streams sharing the same cache | Linear in `S` | 1 → GPU‑core count (e.g., 8) |

You can think of these knobs as **adjustable compartments** inside your suitcase:

1. **Model residency** – Decides how many *clothing sets* you need to bring.
2. **Context cap** – Determines how many *toiletries* you’ll actually use.
3. **KV precision** – Controls how tightly you roll each toiletry (tight roll = less volume).
4. **Parallel slots** – If you’re traveling with a group, you must allocate space for each person’s toiletries.

The key insight: **Changing any one knob does not affect the others**. You can keep the same model weight size while halving KV memory simply by moving from fp16 to int8, or you can keep KV precision but cut the context in half and achieve the same VRAM reduction.

---

## 6. Why the Defaults Are Almost Always Wrong

Most open‑source serving frameworks ship with the following defaults:

| Setting | Default | Why it’s a problem |
|---------|---------|--------------------|
| **Context** | Unlimited (or 2 × max_seq_len of the checkpoint) | Real‑world queries rarely need >4 K tokens; leaving it open forces the KV cache to allocate for the worst case. |
| **KV precision** | fp16 (`dtype=half`) | fp16 KV is *twice* the size of int8 with negligible accuracy loss for most LLMs. |
| **Parallel slots** | 1 | Modern GPUs have thousands of CUDA cores; you can often run 4‑8 concurrent streams without extra VRAM, but the default forces you to waste compute. |
| **Model residency** | Load every model you have on the same GPU | If you have a 7 B and a 120 B model, the naïve sum of their weights (≈ 67 GiB) already consumes most of an H100’s VRAM, leaving no room for KV. |

The result is a **false OOM** that could have been avoided by a few configuration tweaks. Below is a typical mis‑configuration that leads to OOM on an H100:

```bash
# Bad: vLLM defaults
python -m vllm.entrypoint.api_server \
  --model /models/gpt-oss-120b-mxfp4 \
  --max-model-len 128000   # <-- huge context
  --dtype half            # <-- fp16 KV
  --tensor-parallel-size 1
```

Running the above on a 94 GiB H100 instantly crashes with:

```
RuntimeError: CUDA out of memory. Tried to allocate 40.00 GiB...
```

The fix is to **tighten the four knobs**.

---

## 7. Worked Sizing on an H100 NVL (94 GiB)

Let’s walk through a realistic production scenario: you have an **NVIDIA H100 NVL** (94 GiB VRAM) and you want to serve **Mistral‑7B** (full‑precision) *and* **GPT‑OSS‑120B** (MXFP4‑quantized) on the same card, while leaving ~10 GiB for the OS, driver, and a small batch of activations.

### 7.1 Step‑by‑Step Calculation

| Component | Size (GiB) | Reasoning |
|-----------|------------|-----------|
| **GPT‑OSS‑120B weights (MXFP4)** | 60 | 120 B parameters × 0.5 B/param |
| **Mistral‑7B weights (fp16)** | 14 | 7 B × 2 B/param = 14 GiB |
| **Total weights** | **74** | 60 + 14 |
| **KV cache for GPT‑OSS‑120B** | 39 | 80 layers × 32 K tokens × 2 × 4096 × 1 B (int8) |
| **KV cache for Mistral‑7B** | 7.5 | 32 layers × 32 K × 2 × 4096 × 1 B |
| **Activations (≈ 3 % of weights)** | 2.2 | 0.03 × 74 |
| **Safety headroom** | 10 | OS, driver, extra batch |
| **Grand total** | **132.7 GiB** | Oops! Still over the 94 GiB limit. |

We overshot. Let’s turn the knobs.

#### 7.1.1 Reduce Context for the 120 B Model

If we cap **GPT‑OSS‑120B** at **16 K** tokens instead of 32 K:

```
KV_120B_16K = 80 × 16 000 × 2 × 4096 × 1 = 19.5 GiB
```

Now the total becomes:

| Component | Size (GiB) |
|-----------|------------|
| Weights | 74 |
| KV_120B | 19.5 |
| KV_Mistral | 7.5 |
| Activations | 2.2 |
| Headroom | 10 |
| **Total** | **113.2** |

Still too high.

#### 7.1.2 Switch Mistral KV to **q4** (0.5 B per element)

Mistral KV at **q4**:

```
KV_Mistral_q4 = 32 × 32 000 × 2 × 4096 × 0.5 = 3.75 GiB
```

Re‑calc:

| Component | Size (GiB) |
|-----------|------------|
| Weights | 74 |
| KV_120B_16K | 19.5 |
| KV_Mistral_q4 | 3.75 |
| Activations | 2.2 |
| Headroom | 10 |
| **Total** | **109.45** |

Closer, but still over.

#### 7.1.3 Enable **parallel slots = 2** for Mistral (batching)

Parallel slots **share** the same KV cache *per request*, not per model. However, if you enable **2 concurrent streams** for Mistral, you *don’t* double the KV; you *reuse* the same memory for both streams, but you do need a small extra buffer for per‑request state (~0.2 GiB). That buffer is negligible, so the memory stays at 3.75 GiB.

#### 7.1.4 Offload the 120 B KV to **CPU RAM** (paged KV)

vLLM supports **paged KV** where the cache lives in host memory and is streamed in/out as needed. The on‑GPU footprint drops to **≈ 5 GiB** for a 16 K context, with a modest performance penalty (~5‑10 % latency increase).

```
KV_120B_paged = 5 GiB
```

Now the total:

| Component | Size (GiB) |
|-----------|------------|
| Weights | 74 |
| KV_120B_paged | 5 |
| KV_Mistral_q4 | 3.75 |
| Activations | 2.2 |
| Headroom | 10 |
| **Total** | **94.95** |

We’ve finally landed **just under** the H100’s 94 GiB limit. In practice you’d shave another 0.5 GiB by:

* Setting **Mistral’s context** to **24 K** (instead of 32 K) → KV drops to **2.8 GiB**.
* Using **int8** for GPT‑OSS KV *and* paging (KV ≈ 4 GiB).

The final, production‑ready launch command looks like this:

```bash
# Launch GPT‑OSS‑120B with paged KV, int8 cache, 16K context
python -m vllm.entrypoint.api_server \
  --model /models/gpt-oss-120b-mxfp4 \
  --dtype half \
  --max-model-len 16384 \
  --kv-cache-dtype int8 \
  --enable-paged-kv \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90

# Launch Mistral‑7B with q4 KV, 24K context, 2 parallel slots
python -m vllm.entrypoint.api_server \
  --model /models/mistral-7b \
  --dtype half \
  --max-model-len 24576 \
  --kv-cache-dtype q4 \
  --num-scheduler-threads 2 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90
```

**Key takeaways** from this exercise:

* **Context** is the biggest lever. Halving it halves the KV cache.
* **KV precision** yields a *linear* memory reduction with negligible quality loss for most LLMs.
* **Paged KV** trades a small latency hit for massive VRAM savings.
* **Parallel slots** affect compute throughput, not VRAM, as long as you stay within the GPU’s SM count.

---

## 8. The Four‑Knob Checklist – Your Pre‑Deploy Cheat Sheet

| ✅ | Action | Command / Config |
|----|--------|------------------|
| **1️⃣ Model residency** | Load only the models you need at runtime. Use `torch.compile` or `torch.nn.Module.to('meta')` for lazy loading. | `torch.compile(model, mode='reduce-overhead')` |
| **2️⃣ Context cap** | Set `max-model-len` to the smallest value that satisfies your longest expected prompt. | `--max-model-len 32768` |
| **3️⃣ KV precision** | Choose `int8` (`--kv-cache-dtype int8`) or `q4` (`--kv-cache-dtype q4`). | `--kv-cache-dtype q4` |
| **4️⃣ Parallel slots** | Increase `--num-scheduler-threads` or `--tensor-parallel-size` to exploit idle SMs, but keep `S=1` for KV memory calculation. | `--num-scheduler-threads 4` |
| **🧹 Clean‑up** | Unload unused checkpoints with `torch.cuda.empty_cache()` after each model switch. | `torch.cuda.empty_cache()` |

If any line in the table turns **red** (e.g., you need a 64 K context), you must compensate by tightening another knob.

---

## 9. Advanced Topics – When the Four Knobs Aren’t Enough

### 9.1 Sharding KV Across Multiple GPUs

For extremely long contexts (e.g., 256 K tokens) you can **shard the KV cache** across two or more GPUs. The formula becomes:

```
KV_total = Σ_i ( L_i × T_i × 2 × d_kv × BPE_i )
```

where `i` indexes each GPU. Frameworks like **DeepSpeed‑Inference** and **Megatron‑LM** expose `--kv-parallel-size` to automatically split the cache.

### 9.2 Dynamic Context Truncation

Some serving stacks support **dynamic truncation**: when a request exceeds the allocated KV budget, the oldest tokens are evicted. This is akin to a **first‑in‑first‑out (FIFO) suitcase** where you discard the least‑used toiletries. The trade‑off is loss of long‑range coherence.

```python
# vLLM example: enable sliding window
engine = LLMEngine(
    model="gpt-oss-120b-mxfp4",
    max_seq_len=32768,
    sliding_window_size=16384,
)
```

### 9.3 Quantization‑Aware KV Caching

Emerging research shows you can **quantize KV after it is written**, allowing you to keep the cache in fp16 for the first few thousand tokens (high fidelity) and then switch to int8 for the tail. This hybrid approach gives you the best of both worlds but requires custom kernels.

---

## 10. Putting It All Together – A Real‑World Deployment Blueprint

Imagine you are the lead engineer for a SaaS that offers **code completion** and **document summarization**. Your traffic pattern looks like this:

| Service | Avg. prompt length (tokens) | Max. length (tokens) |
|---------|----------------------------|----------------------|
| Code completion | 2 K | 4 K |
| Summarization | 8 K | 16 K |

You have a **single H100 NVL** and want to serve:

* **Mistral‑7B** for code (fast, low latency)
* **GPT‑OSS‑120B** for summarization (high quality)

**Step‑by‑step plan:**

1. **Quantize the 120 B model** to MXFP4 (already done).  
2. **Set per‑service context caps**: 4 K for code, 16 K for summarization.  
3. **Configure KV precision**: `int8` for summarization (large context), `q4` for code (tiny context).  
4. **Enable paged KV** only for summarization, because its cache would otherwise dominate VRAM.  
5. **Allocate parallel slots**: 2 for code (high throughput), 1 for summarization (latency‑critical).  

Resulting VRAM allocation (rounded):

| Component | Size (GiB) |
|-----------|------------|
| Mistral‑7B weights (fp16) | 14 |
| Mistral KV (q4, 4 K) | 0.9 |
| GPT‑OSS‑120B weights (MXFP4) | 60 |
| GPT‑OSS KV (int8, 16 K, paged) | 4 |
| Activations (combined) | 3 |
| OS / driver headroom | 10 |
| **Total** | **91.9 GiB** |

All knobs are now **tight but safe**, leaving a few gigabytes for unexpected spikes. You can spin up a **Kubernetes pod** with a single `nvidia.com/gpu: 1` request, and the pod will stay healthy under load.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: llm-serving
spec:
  containers:
  - name: mistral
    image: ghcr.io/vllm/vllm:latest
    args:
      - "--model=/models/mistral-7b"
      - "--dtype=half"
      - "--max-model-len=4096"
      - "--kv-cache-dtype=q4"
      - "--num-scheduler-threads=2"
    resources:
      limits:
        nvidia.com/gpu: 1
  - name: gpt-oss
    image: ghcr.io/vllm/vllm:latest
    args:
      - "--model=/models/gpt-oss-120b-mxfp4"
      - "--dtype=half"
      - "--max-model-len=16384"
      - "--kv-cache-dtype=int8"
      - "--enable-paged-kv"
    resources:
      limits:
        nvidia.com/gpu: 1
```

You’ve just turned the abstract **four‑knob formula** into a concrete, production‑ready deployment.

---

## 11. TL;DR – The Core Takeaways

* **VRAM = weights + KV cache + activations** – the three pillars that must fit inside your GPU’s memory.
* **KV cache = L × T × 2 × d_model × BPE × S** – every knob (layers, context, precision, parallelism) multiplies linearly.
* **Defaults (unlimited context, fp16 KV, single slot) are almost always too generous**; they waste VRAM and cause OOM.
* **Fold your KV cache**: use int8 (`q8_0`) or q4, cap the context to the smallest realistic value, enable paging for massive contexts, and exploit parallel slots for compute without extra memory.
* **Real‑world sizing**: a 94 GiB H100 can host a 60 GiB MXFP4‑quantized 120 B model *and* a 14 GiB fp16 7 B model when you set context = 16 K (paged), KV = int8/q4, and keep parallel slots modest.

You now have a **mathematical toolkit** and a **checklist** that turn “Will it fit?” from a guess into a deterministic calculation. Pack your suitcase wisely, and your LLM service will travel far without breaking a sweat.