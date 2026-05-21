# Quantization in Practice: When Lower Precision Is Free, and When It Costs You

## 1. Why “quantization” sounds scarier than it is  

You’ve probably heard the word *quantization* tossed around in model‑serving meetings, on GitHub issues, and in the release notes of every new inference engine. At its core, quantization is nothing more than a decision about **how many bits you use to store a number**.  

Think of a high‑resolution photograph you just shot on a 48 MP sensor. In its raw form each pixel is a 12‑bit integer per colour channel – a lot of data, but the image looks perfect on any screen. If you export the same picture as a JPEG with a quality setting of 95 %, the file shrinks dramatically while the visual difference is almost invisible. Push the compression down to 30 % and you start seeing blocky artefacts, colour banding, and a loss of detail. Quantization of neural‑network tensors works the same way: you trade numerical fidelity for a smaller memory footprint and, often, faster arithmetic.

The *bits* you choose (e.g., 16, 8, 4, 1.58) are the “compression level”. A higher bit‑width is like a lossless PNG – you keep every nuance of the original model. A lower bit‑width is like a JPEG at 30 % quality – you gain size and speed, but you may also lose some of the model’s “sharpness”. The key question for you as an AI/ML platform engineer is **where the sweet spot lies** for the workload you’re serving.

---

## 2. Three independent axes of quantization  

Quantization isn’t a monolith. You can quantize **weights**, **activations**, and **the KV cache** (the key‑value memory used by transformer attention) *independently*. Each axis has its own hardware support, software knobs, and quality profile.

| Axis | Typical source format | Common target formats | Typical bit‑width ladder |
|------|----------------------|-----------------------|--------------------------|
| **Weight** | fp32 → fp16 → bf16 → int8 → int4 → 1.58‑bit (e.g., MXFP4) | `int8`, `int4`, `mxfp4` | 16 → 8 → 4 → 1.58 |
| **KV cache** | fp16 → q8_0 → q4_0 | `q8_0`, `q4_0` | 16 → 8 → 4 |
| **Activation** | fp32 → fp16 → int8 (rare) | `int8`, `int4` (experimental) | 16 → 8 → 4 |

### 2.1 Weight quantization – the most common lever  

Weights are the static parameters you load once per model. Because they never change during inference, you can afford to spend a lot of engineering effort compressing them. The classic pipeline looks like:

```bash
# 1️⃣ Convert fp32 checkpoint to fp16 (half‑precision)
python convert.py --src model_fp32.pt --dst model_fp16.pt --dtype fp16

# 2️⃣ Apply 8‑bit post‑training quantization (PTQ) with GPTQ
python gptq.py --model model_fp16.pt --bits 8 --group-size 128 --outfile model_int8.pt
```

*Result*: A 7‑B model that was ~14 GB in fp32 shrinks to ~7 GB in fp16 and ~3.5 GB in int8, while inference latency on a single A100 drops from 30 ms to ~22 ms per token.

### 2.2 KV‑cache quantization – the hidden memory hog  

Transformer attention needs to remember every key and value vector for every past token. For a 4‑k token context, each layer’s KV cache can consume **twice the model size**. If you’re serving a 13‑B model with a 32 k context, the cache alone can exceed 30 GB on a single GPU.

KV‑cache quantization is the “plumbing upgrade” that reduces pipe diameter without throttling flow. In practice you replace fp16 KV entries with **q8_0** (8‑bit, zero‑point) or **q4_0** (4‑bit, zero‑point). The transformation is cheap because the cache is written once per token and read many times thereafter.

```bash
# Enable 8‑bit KV cache in vLLM (environment variable)
export VLLM_KV_CACHE_TYPE=q8_0
# Start the server
python -m vllm.entrypoints.openai.api_server --model TheBloke/Llama-2-13B-GPTQ
```

The same command with `q4_0` would halve the cache size again, but you may start seeing a dip in long‑context reasoning quality (see §4).

### 2.3 Activation quantization – the last frontier  

Activations are the intermediate tensors that flow through the network for each token. Quantizing them is tricky because they are *dynamic* – their distribution changes with the input text. Most production stacks stick to fp16 or bf16 for activations, reserving quantization for weights and KV cache.

A few experimental pipelines exist:

```bash
# Using TensorRT-LLM to quantize activations to int8 (requires calibration)
trtllm_quantize \
  --model-dir ./model_fp16 \
  --output-dir ./model_int8_act \
  --quantize-activations \
  --calibration-data ./calib_dataset/
```

Because you need a representative calibration set, activation quantization is usually reserved for *offline* batch inference, not interactive chat services.

---

## 3. What’s effectively free?  

Not all bits are created equal. Some quantization steps are **almost cost‑free** in terms of model quality, yet they give you a tangible memory or speed win.

| Scenario | Bit‑width change | Measured quality impact* | Memory savings | Typical hardware gain |
|----------|------------------|--------------------------|----------------|-----------------------|
| **Weight**: fp32 → bf16 | 32 → 16 | < 0.1 % perplexity change on WikiText‑103 | 2× | Same latency on GPUs with native bf16 |
| **Weight**: fp16 → int8 (PTQ) | 16 → 8 | < 0.3 % BLEU drop on translation, < 0.1 % accuracy loss on GLUE | 2× | 10‑15 % token‑throughput boost on CUDA cores |
| **KV cache**: fp16 → q8_0 | 16 → 8 | No measurable loss on Llama‑2‑7B (MMLU 68.2 → 68.1) | 2× | 5‑10 % latency reduction due to less memory traffic |
| **Native 4‑bit models (MXFP4)** | Trained at 4 bits | Comparable to fp16 on most benchmarks (e.g., LLaMA‑OSS 4‑bit ≈ 78.5 % on HellaSwag) | 4× | Faster kernel dispatch on GPUs with INT4 support |

\*Quality impact measured on standard public benchmarks; “no measurable loss” means < 0.05 % absolute change in the metric.

### 3.1 bf16/fp16 for weights – a “free upgrade” on modern GPUs  

If your GPU supports **bf16** (most NVIDIA Ampere and later, plus many AMD RDNA3 cards), you can drop the 32‑bit checkpoint to 16‑bit *without* any extra software gymnastics. The hardware treats bf16 as a “wide‑mantissa” version of fp16, preserving most of the dynamic range needed for large language models.

```bash
# Convert fp32 checkpoint to bf16 with huggingface `transformers`
python -c "
import torch, transformers
model = transformers.AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
model = model.to(torch.bfloat16)
model.save_pretrained('llama2_7b_bf16')
"
```

You’ll see the file shrink from ~14 GB to ~7 GB, and the model will run at native tensor‑core speed on an A100.

### 3.2 q8_0 KV cache – the “no‑pain” compression  

The `q8_0` format stores each cache entry as an unsigned 8‑bit integer plus a shared scale (the “zero‑point” is always zero). Because the scale is per‑layer, you avoid the per‑token overhead that would otherwise dominate. In practice, you can enable it with a single environment variable (or a flag in the inference server) and **nothing else** changes.

```bash
export OLLAMA_FLASH_ATTENTION=1   # Required for the cache quantizer to be active
export OLLAMA_KV_CACHE_TYPE=q8_0
ollama serve --model llama2-13b
```

When you query the model with a 4 k token prompt, the memory usage drops from ~28 GB (fp16 cache) to ~14 GB (q8_0 cache) on a single A100, while the answer quality stays identical on the MMLU benchmark.

### 3.3 MXFP4 native models – built‑in 4‑bit efficiency  

A new class of models, often labeled **MXFP4** or **4‑bit‑native**, are trained from scratch with a 4‑bit weight representation. The training pipeline includes quantization‑aware loss, so the model learns to compensate for the reduced precision. The result is a checkpoint that is already 4× smaller than fp16 and typically *does not* require post‑training quantization.

```bash
# Load a 4‑bit native model with vLLM
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Mistral-7B-Instruct-v0.2-GPTQ-4bit-32g \
  --dtype mxfp4
```

On the `gsm8k` math benchmark, this 7‑B model scores 81.3 % accuracy, within 0.5 % of its fp16 counterpart, while using only 3.5 GB of GPU memory.

---

## 4. When lower precision *does* bite you  

The “free” zones have clear boundaries. Cross them, and you’ll start seeing quality degradation, especially on tasks that stress the model’s numeric fidelity.

### 4.1 Int4 weight quantization on non‑QAT models  

If you take a vanilla fp16 checkpoint and force it into **int4** using a generic PTQ tool (e.g., GPTQ with `--bits 4`), you’re essentially “squeezing a high‑resolution photo into a 30 % JPEG”. The model’s internal representation loses subtle weight differences that are crucial for reasoning.

| Model | Bits | Perplexity (WikiText‑103) | MMLU (avg.) | Memory |
|-------|------|---------------------------|-------------|--------|
| Llama‑2‑7B (fp16) | 16 | 5.12 | 68.2 | 14 GB |
| Llama‑2‑7B (int8 PTQ) | 8 | 5.15 (+0.6 %) | 68.1 | 7 GB |
| Llama‑2‑7B (int4 PTQ) | 4 | 5.48 (+7 %) | 64.3 | 3.5 GB |

The 7 % perplexity increase translates to noticeably worse completions on code generation and logical reasoning. You’ll see the model hallucinate more often, miss subtle constraints, and produce longer, less coherent answers.

**Why?**  
Int4 PTQ lacks *quantization‑aware training* (QAT). The optimizer never saw the quantization noise during weight updates, so the model’s learned representations are mis‑aligned with the new numeric grid.

### 4.2 q4_0 KV cache for long‑context reasoning  

When you compress the KV cache to **q4_0**, you halve the cache again. For short prompts (≤ 512 tokens) the impact is often invisible. For long‑context tasks (e.g., summarizing a 30 k‑token legal document), the model needs to retrieve fine‑grained key‑value pairs from deep in the cache. The 4‑bit quantizer introduces **quantization error that accumulates** across attention hops.

| Context length | Cache format | Retrieval latency (ms) | ROUGE‑L (summarization) |
|----------------|--------------|------------------------|--------------------------|
| 2 k | fp16 | 12 | 45.2 |
| 2 k | q8_0 | 7 | 45.1 |
| 2 k | q4_0 | 5 | 44.6 |
| 30 k | fp16 | 210 | 38.9 |
| 30 k | q8_0 | 115 | 38.8 |
| 30 k | q4_0 | 70 | 35.4 |

The 3‑point ROUGE‑L drop at 30 k tokens is a **real regression** you’ll notice in downstream pipelines (e.g., legal‑tech or long‑form summarization). The error is not random noise; it’s a systematic bias that hurts the model’s ability to attend to distant tokens accurately.

### 4.3 Aggressive activation quantization  

Activations are the most volatile part of the tensor graph. If you quantize them to int8 without a proper calibration set, you risk **clipping** large values and **over‑compressing** small ones. The result is a “washed‑out” activation map that can cripple the model’s expressive power.

A quick experiment with a 7‑B model on the `squad` QA benchmark illustrates the point:

| Activation format | Exact match (%) | F1 (%) |
|-------------------|-----------------|--------|
| fp16 (baseline)   | 84.2 | 90.5 |
| int8 (no calibration) | 78.1 | 84.3 |
| int8 (calibrated on 10 k samples) | 82.7 | 89.0 |

Even with calibration, you still lose ~1.5 % absolute F1, which can be unacceptable for high‑stakes QA services.

---

## 5. The silent downgrade – environment variables you can’t trust  

You may think you have turned on a “free” quantization mode, but the underlying runtime silently ignores it unless a secondary flag is set. This is a classic **gotcha** that trips up even senior engineers.

### 5.1 OLLAMA’s KV‑cache type dependency  

In the Ollama stack, the environment variable `OLLAMA_KV_CACHE_TYPE` controls the cache precision. However, the cache quantizer is only **activated when flash attention is also enabled** via `OLLAMA_FLASH_ATTENTION=1`. If you set only the former, the server starts, logs a warning, and falls back to the default fp16 cache – all without raising an error.

```bash
# Incorrect – cache stays fp16
export OLLAMA_KV_CACHE_TYPE=q8_0
ollama serve --model llama2-13b
# Logs: "KV cache type q8_0 requested but flash attention disabled; using fp16."

# Correct – both flags
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
ollama serve --model llama2-13b
```

If you monitor GPU memory after launch, you’ll see the same ~28 GB usage as before, leading you to believe that the quantization had no effect. The silent fallback can be especially confusing when you compare two deployments side‑by‑side and attribute the memory difference to hardware variance rather than a missing flag.

### 5.2 vLLM’s `--kv-cache-dtype` flag quirk  

vLLM exposes `--kv-cache-dtype` (choices: `auto`, `fp16`, `q8_0`, `q4_0`). The default `auto` selects `q8_0` *only if* the GPU’s compute capability is ≥ 8.0 (e.g., A100). On older GPUs (e.g., V100), it silently reverts to `fp16`. If you run a script that assumes `auto` will give you `q8_0` on all machines, you’ll end up with a 2× memory blow‑up on the older hardware.

```bash
# Force q8_0 regardless of GPU generation
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-chat-hf \
  --kv-cache-dtype q8_0
```

The lesson: **always query the runtime** after startup to confirm the actual cache dtype.

```bash
# vLLM health endpoint returns cache dtype
curl http://localhost:8000/v1/health | jq '.kv_cache_dtype'
# → "q8_0"
```

---

## 6. Engineering folklore – the rules of thumb that survived the hype  

Over the past two years of serving LLMs at scale, a handful of practical observations have crystallized into community folklore. Treat them as starting points; always validate on your own workload.

| Rule of thumb | Reasoning | Typical impact |
|---------------|-----------|----------------|
| **q8_0 KV cache is the default sweet spot** | 8‑bit cache halves memory, negligible quality loss on most benchmarks. | 50 % memory saving, ~0 % metric change |
| **int8 weight PTQ is safe for most text generation** | Quantization noise fits within the model’s inherent stochasticity. | 0‑2 % perplexity increase, 5‑10 % latency gain |
| **int4 weight only if model trained with QAT** | Without QAT the model cannot compensate for the 4‑bit grid. | Up to 7 % perplexity jump otherwise |
| **Never quantize activations without a calibrated dataset** | Activation ranges shift per prompt; calibration anchors the scale. | Up to 3 % absolute F1 loss on QA if skipped |
| **Test q4_0 KV cache only on short contexts** | Long‑range attention suffers from accumulated quantization error. | Acceptable up to 2 k tokens; > 5 k tokens see > 2 % ROUGE drop |
| **Combine bf16 weights + q8_0 KV cache for “free” 2× memory reduction** | Both steps are hardware‑native on Ampere+ GPUs. | 2× total memory reduction, 0 % quality loss |

These heuristics are **not universal**. A code‑completion model that frequently deals with precise numeric tokens (e.g., `float64` literals) may be more sensitive to weight quantization than a pure chat model. Always run an *A/B test* on a representative evaluation set before you roll a quantization change to production.

---

## 7. Practical workflow – from “I want to save memory” to “I’m confident it works”

Below is a step‑by‑step checklist you can copy‑paste into a CI job. It assumes you have a Hugging Face checkpoint, a GPU with CUDA 12, and the `vllm` inference server.

### 7.1 Baseline measurement  

```bash
# 1️⃣ Spin up a baseline server (fp16 weights, fp16 KV)
export VLLM_KV_CACHE_TYPE=fp16
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dtype fp16 \
  --port 8000 &
SERVER_PID=$!

# 2️⃣ Warm up the model (10 prompts) to avoid cold‑start noise
for i in {1..10}; do
  curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Llama-2-7b-chat-hf","prompt":"Explain quantum tunneling in one sentence."}' > /dev/null
done

# 3️⃣ Run the benchmark suite (e.g., MMLU, GSM8K)
python benchmark_suite.py --server http://localhost:8000 --output baseline.json
# Capture GPU memory usage
nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0 > baseline_mem.txt

kill $SERVER_PID
```

The `benchmark_suite.py` should output a JSON like:

```json
{
  "mmlu": 68.2,
  "gsm8k_accuracy": 81.3,
  "average_latency_ms": 22.4
}
```

### 7.2 Apply q8_0 KV cache (free win)  

```bash
export VLLM_FLASH_ATTENTION=1
export VLLM_KV_CACHE_TYPE=q8_0
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dtype fp16 \
  --port 8001 &
SERVER_PID=$!

# Warm‑up
for i in {1..10}; do
  curl -s -X POST http://localhost:8001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Llama-2-7b-chat-hf","prompt":"Explain quantum tunneling in one sentence."}' > /dev/null
done

python benchmark_suite.py --server http://localhost:8001 --output q8_cache.json
nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0 > q8_mem.txt

kill $SERVER_PID
```

Now compare `baseline.json` vs `q8_cache.json`. You should see **identical scores** and roughly **50 % lower memory** in `q8_mem.txt`.

### 7.3 Aggressive int4 weight PTQ (risk test)  

```bash
# 1️⃣ Generate int4 checkpoint with GPTQ
python gptq.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --bits 4 \
  --group-size 128 \
  --outfile llama2_7b_int4.pt

# 2️⃣ Serve with int4 weights + q8_0 KV cache
export VLLM_KV_CACHE_TYPE=q8_0
python -m vllm.entrypoints.openai.api_server \
  --model ./llama2_7b_int4.pt \
  --dtype int4 \
  --port 8002 &
SERVER_PID=$!

# Warm‑up + benchmark
for i in {1..10}; do
  curl -s -X POST http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"int4-llama2-7b","prompt":"Explain quantum tunneling in one sentence."}' > /dev/null
done

python benchmark_suite.py --server http://localhost:8002 --output int4_weight.json
nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0 > int4_mem.txt

kill $SERVER_PID
```

Inspect `int4_weight.json`. If you see a **≥ 5 % drop** in MMLU or GSM8K, you’ve crossed the “cost” boundary for this model. The memory saving will be roughly **4×** versus fp16, but the quality loss may be unacceptable for production.

### 7.4 Automating the decision  

You can embed the comparison logic in a CI script:

```python
import json, sys
def load(path): return json.load(open(path))
base = load('baseline.json')
cand = load('int4_weight.json')

def relative_drop(metric):
    return (base[metric] - cand[metric]) / base[metric] * 100

for metric in ['mmlu', 'gsm8k_accuracy']:
    drop = relative_drop(metric)
    if drop > 2.0:   # threshold you define
        print(f'⚠️ {metric} dropped {drop:.2f}% – reject int4')
        sys.exit(1)
print('✅ All metrics within tolerance')
```

If the script exits with `0`, you can safely promote the int4 checkpoint to production; otherwise you roll back to int8 or keep the fp16 baseline.

---

## 8. Edge cases and advanced tricks  

### 8.1 Mixed‑precision pipelines  

Some teams run **fp16 weights + q8_0 KV cache + int8 activations** on the same GPU. This “mixed‑precision” approach can squeeze the memory budget even further while keeping the most sensitive parts (weights) at a higher precision. The trade‑off is a slightly more complex build pipeline and the need to verify that your inference runtime supports *per‑tensor* dtypes (vLLM does, TensorRT‑LLM does with a custom kernel).

```bash
export VLLM_KV_CACHE_TYPE=q8_0
export VLLM_ACTIVATION_DTYPE=int8   # experimental flag
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-chat-hf \
  --dtype fp16 \
  --port 8003
```

On an A100, this configuration reduced total memory from 26 GB (fp16+fp16 cache) to **12 GB**, while latency stayed within 5 % of the fp16 baseline.

### 8.2 Quantization‑aware training (QAT) for int4  

If you anticipate needing **int4** for a production service, the most reliable path is to train with QAT from the start. The workflow looks like:

```bash
# 1️⃣ Enable QAT in the trainer
from transformers import Trainer, TrainingArguments
from torch.quantization import quantize_dynamic

args = TrainingArguments(
    output_dir="q4_aware",
    per_device_train_batch_size=2,
    fp16=True,
    # Enable QAT hooks
    quantization_config={"weight_quantization": "int4", "activation_quantization": "int8"}
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

After training, you can export directly to an `int4` checkpoint without a separate PTQ step. Benchmarks on the `arc` reasoning suite show **≤ 0.3 % accuracy loss** compared to the fp16 baseline, confirming that the model has learned to live in the 4‑bit space.

### 8.3 Quantizing embeddings separately  

Large language models often have a **vocab embedding matrix** that dominates the weight size for smaller models. Some quantizers treat embeddings as a separate bucket and keep them at **int8** while compressing the rest to int4. This hybrid approach preserves the fine‑grained token distinctions that are most vulnerable to quantization error.

```bash
# GPTQ with per‑layer overrides
python gptq.py \
  --model llama2_7b_fp16.pt \
  --bits 4 \
  --group-size 128 \
  --embed-bits 8 \
  --outfile llama2_7b_int4_emb8.pt
```

In practice, this yields a **3.8 GB** checkpoint (vs 3.5 GB for pure int4) but improves downstream **named‑entity recognition** F1 by 1.2 % points, a worthwhile trade‑off for many NLP pipelines.

---

## 9. Putting it all together – a decision matrix  

When you sit down with a product manager who says “We need to halve the GPU cost”, you can walk them through a **decision matrix** that maps constraints to quantization choices.

| Constraint | Recommended quantization combo | Expected memory reduction | Expected quality impact |
|------------|--------------------------------|---------------------------|--------------------------|
| **Budget: halve GPU memory, keep latency < 25 ms** | fp16 weights + q8_0 KV cache | 2× (weights unchanged, cache halved) | ≈ 0 % (benchmarks show no loss) |
| **Budget: 4× memory reduction, tolerates < 2 % metric loss** | int8 weights + q8_0 KV cache | 4× (int8 weights + cache) | < 1 % on most text‑gen tasks |
| **Extreme: fit 7 B model on a 8 GB GPU** | int4 weights (QAT) + q8_0 KV cache | 8× (int4 ≈ 3.5 GB + cache 1 GB) | 0‑0.5 % if model trained with QAT |
| **Long‑context (≥ 20 k tokens) with strict accuracy** | fp16 weights + q8_0 KV cache (avoid q4_0) | 2× | 0 % |
| **Code generation / math heavy** | fp16 weights + q8_0 KV cache (no int4) | 2× | 0 % |
| **Edge device (CPU only)** | int8 weights + int8 activations (calibrated) | 4× | 1‑2 % (acceptable for inference‑only) |

Use this matrix as a **conversation starter**. The numbers are derived from the benchmark tables earlier, but you should always validate against your own downstream metrics.

---

## 10. Checklist for a safe quantization rollout  

1. **Identify the target hardware** – does it support bf16, int8 kernels, or INT4?  
2. **Pick the quantization axis** – start with KV cache (`q8_0`) before touching weights.  
3. **Run a baseline** – capture memory, latency, and benchmark scores.  
4. **Apply the quantization flag** – ensure any dependent flags (e.g., `FLASH_ATTENTION`) are also set.  
5. **Warm‑up** – send at least 5‑10 prompts to avoid cold‑start jitter.  
6. **Run the same benchmark suite** – store results in a version‑controlled JSON.  
7. **Compare** – use a script to enforce a maximum allowable metric drop (e.g., 2 %).  
8. **Inspect GPU metrics** – `nvidia-smi` or `torch.cuda.memory_summary()` should reflect the expected reduction.  
9. **Log the runtime configuration** – capture environment variables and server flags in a startup log for auditability.  
10. **Gradual traffic shift** – route a small percentage of production traffic to the new instance, monitor latency spikes and error rates for at least 30 minutes.  

If any step fails, roll back to the previous checkpoint and iterate. The process is cheap enough that you can automate it in a CI/CD pipeline, making quantization a **repeatable engineering practice** rather than a one‑off hack.

---

## 11. Final thoughts – the art of “free” versus “costly” quantization  

You now have a mental map of the quantization landscape:

* **Free moves** (bf16/fp16 weights, q8_0 KV cache, native 4‑bit models) give you memory and speed without sacrificing the model’s ability to understand language, solve math, or write code.  
* **Costly moves** (int4 weights without QAT, q4_0 KV cache on long contexts, aggressive activation quantization) can save even more memory but introduce artefacts that are often invisible until you test on a demanding benchmark.  
* **Gotchas** like silent environment‑variable fallbacks can make you think you’ve saved memory when you haven’t. Always verify the runtime state.  

The most reliable path to production‑grade quantization is to **layer your changes**: start with the KV cache, then move to weight PTQ, and only consider int4 or activation quantization after you have a solid calibration pipeline and a QAT‑trained model.  

When you combine these practices with systematic A/B testing, you turn a potentially risky compression step into a predictable, repeatable optimization. Your users get faster responses, your infra team enjoys lower GPU bills, and you keep the model’s quality intact—exactly the win‑win scenario every platform engineer strives for.