# Cloud-Routing vs Self-Hosting: A Framework for the Build-vs-Buy Decision  

You’ve probably heard the question framed as “Should we run our LLMs in the cloud or on‑premises?” The moment you answer “cloud OR local” you’ve already narrowed the solution space to a false binary. In practice the optimal architecture is a **mix** that routes each request to the environment that best matches its characteristics—just like you own a sedan for daily commuting and rent a van for a weekend road‑trip.  

In this article we’ll start from the ground up: what “cloud” and “self‑hosted” really mean, why the four‑axis decision matrix matters, how to implement a clean hybrid router, and which economic traps to avoid. By the end you’ll have a reusable framework you can apply to any LLM workload, whether you’re building a chatbot for internal help‑desk tickets or a high‑frequency trading signal generator that needs sub‑100 ms latency.

---

## 1. The False Binary – Why “Cloud OR Local” Is a Myth  

### 1.1 The Car‑Rental Analogy  

Imagine you live in a city where a compact car gets you to work, the grocery store, and the gym. You love the convenience of owning it: you never have to think about insurance, registration, or parking permits. Yet when you plan a family vacation to the mountains, you suddenly need a larger vehicle, better fuel economy, and perhaps all‑wheel drive. You could buy a second car, but that would mean paying for a vehicle you’ll only use a few weekends a year. Instead you **rent** a van for the trip, keep the sedan for daily life, and enjoy the best of both worlds.

LLM workloads behave the same way. A small, steady‑state inference job (e.g., a “what‑can‑I‑do‑today” assistant) maps nicely to a cloud API that charges per token. A bursty, high‑throughput use case (e.g., batch‑processing millions of documents overnight) may be cheaper and faster on a self‑hosted GPU farm you already own. The decision isn’t a toggle; it’s a **routing policy** that decides, per request, which “vehicle” to dispatch.

### 1.2 The Plumbing Analogy  

Think of your data pipeline as a building’s plumbing. The main water line (the cloud) brings in a huge volume of water at a fixed pressure. For most sinks and toilets you just tap into that line—no extra work. But a high‑pressure industrial washer (a latency‑critical inference service) may need a dedicated pump and thicker pipes to avoid pressure drops. You don’t replace the whole municipal supply; you add a local loop that branches off only where needed.

### 1.3 The Kitchen Analogy  

A restaurant kitchen has a **sauce station** that prepares a handful of sauces in bulk each night (cloud batch jobs). When a guest orders a custom dish, the chef may pull a pre‑made sauce from the fridge (cloud API) or fire up the stovetop for a fresh reduction (local GPU) if the guest wants “extra‑fresh”. The menu isn’t limited to “all‑sauce” or “all‑stovetop”; it’s a hybrid that maximizes flavor while minimizing waste.

These analogies set the stage for the **four‑axis decision matrix** that tells you when to use which “vehicle”.  

---

## 2. The Four‑Axis Decision Matrix  

| Axis | What it measures | Cloud‑friendly signal | Self‑hosted signal | Typical break‑even point |
|------|------------------|-----------------------|--------------------|--------------------------|
| **Data Sensitivity** | How much raw input can leave your premises | Low‑sensitivity (public web queries, anonymized logs) | High‑sensitivity (PII, trade secrets) | If > 30 % of requests contain regulated data, self‑host becomes mandatory |
| **Traffic Pattern** | Temporal shape of request volume | Steady, predictable load (≤ 5 k tpm) | Bursty or seasonal spikes (≥ 20 k tpm for > 4 h) | 70 % GPU utilization over a month → self‑host cheaper |
| **Model Variety** | Number of distinct model families & version combos | Many low‑traffic models (≥ 10 models, < 500 tpm each) | Few high‑traffic models (≤ 3 models, > 10 k tpm each) | TCO flips when per‑model API cost > $0.02 per 1 k tokens |
| **Latency Tolerance** | Maximum acceptable round‑trip time | 100‑500 ms (chat, summarization) | < 100 ms (real‑time recommendation, trading) | Sub‑100 ms → local GPU required |

Let’s unpack each axis with concrete numbers and a short decision flow.

### 2.1 Data Sensitivity  

Cloud LLM APIs (OpenAI, Anthropic, Cohere) ingest the **entire request payload** for tokenization, routing, and policy enforcement. If your request contains personally identifiable information (PII), protected health information (PHI), or intellectual property (IP) that falls under GDPR, HIPAA, or internal compliance, you must **guarantee that the data never leaves your trusted boundary**.

**Example:**  
You run a legal‑document summarizer that receives excerpts of contracts. A single request may contain 2 KB of text, 30 % of which is client‑specific clauses. If you process 100 k requests per month, that’s 200 MB of sensitive data. Even at a modest $0.001 per token (≈ $0.02 per 1 k tokens), the **privacy risk** outweighs the $2 k monthly API bill.

**Self‑hosted rule of thumb:** If > 30 % of your payloads are classified as “restricted”, you should route those to a local inference service. The remaining “public” traffic can still go to the cloud for cost efficiency.

### 2.2 Traffic Pattern  

Cloud providers excel at **elastic scaling**: you spin up more request capacity instantly and pay per token. Self‑hosting shines when you can keep your GPUs **busy** for long stretches, amortizing the capital expense.

**Burst example:**  
A nightly batch job processes 5 M documents (≈ 50 TB of raw text). The job runs for 6 h, peaking at 30 k tpm (tokens per minute). A single A100 GPU can handle ~ 30 k tpm for a single model. If you have 4 GPUs idle the rest of the day, you can finish the batch in 6 h with 25 % utilization, costing you only electricity and depreciation. The same workload on a cloud API would be billed at ~ $0.02 per 1 k tokens → ≈ $2 M per month.

**Steady example:**  
A customer‑support chatbot receives 2 k tpm uniformly throughout the day. A single GPU would be under‑utilized (≈ 5 % load). Cloud API cost: 2 k tpm × 60 min × 30 days ≈ 86 M tokens → $1.7 k/month. Running a GPU 24/7 for that load would waste electricity (≈ $300/month) and incur depreciation (≈ $1 k/month). Cloud wins.

### 2.3 Model Variety  

If your product ships **ten different fine‑tuned models** (e.g., domain‑specific assistants for finance, health, legal, etc.) each receiving < 500 tpm, the per‑model API cost adds up, but you avoid the operational overhead of maintaining ten GPU‑dedicated endpoints.

**TCO calculation:**  

| Scenario | # Models | Avg tpm | Cloud cost (per 1 k tokens) | Monthly API spend | GPU count needed | GPU depreciation* | Monthly GPU cost |
|----------|----------|--------|-----------------------------|-------------------|------------------|-------------------|------------------|
| Cloud only | 10 | 400 | $0.02 | $2 400 | – | – | – |
| Self‑hosted | 10 | 400 | – | – | 2 (assuming 30 k tpm per GPU) | $2 000 | $2 300 |

\*Depreciation: $2 000 per GPU (4‑year straight‑line on $8 k GPU).  

If the **average tpm per model** climbs above ~ 5 k, the GPU count explodes and self‑hosting becomes cheaper. The break‑even point is roughly **$0.02 per 1 k tokens** vs **$0.015 per GPU‑hour** (including power and amortization).  

### 2.4 Latency Tolerance  

Latency budgets are often expressed as **service‑level objectives (SLOs)**. For a conversational UI, 200 ms of perceived latency feels instantaneous; 500 ms is acceptable. For a high‑frequency trading (HFT) signal, you need < 50 ms from market data receipt to model output.

**Cloud latency numbers (2024‑Q2 data):**  

- Intra‑region API call (OpenAI `gpt‑4o`) → 120 ms median, 300 ms 99th percentile.  
- Cross‑region call (US‑East → EU‑West) → 250 ms median, 600 ms 99th percentile.

**Local vLLM latency (A100, 80 GB):**  

- Single‑prompt (≤ 256 tokens) → 30 ms median, 70 ms 99th percentile.  
- Batch of 32 prompts → 55 ms median, 120 ms 99th percentile.

If your SLO is **< 100 ms**, you must route to a local inference service. If you can tolerate up to **500 ms**, the cloud is acceptable and you gain the elasticity advantage.

---

## 3. Building a Hybrid Router – The “Model‑Suffix” Pattern  

The simplest way to make routing transparent to your downstream services is to **encode the execution target in the model identifier**. The pattern looks like this:

```
gpt-oss:120b               → cloud API (OpenAI, Anthropic, etc.)
gpt-oss:120b_vllm          → local vLLM endpoint
mixtral-8x7b_instruct_vllm → local vLLM
mixtral-8x7b_instruct       → cloud (if offered)
```

Your client code never needs to know whether the request will hit a remote HTTP endpoint or a local Docker container; the router decides based on the suffix.

### 3.1 Router Implementation (Python)

```python
import os
import httpx
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()
CLOUD_ENDPOINT = os.getenv("CLOUD_ENDPOINT", "https://api.openai.com/v1/chat/completions")
CLOUD_KEY = os.getenv("CLOUD_API_KEY")
LOCAL_VLLM_URL = os.getenv("LOCAL_VLLM_URL", "http://localhost:8000/v1/completions")

def is_local(model_name: str) -> bool:
    # Anything ending with "_vllm" or "_local" is routed locally
    return model_name.endswith(("_vllm", "_local"))

async def forward_to_cloud(payload: dict):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            CLOUD_ENDPOINT,
            json=payload,
            headers={"Authorization": f"Bearer {CLOUD_KEY}"}
        )
        resp.raise_for_status()
        return resp.json()

async def forward_to_local(payload: dict):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(LOCAL_VLLM_URL, json=payload)
        resp.raise_for_status()
        return resp.json()

@app.post("/v1/completions")
async def route(request: Request):
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model field required")
    # Strip the suffix before sending to the backend
    if is_local(model):
        body["model"] = model.replace("_vllm", "")
        return await forward_to_local(body)
    else:
        return await forward_to_cloud(body)
```

Deploy this router as a **single FastAPI service** behind your internal load balancer. All downstream services point to `https://inference.mycorp.com/v1/completions`. The router handles:

1. **Suffix detection** – decides cloud vs local.  
2. **Model name normalization** – removes the suffix before the backend sees it.  
3. **Transparent failover** – if the local endpoint returns a 5xx, the router can fall back to the cloud (optional, see Section 3.3).  

### 3.2 Deploying the Local vLLM Service  

```bash
# Pull the official vLLM image (as of 2024‑04)
docker run -d \
  --name vllm-120b \
  -p 8000:80 \
  -e VLLM_MODEL="meta-llama/Meta-Llama-3.1-120B" \
  -e VLLM_MAX_BATCH_SIZE=32 \
  ghcr.io/vllm-project/vllm:latest
```

**Key flags explained:**

| Flag | Meaning |
|------|---------|
| `VLLM_MODEL` | HuggingFace repo identifier (must be pre‑downloaded or pulled at container start). |
| `VLLM_MAX_BATCH_SIZE` | Upper bound for request batching; higher values improve GPU utilization but increase per‑request latency. |
| `-p 8000:80` | Exposes the vLLM HTTP API on port 8000 of the host. |

You can verify the endpoint with:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss:120b_vllm","messages":[{"role":"user","content":"Explain quantum tunneling"}],"max_tokens":128}'
```

The response mirrors the OpenAI schema, making the router’s job trivial.

### 3.3 Transparent Failover (Optional)

Add a retry policy inside `forward_to_local`:

```python
async def forward_to_local(payload: dict):
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(LOCAL_VLLM_URL, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            # Fallback to cloud if local is down
            return await forward_to_cloud(payload)
```

This pattern guarantees **high availability** without exposing the complexity to the caller. You can also add health‑check endpoints (`/healthz`) for both backends and let a service mesh (e.g., Istio) perform circuit‑breaking.

---

## 4. Build‑vs‑Buy Economic Traps  

Even with a solid routing policy, teams often fall into three classic cost‑illusion traps. Recognizing them early prevents budget blow‑outs.

### 4.1 “Self‑Hosting Is Cheaper” – The Scale Threshold  

A common mantra is “we already own GPUs, so we should self‑host”. The hidden cost is **utilization**. Let’s calculate a realistic scenario.

| Metric | Value |
|--------|-------|
| GPU model | NVIDIA A100 80 GB |
| Purchase price | $8,000 |
| Expected lifetime | 4 years |
| Power draw (full load) | 400 W |
| Electricity cost (US avg) | $0.13/kWh |
| Annual depreciation | $2,000 |
| Annual power cost (full load) | 400 W × 24 h × 365 d × $0.13/kWh ≈ $455 |

**Total annual cost per GPU** ≈ **$2,455** (depreciation + power).  

If you achieve **70 % average utilization**, the effective cost per GPU‑hour drops to:

```
$2,455 / (365 days × 24 h × 0.70) ≈ $0.40 per GPU‑hour
```

Now compare to a cloud API that charges $0.02 per 1 k tokens. Suppose your model generates 2 k tokens per request and you handle 10 k requests per hour:

```
Tokens per hour = 10 k × 2 k = 20 M tokens
Cost per hour = (20 M / 1 k) × $0.02 = $400
```

If the same workload can be served by a single A100 at $0.40 per hour, **self‑hosting wins**. But if you only hit 30 % utilization, the per‑hour cost rises to $0.57, and the cloud becomes cheaper. The break‑even utilization is roughly **55 %** for this token volume.

**Trap:** Assuming “ownership = cheap” without measuring real utilization. The remedy is to instrument GPU usage (e.g., `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`) and calculate **effective cost per inference**.

### 4.2 “Cloud Is Risky” – Data Classification Blind Spot  

Security teams sometimes block any outbound traffic to LLM APIs, citing “data leakage”. The reality is nuanced:

| Risk | Cloud mitigation | Self‑host mitigation |
|------|-------------------|----------------------|
| Data exfiltration | Encryption‑in‑transit, audit logs, data‑masking policies | Physical isolation, air‑gapped network |
| Model drift | Provider updates automatically | You must schedule upgrades manually |
| Vendor lock‑in | OpenAI’s `model` parameter is stable; alternative providers expose same schema | You own the model weights; you can switch frameworks |

If **only 10 %** of your payloads contain regulated data, you can **segregate** those requests to the local endpoint (suffix `_vllm`). The remaining 90 % can safely ride the cloud, benefiting from the latest model improvements. The “cloud is risky” narrative collapses once you adopt a **data‑classification pipeline**.

### 4.3 “We Already Have the GPU” – Sunk‑Cost Fallacy  

You might think, “We bought an A100 for training; let’s reuse it for inference.” Training workloads are **memory‑intensive** (up to 80 GB per model) and **short‑lived** (hours). Inference, especially for LLMs, is **throughput‑oriented** and often requires **batching** to achieve high utilization.

If you allocate the GPU exclusively for inference, you lose the ability to run **fine‑tuning** or **research experiments**. The opportunity cost can be expressed as:

```
Opportunity cost per hour = (Potential training runs per month) × (Revenue per trained model)
```

Suppose you could train a domain‑specific model once per month that brings $5,000 in revenue. If you lock the GPU for inference 24/7, you forfeit that $5,000. The **net cost** of self‑hosting becomes:

```
GPU operational cost + opportunity cost = $0.40/h + ($5,000 / 720 h) ≈ $7.34/h
```

Now the cloud API at $0.02 per 1 k tokens may be cheaper for the same inference load. The key is to **compare total economic impact**, not just the hardware purchase price.

---

## 5. Migration Signals – When to Flip the Switch  

Your routing policy should be **dynamic**, not static. Below are concrete, measurable signals that indicate it’s time to shift traffic in one direction or the other.

### 5.1 Signs to Move **More Traffic to the Cloud**

| Metric | Threshold | Why it matters |
|--------|-----------|----------------|
| GPU idle time (average) | > 70 % over 30 days | Under‑utilized hardware → wasted capital. |
| Ops cost (on‑call, monitoring, patching) | > 30 % of total ML budget | Human cost often eclipses raw compute cost. |
| API error rate (cloud) | < 0.1 % for 7 days | Cloud reliability is high; you can safely offload. |
| Token volume growth | > 25 % month‑over‑month for > 3 months | Elastic scaling avoids capacity planning headaches. |

**Action:** Gradually add the `_vllm` suffix to a subset of low‑sensitivity models, monitor latency, and increase the proportion until GPU utilization drops below 50 %.

### 5.2 Signs to Move **More Traffic to Self‑Hosted**

| Metric | Threshold | Why it matters |
|--------|-----------|----------------|
| Cloud spend (monthly) | > $10 k and rising > 15 % QoQ | Direct cost pressure. |
| Data classification upgrade | New regulation adds 40 % of payloads to “restricted” tier. |
| Latency SLA breaches | > 5 % of requests exceed 150 ms (chat) or 80 ms (real‑time). |
| Fine‑tuning demand | > 2 new fine‑tuned models per quarter. |

**Action:** Spin up a new vLLM container, copy the model weights locally, and start routing the newly‑restricted models with the `_vllm` suffix. Use the router’s health checks to ensure the local endpoint stays within the latency budget.

### 5.3 A Real‑World Migration Timeline  

| Week | Activity | KPI |
|------|----------|-----|
| 1‑2 | Instrument GPU utilization (`nvidia-smi` + Prometheus). | Baseline utilization 45 %. |
| 3‑4 | Deploy router, add `_vllm` suffix for 20 % of traffic (low‑sensitivity). | Latency increase < 10 ms, error rate < 0.2 %. |
| 5‑6 | Re‑balance: move another 30 % of traffic based on utilization. | GPU avg. utilization 62 %. |
| 7‑8 | Review cloud spend vs. GPU cost. | Cloud spend down 18 %, GPU cost up 12 %. |
| 9‑10 | Add failover to cloud for local outages. | Zero SLA breaches. |

This cadence demonstrates that **migration is incremental**, not a “big‑bang” switch. The router lets you test the waters with a single model name change.

---

## 6. Anti‑Pattern: One‑Time Decision & Forgetting to Re‑Evaluate  

A dangerous habit is to **lock in a single architecture** after an initial cost‑analysis and then never look back. The LLM ecosystem evolves at a breakneck pace:

| Change driver | Typical impact on decision |
|---------------|-----------------------------|
| Cloud API price cuts (e.g., $0.018 per 1 k tokens) | Cloud becomes more attractive for medium traffic. |
| New model releases (e.g., 200 B open‑source) | Self‑host cost may drop if the model is more efficient. |
| GPU price drops (e.g., H100 at $4,500) | Capital cost curve shifts, making self‑host cheaper at lower volumes. |
| Regulatory updates (e.g., EU AI Act) | Data‑sensitivity axis may swing toward self‑host. |

**Rule of thumb:** Re‑run the four‑axis matrix **every six months**. Automate the data collection:

```bash
# Example Prometheus query to export average GPU utilization
curl -G 'http://prometheus.mycorp.com/api/v1/query' \
  --data-urlencode 'query=avg_over_time(nvidia_gpu_utilization[30d])'
```

Store the results in a version‑controlled CSV and feed them into a simple Python script that recomputes the break‑even points. Treat the output as a **decision ticket** that must be reviewed by the product, finance, and security stakeholders.

---

## 7. Putting It All Together – A Practical Checklist  

1. **Classify your data** – Tag each request as `public` or `restricted`.  
2. **Measure traffic** – Capture tpm, token count, and latency per model.  
3. **Populate the matrix** – Fill the four‑axis table with real numbers.  
4. **Define suffix conventions** – Choose a clear naming scheme (`_vllm`, `_cloud`).  
5. **Deploy the router** – Use the FastAPI example, add health checks, enable failover.  
6. **Spin up local vLLM** – Containerize the models you need locally; expose the OpenAI‑compatible endpoint.  
7. **Instrument costs** – Log GPU‑hour usage, cloud token spend, and ops labor.  
8. **Set migration thresholds** – Encode the signals from Section 5 as alerts in your monitoring stack.  
9. **Schedule re‑evaluation** – Calendar event every 6 months; automate data pull and matrix recompute.  
10. **Iterate** – Adjust suffixes, add new models, or retire old ones based on the refreshed matrix.

By following this framework you turn the nebulous “build vs. buy” debate into a **data‑driven routing policy** that can evolve with your organization’s needs, regulatory landscape, and the rapidly shifting economics of LLM inference.

---  

### Final Thoughts  

You now have a concrete, repeatable process that starts with the **foundational truth**—no workload is purely “cloud” or “local”. The **four‑axis matrix** gives you a quantitative lens to decide where each request belongs. The **model‑suffix router** makes the decision invisible to downstream services while providing a safety net through transparent failover. And the **economic trap checklist** plus **migration signals** keep you from falling into the classic cost‑illusion pitfalls.

The real power of this approach is its **feedback loop**: as you collect more telemetry, the matrix updates, the router re‑balances, and you continuously chase the lowest total cost of ownership while respecting latency, compliance, and operational constraints. In a world where LLM APIs drop in price, new open‑source giants appear, and regulations tighten overnight, that loop is the only way to stay ahead.  

Start today by classifying a single high‑traffic model, add the `_vllm` suffix, and watch your GPU utilization climb. The rest of the framework will follow naturally. Happy routing!