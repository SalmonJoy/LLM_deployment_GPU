# The Economic Reality of GPU Hosting: Dollars-per-Hour vs Dollars-per-Token

## 1. Why the “per‑hour” metric feels right – and why it’s only half the story  

When you first glance at a cloud provider’s catalog you see a line that reads something like:

| Region | Instance Type | Hourly Price |
|--------|---------------|--------------|
| us‑east‑1 | `Standard_ND96as_v4` (8 × NVIDIA H100) | **$12.00** |
| eu‑central‑1 | `Standard_ND96as_v4` | **$9.50** |
| ap‑south‑1 | `Standard_ND96as_v4` | **$4.20** |

Those numbers look a lot like a **rent‑check** for a high‑end office. The analogy most engineers use is “you pay rent for the space, whether you’re working in it or not.” It works because the cloud’s billing model is *time‑based*: the moment the virtual machine (VM) is provisioned, the clock starts ticking.

Think of the GPU as a **high‑capacity kitchen stove**. You rent the stove by the hour, but you only get value when you’re actually cooking. If you leave the burners on all night while you’re sleeping, you’re still paying for the gas that never touches a pot. The same principle applies to an H100: it draws power, incurs depreciation, and occupies a data‑center slot even when it’s idle.

### 1.1 Real‑world command to spin up an H100 VM (Azure)

```bash
az vm create \
  --resource-group rg-llm \
  --name h100-prod \
  --image UbuntuLTS \
  --size Standard_ND96as_v4 \
  --admin-username azureuser \
  --generate-ssh-keys
```

Once that command finishes, Azure starts billing you **$4‑$12 per hour** (depending on the region and any reserved‑instance discounts you may have negotiated). There’s no “free‑tier” for the GPU; the moment the VM exists, the meter runs.

## 2. The hidden cost: idle capacity  

### 2.1 The “24‑hour kitchen” problem  

Imagine you run a restaurant that only serves lunch from 11 am – 2 pm. You’ve rented a kitchen that costs $10 per hour, but you keep the burners on from 6 am – 10 pm. The extra 13 hours of idle heat are a **pure waste**. The same thing happens with GPUs.

If you keep an H100 VM up 24 hours a day but only receive inference traffic during a typical 9‑5 business day, you’re paying for **16 hours of dead weight** every day.

### 2.2 Concrete cost‑saving pattern: daily de‑allocate / re‑allocate  

A common pattern is to **de‑allocate** the VM at night and **re‑start** it in the morning. De‑allocation releases the underlying hardware, so you stop incurring the hourly charge (you still pay for the attached storage, but that’s a fraction of the GPU cost).

#### Example schedule

| Action | Time (local) | Cost impact |
|--------|--------------|-------------|
| De‑allocate | 20:00 | Stops $4‑$12/hr charge |
| Re‑allocate | 09:00 | Resumes charge |

That gives you **13 hours saved per day**.

#### Daily savings calculation (low‑end $4/hr)

```
13 hours × $4/hr = $52 per day
13 hours × $12/hr = $156 per day
```

Over a 30‑day month you’d save **$1,560 – $4,680** simply by turning the GPU off when you’re not serving requests.

### 2.3 Automation with Azure CLI + cron

```bash
# Save this as /usr/local/bin/h100-scheduler.sh
#!/usr/bin/env bash
ACTION=$1   # start or stop
VM_NAME="h100-prod"
RG="rg-llm"

if [[ "$ACTION" == "stop" ]]; then
  az vm deallocate --resource-group $RG --name $VM_NAME
elif [[ "$ACTION" == "start" ]]; then
  az vm start --resource-group $RG --name $VM_NAME
else
  echo "Usage: $0 {start|stop}"
fi
```

Add to crontab (run as root or a service account with Azure CLI auth):

```
0 20 * * * /usr/local/bin/h100-scheduler.sh stop   # 20:00 daily
0 9  * * * /usr/local/bin/h100-scheduler.sh start  # 09:00 daily
```

Now the VM follows the same “restaurant‑close‑at‑night” schedule automatically, and you never forget to shut it down.

## 3. From dollars‑per‑hour to dollars‑per‑token  

### 3.1 Token throughput fundamentals  

A **token** is roughly 4 characters of English text (≈0.75 words). Modern LLMs such as GPT‑4‑Turbo or Claude can generate **hundreds of tokens per second** on a single H100. Let’s pick a concrete figure: **116 tokens / second** (a realistic throughput for a 70 B‑parameter model on an H100 when running inference with batch‑size 1).

```
Tokens per hour = 116 tok/s × 3600 s/h = 417,600 tok/h
```

### 3.2 Cost per token at 100 % utilization  

Take the low‑end $4/hr price:

```
$4.00 / 417,600 tok ≈ $0.00000958 per token
```

That’s **$9.58 per 1 M tokens**. At the high‑end $12/hr price the number triples to **$28.74 per 1 M tokens**.

| Hourly price | Tokens/h | $/token | $/M tokens |
|--------------|----------|---------|------------|
| $4           | 417,600  | $9.58   | $9.58      |
| $8           | 417,600  | $19.16  | $19.16     |
| $12          | 417,600  | $28.74  | $28.74     |

### 3.3 What happens when you’re only 25 % busy?  

If you only serve traffic for 6 hours a day (25 % of a full day), the **effective** cost per token rises proportionally because you still pay the full hour price for the idle time.

```
Effective tokens per hour = 0.25 × 417,600 = 104,400 tok/h
$4 / 104,400 ≈ $0.0000383 per token → $38.3 per 1 M tokens
```

| Utilization | Effective tokens/h | $/M tokens (at $4/hr) |
|-------------|-------------------|-----------------------|
| 100 %       | 417,600           | $9.58                 |
| 75 %        | 313,200           | $12.78                |
| 50 %        | 208,800           | $19.17                |
| 25 %        | 104,400           | $38.34                |
| 10 %        | 41,760            | $95.85                |

The math makes it crystal clear: **idle capacity is not a free lunch**; it inflates your per‑token price dramatically.

## 4. API pricing side‑by‑side  

Most teams compare self‑hosted GPU costs against the public API rates of OpenAI, Anthropic, or Cohere. Below is a snapshot of the **pay‑as‑you‑go** pricing as of mid‑2026:

| Provider | Model (approx.) | $/M tokens (prompt + completion) |
|----------|----------------|----------------------------------|
| OpenAI   | gpt‑4o         | $5 – $15                         |
| Anthropic| Claude‑3‑Opus  | $3 – $15                         |
| Cohere   | Command‑R+     | $4 – $12                         |

These ranges reflect **tiered discounts** (volume, committed spend) and **prompt vs. completion** splits. The key takeaway: the **public‑API sweet spot sits between $3 and $15 per million tokens**.

If your self‑hosted GPU costs **more than $15 per million tokens**, the API is cheaper *even before* you factor in hidden ops overhead.

## 5. Break‑even analysis – when does self‑hosting actually win?  

Let’s formalize the break‑even point. Define:

- `P` = hourly price (e.g., $4)
- `T` = tokens per second at full utilization (e.g., 116)
- `U` = utilization fraction (0 – 1)
- `C_api` = API cost per million tokens (e.g., $15)

Self‑hosted cost per million tokens:

```
cost_per_M = (P / (T × 3600 × U)) × 1,000,000
```

Set `cost_per_M = C_api` and solve for `U`:

```
U_break = P / (C_api × T × 3600 / 1,000,000)
```

Plugging numbers (`P=$4`, `T=116`, `C_api=$15`):

```
U_break = 4 / (15 × 116 × 3600 / 1,000,000)
        = 4 / (15 × 417,600 / 1,000,000)
        = 4 / (6.264)
        ≈ 0.639  → 63.9 %
```

**Interpretation:** With a $4/hr H100 you must keep the GPU busy **at least 64 % of the time** (≈15.4 hours per day) to beat a $15/1M‑token API. If you can only achieve 40 % utilization, the API wins.

### 5.1 Break‑even table for different hourly rates

| Hourly price | Break‑even utilization (vs $15/M) |
|--------------|-----------------------------------|
| $4           | 64 %                              |
| $8           | 32 %                              |
| $12          | 21 %                              |

Notice how a higher hourly price **lowers** the utilization needed to break even because you’re already paying more per hour; you need to squeeze more tokens out of each hour to justify it.

## 6. Hidden ops costs that turn the numbers upside‑down  

Even if the raw $/token math looks favorable, you must add the **operational overhead** that every self‑hosted deployment incurs.

| Category | Typical cost factor | What it includes |
|----------|--------------------|------------------|
| Engineering time | 30‑50 % of GPU cost | Model integration, CI/CD pipelines, GPU driver updates |
| Model download & storage | $0.10‑$0.30 per GB per month | 80 GB for a 70 B model + checkpoint sharding |
| Monitoring & alerting | $0.05‑$0.15 per GB of logs | Prometheus, Grafana, Loki |
| Security & compliance audits | Fixed $2k‑$5k per audit | Pen‑test, IAM hardening, data‑at‑rest encryption |
| Cloud‑network egress | $0.09 per GB (AWS) | Streaming responses to end‑users |

### 6.1 Quantifying a 30 % ops overhead  

Assume you’re paying $8/hr for the GPU (mid‑range region). Monthly GPU cost for 24/7 operation:

```
$8/hr × 24 hr/day × 30 days = $5,760
```

Add a **30 % ops buffer**:

```
Ops overhead = 0.30 × $5,760 = $1,728
Total = $7,488 per month
```

Now compute $/M tokens at 100 % utilization:

```
Tokens/month = 116 tok/s × 3600 s/h × 24 h/d × 30 d = 301,056,000 tok
$ per M tokens = $7,488 / (301.056) ≈ $24.87 per M tokens
```

Even with perfect utilization, you’re now **$24.87 per million tokens**, already above the $15 API ceiling. If utilization drops to 50 %, the cost doubles to **≈$49.7 per M tokens**.

### 6.2 Real‑world ops script – automated model version roll‑out

```yaml
# .github/workflows/deploy-llm.yml
name: Deploy LLM
on:
  push:
    tags:
      - 'v*.*.*'
jobs:
  download-model:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v3
      - name: Pull model from S3
        run: |
          aws s3 cp s3://my-llm-bucket/70B-v${{ github.ref_name }}.tar.gz /tmp/
          tar -xzvf /tmp/70B-v${{ github.ref_name }}.tar.gz -C /opt/models/
      - name: Restart VM
        run: |
          az vm deallocate -g rg-llm -n h100-prod
          az vm start -g rg-llm -n h100-prod
```

Every new tag triggers a download of a **multi‑gigabyte checkpoint**, a storage write, and a VM bounce. The engineering time to maintain this pipeline (review, test, rollback) is non‑trivial and should be baked into the overhead factor.

## 7. Scenarios where self‑hosting still makes sense  

Even when the pure $/token metric favors an API, there are **strategic reasons** to run your own GPU cluster.

### 7.1 Data privacy & regulatory compliance  

- **Prompt leakage**: Public APIs log every request for abuse detection. If you handle PHI, PII, or classified data, you may be legally required to keep prompts on‑premise.
- **Jurisdiction**: Some countries mandate that data never leave national borders. Hosting an H100 in a compliant region eliminates the cross‑border transfer risk.

### 7.2 Fine‑tuned or proprietary models  

- **Domain‑specific vocabularies**: You might have a custom‑trained 70 B model that outperforms GPT‑4 on legal contracts. No public API offers that model.
- **Intellectual property**: The model weights themselves are a competitive asset. Keeping them in your own VPC prevents accidental exposure.

### 7.3 Predictable high‑volume workloads  

If you know you will generate **tens of billions of tokens per month** (e.g., a large‑scale translation service), the **fixed‑cost nature** of a GPU fleet gives you a ceiling you can budget against, unlike a usage‑based API that could spike dramatically.

#### Example: 10 B tokens per month

- API at $10/M → $100,000/month.
- Self‑hosted: 4 H100s at $8/hr each, 24/7 → $8 × 4 × 24 × 30 = $23,040.
- Add 30 % ops overhead → $29,952.
- **Savings**: $70k+ per month, even after ops.

### 7.4 Latency‑critical applications  

Public APIs incur **network RTT** (often 40‑80 ms) plus queuing latency in the provider’s inference farm. For real‑time chatbots, autonomous agents, or gaming AI, you may need **sub‑10 ms** response times, achievable only when the GPU sits in the same VPC or edge location as your service.

### 7.5 Experimentation speed  

When you own the hardware, you can:

- Run **batch inference** with custom batch sizes.
- Profile kernel execution with `nsight-systems`.
- Swap out the model version instantly without waiting for a provider’s rollout schedule.

These productivity gains are hard to quantify but can shave weeks off a product roadmap.

## 8. Decision framework – a checklist you can run in a CI pipeline  

Below is a **decision matrix** you can paste into a markdown file and tick off as you evaluate a new LLM project.

| Question | Yes → Favor self‑host | No → Favor API |
|----------|----------------------|----------------|
| Do you need **sub‑10 ms** latency for > 90 % of requests? | ✅ | ❌ |
| Are you handling **regulated data** that cannot leave the data‑center? | ✅ | ❌ |
| Will you generate **> 5 B tokens/month**? | ✅ (economies of scale) | ❌ |
| Do you have **in‑house GPU ops expertise** (driver, NCCL, monitoring)? | ✅ | ❌ |
| Is your **utilization forecast** > 60 % (or > 30 % for $8/hr)? | ✅ | ❌ |
| Do you need **custom fine‑tuned weights** that are not available via API? | ✅ | ❌ |
| Is the **budget for security audits** (≥ $2k per quarter) already allocated? | ✅ | ❌ |
| Are you comfortable with **monthly storage costs** for multi‑TB checkpoints? | ✅ | ❌ |
| Do you have **reserved‑instance discounts** that bring hourly price below $4/hr? | ✅ | ❌ |
| Do you need **instant rollback** of model versions? | ✅ | ❌ |

If the majority of rows are “Yes”, self‑hosting is worth a deeper financial model. If “No” dominates, the API path is likely cheaper and faster to market.

## 9. Putting the numbers together – a full‑stack cost model  

Below is a **sample Python script** that ingests your region, hourly price, expected utilization, and ops overhead, then spits out the $/token and a break‑even comparison against a chosen API price.

```python
#!/usr/bin/env python3
import argparse

def cost_per_million(hourly_price, tokens_per_sec, utilization, ops_factor):
    # Effective tokens per hour
    tokens_per_hour = tokens_per_sec * 3600 * utilization
    # Total monthly cost (30 days) with ops overhead
    monthly_gpu = hourly_price * 24 * 30
    monthly_total = monthly_gpu * (1 + ops_factor)
    # Tokens per month
    tokens_month = tokens_per_hour * 24 * 30
    cost_per_m = monthly_total / (tokens_month / 1_000_000)
    return cost_per_m

def break_even_utilization(hourly_price, tokens_per_sec, api_price):
    # Solve for U where cost_per_m = api_price
    # api_price = hourly_price / (tokens_per_sec * 3600 * U) * 1e6
    U = hourly_price * 1_000_000 / (api_price * tokens_per_sec * 3600)
    return U

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hourly", type=float, required=True, help="GPU hourly price")
    p.add_argument("--tps", type=float, default=116, help="Tokens per second at full load")
    p.add_argument("--util", type=float, default=0.5, help="Utilization fraction (0‑1)")
    p.add_argument("--ops", type=float, default=0.3, help="Ops overhead fraction")
    p.add_argument("--api", type=float, default=15, help="API $/M tokens")
    args = p.parse_args()

    cost = cost_per_million(args.hourly, args.tps, args.util, args.ops)
    be = break_even_utilization(args.hourly, args.tps, args.api)

    print(f"Effective $/M tokens (incl. {int(args.ops*100)}% ops): ${cost:.2f}")
    print(f"Break‑even utilization vs ${args.api}/M API: {be*100:.1f}%")
```

Run it like:

```bash
python3 cost_model.py --hourly 8 --util 0.4 --ops 0.35 --api 12
```

You’ll get a quick, reproducible estimate that you can embed in a cost‑review PR.

## 10. Real‑world case study – a fintech chatbot  

**Background**: A fintech startup needed a conversational assistant that could answer account‑balance queries, compliance questions, and generate regulatory disclosures. They processed ~2 M tokens per day (≈ 60 M per month) and required **sub‑20 ms latency** because the UI was a mobile app with a tight user‑experience budget.

### 10.1 Initial cost model (API)

- Chosen API: OpenAI gpt‑4o at $12/M tokens.
- Monthly token cost: 60 M × $12 = **$720**.
- No ops overhead, but latency measured at **45 ms** (including internet RTT).

### 10.2 Self‑hosted pilot

- Deployed a single `Standard_ND96as_v4` (8 × H100) in the same Azure region as the app.
- Hourly price: $9.50 (eu‑central‑1).
- Utilization: 70 % (the bot was active 16 h/day, idle 8 h).
- Ops overhead: 30 % (monitoring, model versioning).

**Cost breakdown**:

| Item | Monthly cost |
|------|--------------|
| GPU (24/7) | $9.50 × 24 × 30 = $6,840 |
| Ops overhead (30 %) | $2,052 |
| **Total** | **$8,892** |

**Token throughput**:

- 116 tok/s × 3600 × 24 × 30 × 0.70 = **210 M tokens** (well above the 60 M needed).
- $/M tokens = $8,892 / 210 ≈ **$42.34**.

At first glance the per‑token price is **3‑4× higher** than the API. However, the **latency** dropped to **9 ms** (GPU in‑VPC, no internet hop) and the team gained **full control over model updates** (they rolled a custom compliance fine‑tune in a week).

### 10.3 Decision  

Because the product’s **value proposition** hinged on ultra‑low latency and data residency (EU‑only storage), the startup accepted the higher $/token cost. They also negotiated a **reserved‑instance contract** that cut the hourly price to $6.80, bringing the monthly total down to ~$6,300 and $/M tokens to **$30**, still above the API but within the budget for a premium feature.

**Lesson**: The $/token metric is a decisive factor, but **business requirements** (latency, compliance, brand differentiation) can justify a higher spend.

## 11. Practical tips to squeeze more value out of your H100  

| Tip | How it works | Approx. savings |
|-----|--------------|-----------------|
| **Batch inference** (e.g., `torch.compile` + `torch.cuda.Stream`) | Process multiple requests together, increasing `tokens_per_sec` by 1.5‑2× | Reduces $/M tokens by up to 40 % |
| **Spot instances** (if workload is tolerant to interruptions) | Pay 60‑70 % of on‑demand price, with automatic checkpointing | $2‑$4/hr vs $8‑$12/hr |
| **Multi‑tenant sharing** (run two services on the same GPU) | Partition GPU memory with MIG (Multi‑Instance GPU) | Effectively halves per‑service cost |
| **Reserved capacity** (1‑year commitment) | Up to 30 % discount on hourly price | $8 → $5.6/hr |
| **Cold‑start caching** (keep model in RAM, avoid re‑load) | Avoid the 30‑second load penalty on each start | Improves throughput, reduces idle time |

### 11.1 Example: Enabling MIG on an H100

```bash
# Install NVIDIA driver and nvidia-smi (Azure images usually have it)
sudo nvidia-smi -i 0 -mig 1  # enable MIG mode
# Create two 1‑GPU instances (each gets 1/2 of the physical GPU)
sudo nvidia-smi -i 0 -cgi 19,19   # 19 = 1/2 GPU profile
sudo nvidia-smi -i 0 -lgc 0,0     # lock clocks to max for stability
```

Now you can run two Docker containers, each bound to a different MIG instance (`--gpus '"device=0:0"'` and `--gpus '"device=0:1"'`). Each container sees a **virtual GPU** with half the memory and compute, but you’ve effectively doubled the **service density** on a single H100.

## 12. Summing up the economics  

- **Visible cost** is a simple hourly rent: $4‑$12 per hour for an H100 VM.
- **Idle cost** can be the biggest leak; a daily de‑allocate schedule can save **$52‑$156 per day**.
- **$ per token** depends linearly on utilization. At 100 % you’re around **$9‑$29 per M tokens**; at 25 % it balloons to **$38‑$115 per M tokens**.
- Public APIs sit in the **$3‑$15 per M token** band. Self‑hosting only beats them when you sustain **high utilization** (≥ 40‑60 % depending on hourly price) and have **low ops overhead**.
- **Ops overhead** (engineering, monitoring, security) adds **30‑50 %** to the raw GPU bill, often pushing the effective $/token above the API range.
- **Strategic factors**—privacy, fine‑tuning, latency, predictable high‑volume spend—can outweigh pure dollar calculations.

When you walk through the numbers with the scripts, tables, and analogies above, the decision becomes a **multidimensional trade‑off** rather than a simple “cheaper vs. more expensive” question. Use the checklist, run the cost model, and you’ll have a data‑driven answer that aligns with both your budget and your product goals.