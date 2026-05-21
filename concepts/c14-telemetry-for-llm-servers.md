# Telemetry for LLM Servers: What to Watch and Why

## 1. Why LLM‑specific telemetry isn’t just “more web metrics”

You probably already have a monitoring stack that tells you **bytes‑in**, **bytes‑out**, **HTTP 2xx/5xx rates**, and maybe **CPU load**. Those numbers are great for a classic web service that spends most of its time waiting for I/O or doing modest arithmetic. An LLM inference server, however, is more like a **restaurant kitchen** than a front‑of‑house dining room.

| Kitchen view                               | Web service view                      |
|--------------------------------------------|---------------------------------------|
| Oven temperature, stovetop flame, stock levels | Number of visitors, request size |
| How long a dish sits before it’s plated    | Request latency (p50/p99)            |
| How many chefs are busy vs idle            | CPU utilization                      |

If you only count “people walking through the door” (bytes‑in) you miss the fact that the **oven is at 150 °C** (GPU util) or that you’re **out of flour** (VRAM). A sudden spike in “people” could be harmless if the kitchen is already at capacity, or it could be a disaster if the oven is cooling down because the chef left it idle.

LLM inference is **GPU‑bound**, **memory‑bound**, and **state‑bound** (the KV cache). The metrics that tell you whether the oven is hot, whether you have enough flour, and whether the line of orders is growing are the ones that matter.

---

## 2. The must‑have signals for every LLM server

Below is the checklist you should keep on your monitoring dashboard. For each metric we explain *what* you should see, *why* it matters, and *how* to collect it.

### 2.1 GPU Utilization

| What to watch | Desired range | Why it matters |
|---------------|---------------|----------------|
| `gpu_util_percent` (0‑100) | **90‑100 %** during decode, **≥70 %** during batch pre‑fill | The GPU is the workhorse. If it’s idling, you’re paying for cycles you don’t use. |
| `gpu_mem_util_percent` (0‑100) | **≥95 %** sustained is OK, but watch for spikes that push you over 100 % | VRAM is a hard ceiling; once you cross it the kernel OOM‑kills the process. |

**Example**: With a single A100 (40 GB) running `vllm` on Llama‑2‑13B, you might see:

```bash
$ nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
98, 37800, 39960
```

That reads **98 % GPU util**, **37.8 GB used** of **39.96 GB total** – a healthy, fully‑loaded GPU.

### 2.2 VRAM usage vs total

VRAM is the pantry. If you keep pulling ingredients (model weights, KV cache, temporary tensors) without restocking, you’ll eventually run out and the kitchen will shut down.

| Metric | How to collect | Thresholds |
|--------|----------------|------------|
| `vram_used_bytes` | `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` | **>95 %** for >5 min → alert |
| `vram_free_bytes` | `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits` | **<2 GB** → risk of OOM |

**Worked example**: Your service runs two 70 B models (each ~30 GB) on a single 80 GB GPU. After a few minutes of heavy traffic you see:

```bash
$ nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
77000, 3000
```

That’s **77 GB used**, **3 GB free** – you’re teetering on the edge. A single extra request that adds 0.5 GB of KV cache could push you into OOM territory.

### 2.3 KV‑cache pool usage (vLLM)

The KV cache is a **refrigerated shelf** that stores “already‑cooked” token embeddings so you don’t have to recompute them for each subsequent token. vLLM exposes the pool size and usage via Prometheus metrics.

| Metric | Meaning | Healthy range |
|--------|---------|---------------|
| `vllm_kv_cache_used_bytes` | Bytes currently occupied by KV entries | **≤80 %** of `vllm_kv_cache_capacity_bytes` |
| `vllm_kv_cache_capacity_bytes` | Total bytes allocated for KV cache | Determined by `--kv-cache-dim` and `--max-num-batched-token` |
| `vllm_queue_depth` | Number of pending requests waiting for KV space | **≤5** for low latency, **≤20** before you need to scale |

**Concrete numbers**: Suppose you launch vLLM with:

```bash
vllm serve Llama-2-13b-chat-hf \
  --tensor-parallel-size 2 \
  --kv-cache-dim 128 \
  --max-num-batched-token 8192 \
  --port 8000
```

The KV cache capacity works out to roughly **12 GB**. If you see:

```bash
vllm_kv_cache_used_bytes 11534336
vllm_kv_cache_capacity_bytes 12582912
vllm_queue_depth 23
```

You’re at **92 %** capacity and the queue is growing. The next request will be forced to wait for space, inflating first‑token latency.

### 2.4 Queue depth

Queue depth is the **line of orders** waiting for the chef. A short line means the kitchen can keep up; a long line means you’re bottlenecked.

| Metric | How to collect | Normal range |
|--------|----------------|--------------|
| `request_queue_length` (vLLM) | `/metrics` endpoint → `vllm_queue_depth` | **0‑5** for low‑traffic, **≤20** for high‑traffic |
| `request_wait_time_seconds` | Custom instrumentation (time from arrival to start of decode) | **≤0.2 s** typical |

**Example**: During a load test you observe a sudden jump:

```bash
# curl -s http://localhost:8000/metrics | grep vllm_queue_depth
vllm_queue_depth 42
```

That’s a red flag: the queue is more than double the “healthy” ceiling.

### 2.5 Latency percentiles (p50/p99)

Latency is the **time from order placed to dish served**. Percentiles give you a realistic picture because the average can be skewed by outliers.

| Metric | Collection | Target |
|--------|------------|--------|
| `first_token_latency_seconds` (p50/p99) | vLLM `/metrics` → `vllm_first_token_latency_seconds` | p99 ≤ 150 ms for 13 B models on A100 |
| `end_to_end_latency_seconds` (p50/p99) | HTTP middleware → Prometheus histogram | p99 ≤ 1 s for chat‑style interactions |

**Real numbers**: A production endpoint serving 13 B Llama on a single A100 reports:

```
vllm_first_token_latency_seconds_bucket{le="0.05"} 1200
vllm_first_token_latency_seconds_bucket{le="0.1"}  3500
vllm_first_token_latency_seconds_bucket{le="0.2"}  4700
vllm_first_token_latency_seconds_bucket{le="+Inf"} 5000
```

From the histogram you can compute p99 ≈ **0.19 s**, which is comfortably below the 150 ms target.

### 2.6 Tokens‑per‑second (throughput)

Throughput is the **kitchen’s dishes‑per‑hour** metric. It aggregates many requests and tells you how efficiently you’re using the GPU.

| Metric | How to compute | Typical range |
|--------|----------------|---------------|
| `tokens_per_second` | `total_tokens_generated / total_time_seconds` (Prometheus `rate(vllm_generated_tokens_total[1m])`) | 150 tok/s per A100 for 13 B model (decode) |
| `requests_per_second` | `rate(http_requests_total[1m])` | 30 rps for chat‑style workloads |

**Sample calculation**:

```bash
# Assume Prometheus scraped the following over the last minute:
# vllm_generated_tokens_total  = 9,000
# vllm_generated_tokens_total{instance="gpu0"}  = 9,000
# vllm_generated_tokens_total{instance="gpu0"}[1m] = 9,000

tokens_per_sec = 9000 / 60 = 150 tok/s
```

If you see **tokens_per_second** dropping dramatically while GPU util stays high, you’re likely throttled by KV cache or queue depth.

### 2.7 Model swap events

When the working set exceeds VRAM, the runtime may **swap** a model in/out of GPU memory. Each swap incurs a **cold‑start penalty** (≈ 6 s on A100). Frequent swaps are a sign that you’re over‑committing VRAM.

| Metric | Source | Alarm |
|--------|--------|-------|
| `model_swap_total` | vLLM logs (`Model load` / `Model unload`) or custom counter | > 1 swap per 10 min → alert |

**Log snippet** (vLLM):

```
2024-05-20 12:03:14,321 INFO  Model Llama-2-13b-chat-hf loaded onto GPU 0 (VRAM 38.9 GB)
2024-05-20 12:12:57,874 INFO  Model Llama-2-13b-chat-hf unloaded from GPU 0 (freeing 38.9 GB)
2024-05-20 12:12:58,542 INFO  Model Llama-2-70b-chat-hf loaded onto GPU 0 (VRAM 78.2 GB)
```

Two swaps in ten minutes → you’re likely thrashing VRAM.

---

## 3. Warning signs that scream “something’s wrong”

You can think of each metric as a **gauge on the kitchen’s control panel**. When a gauge moves into the red, the underlying problem is often predictable.

| Symptom | Likely cause | Immediate action |
|---------|--------------|-------------------|
| **GPU util < 50 %** during peak traffic | Under‑provisioned batch size, or model is I/O‑bound (e.g., slow storage) | Increase `max_num_batched_token`, verify storage latency |
| **VRAM usage bouncing 80 % → 100 % → 80 %** | Model thrashing (swap in/out) or KV cache fragmentation | Consolidate models onto same GPU, raise `--kv-cache-capacity` |
| **KV cache at 99 % + queue depth rising** | Not enough cache for concurrent sessions | Add more GPU memory (larger GPU) or enable KV cache eviction (if supported) |
| **p99 first‑token latency spikes while end‑to‑end stays flat** | Queue waiting for KV cache, not compute | Scale out vLLM replicas, increase KV cache pool |
| **Tokens‑per‑second drops while GPU util stays high** | Bottleneck in request dispatch or network back‑pressure | Check host‑side CPU, NIC queue, and request routing logic |
| **Model swap events > 1 per hour** | Working set larger than VRAM, or aggressive autoscaling that evicts models | Reserve dedicated GPUs per model, or use model sharding across GPUs |

**Analogy**: Think of the kitchen’s **fire alarm** (GPU util) and **refrigerator temperature** (VRAM). If the fire alarm never rings but the fridge is constantly opening and closing, you know the problem isn’t the fire—it’s the cooling system.

---

## 4. Where the data lives: pulling the signals

Below we map each metric to the concrete source you can scrape or query.

### 4.1 NVIDIA‑SMI and NVML

`nvidia-smi` is the **hand‑held thermometer** you use for quick checks. For production you should embed NVML (C library) or use the **DCGM exporter** for Prometheus.

```bash
# One‑liner to emit GPU util and memory for Prometheus node‑exporter style
while true; do
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits | \
    awk -F',' '{printf "gpu_util{gpu=\"%s\"} %s\n", $2, $3/100; printf "gpu_mem_used_bytes{gpu=\"%s\"} %s\n", $2, $4*1024*1024}'
  sleep 5
done
```

For a more robust pipeline, spin up the **DCGM exporter**:

```yaml
# docker-compose.yml
services:
  dcgm-exporter:
    image: nvidia/dcgm-exporter:3.3.2
    ports:
      - "9400:9400"
    runtime: nvidia
    command: ["-f", "/etc/dcgm-exporter/dcp-metrics.csv"]
```

DCGM will expose `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`, etc., which Prometheus can scrape.

### 4.2 vLLM `/metrics`

vLLM ships a Prometheus endpoint that already aggregates most of the LLM‑specific signals.

```bash
# curl the endpoint and filter for KV cache
curl -s http://localhost:8000/metrics | grep vllm_kv_cache
```

Typical output:

```
# HELP vllm_kv_cache_used_bytes Current KV cache usage in bytes.
# TYPE vllm_kv_cache_used_bytes gauge
vllm_kv_cache_used_bytes 11534336

# HELP vllm_kv_cache_capacity_bytes Total KV cache capacity in bytes.
# TYPE vllm_kv_cache_capacity_bytes gauge
vllm_kv_cache_capacity_bytes 12582912

# HELP vllm_queue_depth Number of requests waiting for KV cache.
# TYPE vllm_queue_depth gauge
vllm_queue_depth 23
```

You can add a **Prometheus rule** to compute the percentage:

```yaml
# prometheus.yml
record: vllm:kv_cache_utilization
expr: vllm_kv_cache_used_bytes / vllm_kv_cache_capacity_bytes
```

### 4.3 Ollama logs for model load/unload

If you run Ollama, model swaps appear in its structured logs.

```bash
# tail the log and extract swap events
journalctl -u ollama -f | grep -E "Model (loaded|unloaded)"
```

Sample output:

```
2024-05-20T12:03:14.321Z INFO Model llama2:13b loaded onto GPU 0 (VRAM 38.9GB)
2024-05-20T12:12:57.874Z INFO Model llama2:13b unloaded from GPU 0 (freeing 38.9GB)
```

You can ship these logs to Loki/Elastic and create a **counter** in Prometheus via LogQL:

```logql
sum(increase({job="ollama"} |~ "Model .* loaded" [5m]))
```

### 4.4 HTTP latency histograms

Instrument your inference gateway (FastAPI, Flask, or gRPC) with a Prometheus histogram.

```python
# fastapi_example.py
from fastapi import FastAPI, Request
from prometheus_client import Histogram, Counter, start_http_server

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Latency of HTTP requests",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
)
ERROR_COUNTER = Counter("http_requests_total", "Total HTTP requests", ["code"])

app = FastAPI()

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    with REQUEST_LATENCY.labels(endpoint=request.url.path).time():
        response = await call_next(request)
    ERROR_COUNTER.labels(code=response.status_code).inc()
    return response
```

Run `start_http_server(8001)` in a separate thread to expose the `/metrics` endpoint for Prometheus.

---

## 5. Alerting rules that actually catch problems

A good alert is **actionable**, **low‑noise**, and **tied to a business SLA**. Below are concrete Prometheus rules you can drop into `alert.rules.yml`.

```yaml
groups:
- name: llm-inference.rules
  rules:
  # 1️⃣ VRAM saturation → OOM imminent
  - alert: VRAMHighUtilization
    expr: (gpu_mem_used_bytes / gpu_mem_total_bytes) > 0.95
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "GPU {{ $labels.gpu }} memory >95% for 5m"
      description: |
        VRAM usage on GPU {{ $labels.gpu }} is {{ printf \"%.2f\" $value }}.
        Expect OOM in the next few minutes. Consider scaling or freeing cache.

  # 2️⃣ p99 first‑token latency breach
  - alert: FirstTokenLatencySLOViolation
    expr: histogram_quantile(0.99, sum(rate(vllm_first_token_latency_seconds_bucket[5m])) by (le)) > 0.25
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "p99 first‑token latency >250 ms"
      description: |
        The 99th percentile first‑token latency has crossed 250 ms.
        Check KV cache usage and queue depth.

  # 3️⃣ GPU under‑utilization
  - alert: GPUUnderutilized
    expr: avg_over_time(gpu_util_percent[5m]) < 50
    for: 10m
    labels:
      severity: info
    annotations:
      summary: "GPU {{ $labels.gpu }} util <50% for 10m"
      description: |
        Your GPU is idle for most of the time. You may be over‑provisioned
        or batch size is too low.

  # 4️⃣ KV cache saturation + queue growth
  - alert: KVCacheFullAndQueueing
    expr: (vllm:kv_cache_utilization > 0.95) and (vllm_queue_depth > 10)
    for: 3m
    labels:
      severity: critical
    annotations:
      summary: "KV cache >95% and queue depth >10"
      description: |
        KV cache is almost full and requests are queuing. Scale out or increase cache.

  # 5️⃣ Model swap frequency
  - alert: FrequentModelSwaps
    expr: increase(model_swap_total[1h]) > 2
    for: 0m
    labels:
      severity: warning
    annotations:
      summary: "More than 2 model swaps in the last hour"
      description: |
        Frequent model loads/unloads indicate VRAM over‑commitment.
        Consolidate models or allocate dedicated GPUs.

  # 6️⃣ Tokens‑per‑second drop
  - alert: ThroughputDrop
    expr: rate(vllm_generated_tokens_total[5m]) < 0.5 * avg_over_time(rate(vllm_generated_tokens_total[1h])[1h])
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Tokens‑per‑second dropped <50% of hourly average"
      description: |
        Throughput has halved compared to the last hour. Investigate queue depth,
        network back‑pressure, or CPU bottlenecks.
```

**Why the `for:` clause matters**: Without a grace period you’ll get alerts on every transient spike (e.g., a single large request). The `for:` clause ensures the condition is **sustained**, reducing noise.

---

## 6. Building a dashboard that on‑call engineers love

A dashboard is the **control board** in the kitchen: you want a single glance to tell you whether the oven is hot, the fridge is full, and the line is moving.

### 6.1 Layout blueprint

| Row | Panels (in order) | Reason |
|-----|-------------------|--------|
| 1   | **GPU Util %** (time series) <br> **GPU Memory %** (stacked) | Immediate health of the compute resource |
| 2   | **KV Cache Util %** <br> **Queue Depth** (bars) | State‑bound resources and back‑pressure |
| 3   | **p99 First‑Token Latency** <br> **p99 End‑to‑End Latency** (line) | SLA‑related latency signals |
| 4   | **Tokens‑per‑Second** <br> **Requests‑per‑Second** (dual‑axis) | Throughput vs demand |
| 5   | **Model Swap Counter** (rate) <br> **Error Rate** (percentage) | Operational events and reliability |

All panels should use a **5‑minute rolling window** (Grafana “relative time” = `now-5m`). This smooths out short spikes but still reacts quickly enough for on‑call.

### 6.2 Example Grafana JSON snippet (GPU row)

```json
{
  "type": "graph",
  "title": "GPU Utilization & Memory",
  "targets": [
    {
      "expr": "gpu_util_percent",
      "legendFormat": "GPU {{gpu}} Util %",
      "refId": "A"
    },
    {
      "expr": "(gpu_mem_used_bytes / gpu_mem_total_bytes) * 100",
      "legendFormat": "GPU {{gpu}} Mem %",
      "refId": "B"
    }
  ],
  "yaxes": [
    { "format": "percent", "label": "Util / Mem", "min": 0, "max": 100 },
    { "format": "short", "show": false }
  ],
  "interval": "5s",
  "gridPos": { "h": 4, "w": 12, "x": 0, "y": 0 }
}
```

### 6.3 Color‑coding for quick triage

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| GPU util | > 85 % | 70‑85 % | < 70 % |
| VRAM usage | < 85 % | 85‑95 % | > 95 % |
| KV cache | < 80 % | 80‑95 % | > 95 % |
| Queue depth | 0‑5 | 6‑20 | > 20 |
| p99 latency | ≤ 150 ms | 150‑300 ms | > 300 ms |
| Model swaps | 0 | 1‑2 /h | > 2 /h |

Grafana’s **thresholds** let you embed these colors directly into the panels, turning the dashboard into a **traffic‑light system**.

### 6.4 “One‑page health check” checklist

When you open the page, run a mental checklist:

1. **GPU**: Both utilization and memory are in the green band.  
2. **KV cache**: Below 80 % and queue depth ≤ 5.  
3. **Latency**: p99 first‑token ≤ 150 ms, end‑to‑end ≤ 1 s.  
4. **Throughput**: Tokens‑per‑second matches expected baseline (e.g., 150 tok/s).  
5. **Events**: No model swaps in the last 30 min, error rate < 0.1 %.

If any item is red, you know exactly where to start digging.

---

## 7. Advanced patterns and future‑proofing

You now have the basics covered, but as your fleet grows you’ll encounter more subtle challenges.

### 7.1 Multi‑GPU sharding and cross‑GPU KV cache

When you split a 70 B model across **four A100s**, each GPU has its own KV cache slice. The aggregate cache utilization can be misleading if you only look at a single GPU.

**Solution**: Create a **recording rule** that sums KV cache usage across all GPUs:

```yaml
record: vllm:kv_cache_utilization_total
expr: sum(vllm_kv_cache_used_bytes) / sum(vllm_kv_cache_capacity_bytes)
```

Now you can alert on the **global** cache saturation, not just per‑GPU.

### 7.2 Adaptive batch sizing

If you notice GPU util dipping below 70 % during off‑peak hours, you can **dynamically lower `max_num_batched_token`** to reduce latency without sacrificing throughput. A simple controller can read `gpu_util_percent` from Prometheus and adjust the vLLM CLI flag via a side‑car process.

```python
# pseudo‑controller
if gpu_util < 70:
    new_batch = max(32, current_batch - 16)
else:
    new_batch = min(256, current_batch + 16)
# send HTTP POST to vLLM admin endpoint to update batch size
```

### 7.3 Predictive scaling with token‑rate forecasts

You can feed the **tokens‑per‑second** time series into a forecasting model (e.g., Prophet) to predict the next 15 minutes. If the forecast exceeds a threshold, spin up an extra vLLM replica **before** the queue depth spikes.

```python
from prophet import Prophet
import pandas as pd

df = pd.DataFrame({
    'ds': timestamps,
    'y': tokens_per_sec_series
})
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=15, freq='min')
forecast = m.predict(future)
if forecast['yhat'].iloc[-1] > 200:  # tokens/s threshold
    # trigger Kubernetes HPA or custom scaler
    scale_out()
```

### 7.4 Correlating host‑level metrics

Sometimes the GPU looks fine but latency still climbs. Correlate with **CPU steal**, **NUMA node pressure**, or **NIC queue depth**. Adding these to the same Grafana page gives you a **full stack view**.

```yaml
# CPU steal metric from node_exporter
record: node_cpu_steal_percent
expr: (rate(node_cpu_seconds_total{mode="steal"}[1m]) * 100)
```

If `node_cpu_steal_percent` > 5 % while GPU util is high, the host is contending for PCIe bandwidth.

### 7.5 Auditing model version drift

When you roll out a new model checkpoint, you may unintentionally increase VRAM usage (e.g., added adapters). Tag each model load with a **Prometheus label** (`model_version="v2.1"`) and track VRAM usage per version.

```yaml
# Example query: VRAM usage per model version
sum by (model_version) (gpu_mem_used_bytes{model_version=~".+"})
```

If a new version pushes usage to 98 % consistently, you know you need to either prune the model or allocate a larger GPU.

---

## 8. Putting it all together: a day in the life of a healthy LLM service

Imagine you’re on‑call for a SaaS that offers chat‑based LLM access. At 09:00 UTC you open the **LLM Health Dashboard**:

- **GPU Util**: 96 % (green)  
- **VRAM**: 92 % (yellow) – a slight uptick after a weekend model reload.  
- **KV Cache**: 68 % (green), **Queue**: 3 (green)  
- **p99 First‑Token**: 0.12 s (green)  
- **Tokens/s**: 148 (within baseline)  

All panels are green or yellow; you note the VRAM warning but decide it’s acceptable for now. At 10:15 UTC the **KV cache** line climbs to 94 % and **queue depth** spikes to 12. The dashboard’s traffic‑light turns yellow for KV cache and red for queue. You:

1. **Check the alert** – a `KVCacheFullAndQueueing` alert fires.  
2. **Run a quick `nvidia-smi`** to confirm VRAM is still at 93 %.  
3. **Scale out** the vLLM deployment by adding a second replica (via Kubernetes HPA).  
4. **Observe** the queue depth fall back to 4 within two minutes, and latency returns to baseline.

Later at 13:00 UTC the **GPU Util** drops to 48 % for a sustained 15 minutes. The `GPUUnderutilized` info‑level alert fires. You investigate:

- The request pattern shows a lull in traffic (half the usual RPS).  
- No model swaps.  
- CPU steal is 0 %.  

You decide to **temporarily lower the batch size** to improve per‑request latency, knowing the cost impact is negligible during low demand.

At 16:45 UTC you notice a **model swap** event in the logs. The `FrequentModelSwaps` warning appears because the swap rate crossed 2 per hour. You dig into the model catalog and realize a **new 70 B model** was added to the same GPU pool as the 13 B chat model. The solution: **pin the 70 B model to a dedicated GPU** and keep the 13 B model on the existing one. After the change, swap events disappear.

By the end of the day the dashboard remains green, the alerts have been resolved, and the SLA (p99 first‑token ≤ 150 ms) holds. This is the **steady‑state loop** you want: metrics surface problems early, alerts give you a clear action, and the dashboard confirms the fix.

---

## 9. Checklist for a production‑ready telemetry stack

| ✅ Item | How to verify |
|--------|---------------|
| **GPU util & memory** scraped via DCGM exporter | `curl http://dcgm-exporter:9400/metrics | grep gpu_util` |
| **vLLM KV cache & queue** exposed on `/metrics` | `curl http://vllm:8000/metrics | grep vllm_queue_depth` |
| **HTTP latency histograms** instrumented in inference gateway | Verify `http_request_latency_seconds_bucket` exists |
| **Model swap counter** logged and exported | `sum(increase(model_swap_total[5m]))` returns > 0 |
| **Alert rules** loaded in Prometheus (`/rules` endpoint) | `curl http://prometheus:9090/api/v1/rules` |
| **Grafana dashboard** with panels and thresholds | Open dashboard, confirm colors change with synthetic load |
| **SLA definition** (p99 first‑token ≤ 150 ms) documented and referenced in alerts | Alert annotation includes SLA target |
| **Automated scaling** (HPA or custom controller) reacting to tokens‑per‑second | Simulate load, watch replica count increase |
| **Log aggregation** (Loki/ELK) for model load/unload events | Search `Model .* loaded` returns recent entries |
| **Documentation** linking each metric to business impact | Internal wiki page with mapping table |

Running through this list once per quarter keeps your observability stack aligned with the evolving model fleet and traffic patterns.

---

## 10. Takeaway: telemetry that tells a story, not just numbers

You’ve learned that **bytes‑in/out** are the equivalent of counting how many people walked into a restaurant. They’re interesting, but they don’t tell you whether the kitchen is burning, the fridge is empty, or the line is too long. By focusing on **GPU utilization**, **VRAM pressure**, **KV cache health**, **queue depth**, **latency percentiles**, **throughput**, and **model swap events**, you get a narrative that directly maps to cost, performance, and reliability.

When you instrument these signals correctly, feed them into a **low‑noise alerting system**, and surface them on a **single‑page dashboard** with traffic‑light thresholds, on‑call engineers can diagnose and remediate issues in minutes instead of hours. The analogies to a kitchen, a plumbing system, and a highway help you keep the mental model clear: keep the furnace hot, the pipes clear, and the traffic flowing.

Now you have a concrete, production‑ready telemetry blueprint. Plug it into your LLM serving stack, watch the graphs stabilize, and let the alerts guide you to a healthier, more cost‑effective inference service. Happy monitoring!