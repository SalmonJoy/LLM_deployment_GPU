# When Your LLM Endpoint Lies: Detecting Hidden Cloud Routing in API Proxies  

---

## 1. Why “localhost” can be a Trojan horse  

You’ve probably spun up an LLM server on your laptop or on a bare‑metal box and exposed it as  

```bash
http://localhost:11434/api/chat
```  

The URL screams “local”, “private”, “under my control”. Yet, in many modern AI stacks that address is **just a façade**. The process listening on `:11434` may be a thin HTTP wrapper that forwards **some** model calls to a remote SaaS endpoint (e.g., `api.ollama.com`) while keeping the rest on‑prem.  

Think of it like a **receptionist at a corporate office**. When you walk in, the desk greets you with “Welcome, you’re on the 3rd floor”. The receptionist then decides, based on the department you need, whether to escort you down the hallway (local) or hand you a badge that routes you through a **different building** (cloud). The visitor never knows which door they actually walked through.  

In the world of LLM APIs the “receptionist” is often a **proxy process** (sometimes called a “wrapper”, “gateway”, or “router”) that:

| Layer | What you think you’re talking to | What actually happens |
|-------|----------------------------------|-----------------------|
| **Port 11434** | Ollama’s public HTTP API (`/api/chat`) | Wrapper that may forward to `api.ollama.com` for certain model names |
| **Port 11433** | Ollama’s raw gRPC / internal HTTP API | Direct, no‑frills access to the locally stored model binaries |
| **Docker network** | Isolated container network | May have NAT rules that send outbound traffic to the internet without you noticing |

If you never peek behind the receptionist’s desk, you’ll assume every request is handled locally, and you’ll be surprised when your GPU usage stays at 0 % while responses arrive in a flash.

---

## 2. The smoking‑gun symptom: “Impossible” latency  

The most common clue that your endpoint is cheating you is **response time** that defies physics.  

| Model | Expected cold‑load time on a 24 GB RTX 4090 | Observed time on “localhost” |
|-------|--------------------------------------------|------------------------------|
| `llama2-70b` (70 B parameters) | 300 s (≈5 min) to load + 2 s per 100 tokens | 0.8 s total for a 150‑token reply |

If a **120 B** model returns a coherent answer in **< 1 s**, you can be almost certain that the request never touched your GPU. Even a **warm** model (already resident in GPU memory) typically needs **≥ 30 ms per token** for a 4090; a 150‑token response should be at least **4–5 s**.  

That “instant” answer is the **smoking gun** that the wrapper has silently delegated the request to a cloud service that is pre‑warmed on massive clusters. The wrapper then packages the remote answer and sends it back to you as if it came from the local model.

---

## 3. My false leads – what not to chase first  

When I first saw the impossible latency, I went down a rabbit hole of **environment variables** and **CPU‑only inference**. Here’s a chronological log of my missteps, so you can avoid them.

### 3.1. The `OLLAMA_REMOTES` red herring  

```bash
export OLLAMA_REMOTES="registry.ollama.ai"
```

I assumed this variable controlled **where inference happened**. In reality, `OLLAMA_REMOTES` tells Ollama **where to pull model binaries from** (the *registry*). It does **not** affect the runtime routing logic. Changing it only influences the *download* step, not the *execution* step.

### 3.2. Switching to CPU and seeing 0 % CPU  

```bash
ollama run --cpu llama2
```

The `top` output showed:

```
PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
1234 ollama    20   0  2.3G  1.2G  200M S   0.0  5.0   0:00.02 ollama
```

Zero CPU usage made me think the model was still loading from disk, not that the request had been handed off to a remote service. The truth: the wrapper spun up a **background thread** that immediately opened an outbound HTTPS connection, streamed the answer, and then terminated. The CPU spent almost all its time in kernel networking, which `top` shows as **0 %** for the user‑space process.

### 3.3. The “pull‑only” mindset  

I kept checking the model cache directory (`~/.ollama/models`) and saw the 120 B model file present, so I convinced myself the model *must* be local. The presence of a binary does **not guarantee** that the wrapper will use it; the wrapper can still decide to ignore the local copy based on its internal routing table.

---

## 4. The correct diagnostic: bypass the wrapper  

The key to proving (or disproving) local execution is to **talk directly to the lowest layer** that actually loads the model into GPU memory. Ollama ships two HTTP listeners:

| Port | Purpose | Typical URL |
|------|---------|-------------|
| **11433** | Raw inference API (no routing) | `http://localhost:11433/api/generate` |
| **11434** | Public API (may route) | `http://localhost:11434/api/chat` |

If you hit **11433** you eliminate the wrapper entirely.

### 4.1. Measuring cold‑load time on the raw port  

```bash
time curl -X POST http://localhost:11433/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2-120b","prompt":"Explain quantum tunneling in one sentence."}'
```

Sample output on a fresh machine:

```
real    6m20.134s
user    0m2.019s
sys     0m1.345s
```

The **real** time of **380 s** matches the expected GPU load for a 120 B model. The response payload is a single sentence, confirming the model was indeed loaded locally.

### 4.2. Comparing with the wrapper port  

```bash
time curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2-120b","messages":[{"role":"user","content":"Explain quantum tunneling in one sentence."}]}'
```

Result:

```
real    0m0.842s
user    0m0.012s
sys     0m0.005s
```

The **0.84 s** latency is impossible for a cold load, confirming that the wrapper is **bypassing** the local binary and calling the remote service.

---

## 5. Reading the wrapper logs – the hidden subscription error  

When you enable verbose logging for the wrapper (usually via `OLLAMA_LOG=debug`), you’ll see the exact error that the remote service returns. In my case the log line was:

```
2024-05-19T14:32:11Z DEBUG ollama._types.ResponseError: this model requires a subscription
```

That message is **not** generated by the local binary; it comes from **ollama.com**’s SaaS tier, which returns a 403 with a JSON body:

```json
{
  "error": "this model requires a subscription"
}
```

The wrapper catches the HTTP 403, translates it into a Go error, and forwards it as a JSON response. If you never look at the logs, you’ll miss this clue entirely.

---

## 6. A reusable three‑rule diagnostic checklist  

Below is a **compact, repeatable process** you can embed in your incident‑response playbook. It works for any LLM‑as‑a‑service stack that sits behind an HTTP proxy (Ollama, LMStudio, private‑cloud gateways, etc.).

| Rule | What you do | Why it matters |
|------|--------------|----------------|
| **1️⃣ Snapshot processes, ports, and network connections** | ```bash\nps -ef | grep ollama\nnetstat -tulpn | grep :1143[4]\nlsof -i :11434\n``` | Captures the exact binaries listening, their PIDs, and any outbound connections (e.g., to `api.ollama.com`). |
| **2️⃣ Hit the lowest layer directly** | Use the raw port (`11433` for Ollama) or the container’s internal socket (`unix:///run/ollama.sock`). | Guarantees you bypass any routing logic; you measure the true local latency. |
| **3️⃣ Read the bytes, don’t discard them** | ```bash\ncurl -v -X POST http://localhost:11434/api/chat \\\n  -H "Content-Type: application/json" \\\n  -d '{"model":"llama2-120b","messages":[{"role":"user","content":"test"}]}'\n``` | `-v` shows the full request/response cycle, including redirects, TLS handshakes, and any error payloads that the wrapper may swallow when you pipe to `/dev/null`. |

### 6.1. Applying the checklist step‑by‑step  

1. **Snapshot**  
   ```bash
   $ ps -ef | grep ollama
   ollama   1842  1  0 10:12 ?        00:00:12 /usr/local/bin/ollama serve --port 11434
   $ netstat -tulpn | grep :11434
   tcp   0   0 127.0.0.1:11434   0.0.0.0:*   LISTEN   1842/ollama
   $ lsof -i :11434 | grep ESTABLISHED
   ollama   1842 ollama   7u  IPv4 0x1234567890abcdef 0t0  TCP 127.0.0.1:11434->93.184.216.34:443 (ESTABLISHED)
   ```
   The `ESTABLISHED` line shows an outbound TLS connection to `93.184.216.34` (a placeholder for `api.ollama.com`). That’s the first hint that the wrapper is reaching out.

2. **Hit the raw port**  
   ```bash
   $ time curl -s -X POST http://localhost:11433/api/generate \
        -H "Content-Type: application/json" \
        -d '{"model":"llama2-120b","prompt":"Summarize the plot of Inception."}'
   ```
   Observe the wall‑clock time; if it’s in the **minutes** range, you have a local load.

3. **Read the bytes**  
   ```bash
   $ curl -v -X POST http://localhost:11434/api/chat \
        -H "Content-Type: application/json" \
        -d '{"model":"llama2-120b","messages":[{"role":"user","content":"Summarize Inception."}]}'
   ```
   The `> GET /v1/chat/completions` line will be followed by a `> Host: api.ollama.com` if the wrapper is proxying. The response header will include `x-ollama-proxy: true` (or similar). The body will contain the subscription error if you’re not authorized.

---

## 7. Deep dive: How the wrapper decides to route  

Understanding *why* the wrapper forwards certain models helps you prevent future surprises.

### 7.1. Model metadata table  

Ollama stores a small JSON file for each model under `~/.ollama/models/<model-id>/metadata.json`. Example for a subscription‑only model:

```json
{
  "name": "llama2-120b",
  "size_bytes": 210000000000,
  "source": "ollama.com",
  "requires_subscription": true,
  "routing": {
    "default": "remote",
    "local_fallback": false
  }
}
```

The `routing.default` field tells the wrapper to **always** forward to the remote SaaS endpoint. If you manually edit this file to `"default":"local"` and restart the wrapper, the model will be forced to run locally (provided you have the binary).

### 7.2. Environment‑driven overrides  

There *is* a variable that can influence routing: `OLLAMA_FORCE_LOCAL`. Setting it to `1` tells the wrapper to ignore the `requires_subscription` flag and attempt local inference. Example:

```bash
export OLLAMA_FORCE_LOCAL=1
systemctl restart ollama   # or `ollama serve &`
```

But note: if the binary is missing, the wrapper will still error out with “model not found locally”. It won’t silently fall back to remote.

### 7.3. The “receptionist” algorithm (pseudo‑code)

```go
func routeRequest(req Request) Response {
    meta := loadModelMeta(req.Model)
    if meta.RequiresSubscription && !hasValidLicense(req.Auth) {
        // forward to cloud, let cloud decide
        return proxyToRemote(req)
    }
    if env.ForceLocal && localBinaryExists(meta) {
        return runLocally(req)
    }
    // default: remote for large models, local for small ones
    if meta.SizeBytes > 50_000_000_000 { // > 50 B
        return proxyToRemote(req)
    }
    return runLocally(req)
}
```

The logic is **deterministic**; you just need to know which branch you’re hitting. That’s why inspecting the metadata file and the environment is crucial.

---

## 8. Real‑world scenario: A production CI pipeline went rogue  

Below is a condensed timeline from a recent incident at a fintech startup. The steps mirror the checklist above, and the numbers illustrate the cost impact.

| Time (UTC) | Action | Observation |
|------------|--------|-------------|
| 09:12:03 | CI job runs `ollama run llama2-120b` | GPU utilization reported 0 % by `nvidia-smi`. |
| 09:12:05 | `curl -s http://localhost:11434/api/chat` returns a 150‑token answer in **0.71 s**. | Immediate red flag. |
| 09:12:07 | `netstat -tulpn` shows outbound TCP to `34.120.45.67:443`. | Wrapper contacting external host. |
| 09:12:09 | `ps -ef | grep ollama` reveals `ollama serve --port 11434`. | Only wrapper process visible. |
| 09:12:12 | `curl -v http://localhost:11433/api/generate` → **380 s** cold load. | Confirms local binary exists but is not used. |
| 09:12:15 | Log entry: `ResponseError: this model requires a subscription`. | Remote SaaS rejecting unauthenticated request. |
| 09:12:20 | Edit `metadata.json` to `"routing":{"default":"local"}` and set `OLLAMA_FORCE_LOCAL=1`. | Restarted wrapper. |
| 09:12:35 | New request on 11434 now takes **380 s** (local load) and GPU usage spikes to **95 %**. | Issue resolved. |
| 09:13:00 | Added checklist to CI pipeline (snapshot → raw‑port test → verbose curl). | Prevents regression. |

The **cost** of the hidden remote call was twofold:

1. **Financial** – each remote token cost $0.0002; 150 tokens = $0.03 per request, multiplied by 10 000 CI runs = **$300** wasted.
2. **Compliance** – data that should have stayed on‑premise was sent to a third‑party API, violating the company’s data‑sovereignty policy.

---

## 9. Extending the diagnostic to other platforms  

While this article focuses on Ollama, the same principles apply to any **API proxy** that sits in front of a model server.

| Platform | Typical local port | Proxy port | Common routing flag |
|----------|-------------------|-----------|---------------------|
| **LMStudio** | `127.0.0.1:12345` (gRPC) | `127.0.0.1:12346` (REST) | `allow_remote: true` in `config.yaml` |
| **vLLM‑Gateway** | `0.0.0.0:8000` (vLLM) | `0.0.0.0:8080` (gateway) | `gateway.remote_models` list |
| **SageMaker Inference** | N/A (managed) | N/A (API) | `ModelDataUrl` points to S3 vs. `ContainerImage` |

The three‑rule checklist remains identical: **snapshot**, **bypass**, **inspect**. The only variation is the exact commands to locate the raw service (e.g., `grpcurl` instead of `curl`).

---

## 10. Building a “router‑aware” client library  

If you frequently need to guarantee local inference, you can embed the diagnostic logic into a thin wrapper library. Below is a **Python** example that automatically falls back to the raw port if the public endpoint appears too fast.

```python
import time
import requests
import os
import subprocess

PUBLIC_URL = "http://localhost:11434/api/chat"
RAW_URL     = "http://localhost:11433/api/generate"
MAX_LOCAL_LATENCY = 5.0   # seconds, generous warm‑load threshold

def is_local_fast():
    start = time.time()
    try:
        r = requests.post(PUBLIC_URL,
                          json={"model":"llama2-120b",
                                "messages":[{"role":"user","content":"ping"}]},
                          timeout=2)
    except requests.exceptions.Timeout:
        return False
    elapsed = time.time() - start
    return elapsed < MAX_LOCAL_LATENCY

def call_model(prompt: str) -> str:
    if is_local_fast():
        # Proxy is likely remote – use raw port
        payload = {"model":"llama2-120b","prompt":prompt}
        r = requests.post(RAW_URL, json=payload)
    else:
        # Assume local path
        payload = {"model":"llama2-120b",
                   "messages":[{"role":"user","content":prompt}]}
        r = requests.post(PUBLIC_URL, json=payload)
    r.raise_for_status()
    return r.json()["response"]

if __name__ == "__main__":
    print(call_model("Explain why the sky is blue."))
```

The function `is_local_fast` performs a **micro‑benchmark** on the public endpoint. If the response is *too* fast, the client assumes the request was proxied and switches to the raw endpoint. This pattern can be packaged as a small **pip** module and shared across teams to avoid the hidden‑router pitfall.

---

## 11. Preventive configuration – make the router transparent  

You can configure Ollama (and similar tools) to **never** forward to the cloud unless you explicitly ask for it.

### 11.1. Disable remote routing globally  

Add the following to `~/.ollama/config.yaml`:

```yaml
routing:
  allow_remote: false   # block any automatic remote fallback
  force_local: true     # prefer local binaries when available
```

Restart the service:

```bash
systemctl restart ollama   # or `pkill ollama && ollama serve &`
```

Now any request that references a model without a local binary will return:

```json
{
  "error": "model not found locally and remote routing disabled"
}
```

### 11.2. Per‑model opt‑out  

If you need a few large models to stay remote (e.g., for cost reasons), edit their `metadata.json`:

```json
{
  "routing": {
    "default": "remote"
  }
}
```

All other models will default to local execution. This **whitelisting** approach is safer than a global “allow remote” flag.

### 11.3. Auditing with a one‑liner  

```bash
grep -R '"default":' ~/.ollama/models/*/metadata.json | \
  awk -F: '{print $1}' | uniq -c
```

The output shows how many models are set to `"remote"` vs. `"local"`. Run this as part of your CI lint step to ensure you haven’t unintentionally introduced a remote‑only model.

---

## 12. Network‑level hardening – firewalls and egress rules  

Even with the wrapper configured correctly, a mis‑configured network can silently **allow** outbound traffic.

| Layer | Defensive measure | Example command |
|-------|-------------------|-----------------|
| **Host firewall** | Block outbound 443 to `api.ollama.com` | `sudo ufw deny out to 34.120.45.67 port 443` |
| **Container runtime** | Drop egress in Docker bridge network | `docker network create --internal isolated_net` |
| **K8s pod security** | `egress: []` in a NetworkPolicy | ```yaml\napiVersion: networking.k8s.io/v1\nkind: NetworkPolicy\nmetadata:\n  name: deny-ollama-egress\nspec:\n  podSelector:\n    matchLabels:\n      app: ollama\n  policyTypes:\n  - Egress\n``` |

If the wrapper attempts to reach the cloud, the connection will be **refused**, and you’ll see an error like `dial tcp 34.120.45.67:443: connect: connection refused`. That’s a clear, deterministic signal that the routing is being blocked, forcing you to fix the configuration rather than silently falling back.

---

## 13. Performance budgeting – when to accept remote routing  

Sometimes remote routing is intentional (e.g., you want to offload a 200 B model you cannot host). In those cases you should **budget** for the latency and cost.

| Metric | Recommended limit | How to enforce |
|--------|-------------------|----------------|
| **Max per‑request latency** | 2 s for 50‑token replies | Use `OLLAMA_MAX_TIMEOUT=2000` (ms) in the wrapper |
| **Monthly token budget** | 10 M tokens | Set `OLLAMA_TOKEN_QUOTA=10000000` and monitor via `/metrics` |
| **Allowed remote models** | Whitelist only `gpt-4o-mini` | Edit `metadata.json` for each allowed model, set `"routing":{"default":"remote"}` |

By codifying these limits, you avoid surprise bills and keep the system’s behavior predictable.

---

## 14. TL;DR checklist for senior engineers  

| ✅ | Action | Command / Config |
|----|--------|------------------|
| 1 | Capture process, port, and network state | `ps -ef | grep ollama`<br>`netstat -tulpn | grep 1143[4]`<br>`lsof -i :11434` |
| 2 | Verify local load time on raw port | `time curl -X POST http://localhost:11433/api/generate …` |
| 3 | Inspect HTTP exchange verbosely | `curl -v -X POST http://localhost:11434/api/chat …` |
| 4 | Look for subscription errors in logs | `journalctl -u ollama -f | grep ResponseError` |
| 5 | Force local execution (if desired) | `export OLLAMA_FORCE_LOCAL=1 && systemctl restart ollama` |
| 6 | Harden egress to prevent silent fallback | `ufw deny out to api.ollama.com port 443` |
| 7 | Add a CI step that runs the three‑rule checklist on every PR | (script example below) |

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Snapshot
echo "=== PROCESS SNAPSHOT ==="
ps -ef | grep ollama || true
echo "=== PORT SNAPSHOT ==="
netstat -tulpn | grep :1143[4] || true
echo "=== CONNECTIONS ==="
lsof -i :11434 | grep ESTABLISHED || true

# 2. Raw port latency
echo "=== RAW PORT LATENCY ==="
time curl -s -X POST http://localhost:11433/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2-120b","prompt":"test"}' > /dev/null

# 3. Verbose HTTP
echo "=== VERBOSE HTTP TO PUBLIC PORT ==="
curl -v -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2-120b","messages":[{"role":"user","content":"test"}]}' \
  || true
```

Add this script to your repo’s `ci/diagnostics.sh` and run it as a **pre‑merge gate**. If any step exceeds a threshold (e.g., raw‑port latency > 30 s for a warm model), the CI job fails, forcing the developer to investigate.

---

## 15. Takeaways for the seasoned engineer  

- **Never trust the URL**. “localhost” only guarantees the TCP connection terminates on the same host; it says nothing about *what* processes sit behind that socket.
- **Latency is a truth‑telling metric**. When a 120 B model answers in sub‑second time, the only plausible explanation is remote inference.
- **Metadata drives routing**. The tiny `metadata.json` file decides whether a model is local‑only, remote‑only, or hybrid. Editing it (or the environment variables that override it) is the cleanest way to control behavior.
- **Three‑rule checklist** is cheap, repeatable, and works across platforms. Embed it in your observability stack and you’ll catch hidden proxies before they bite.
- **Network egress controls** are a second line of defense. Even a mis‑configured wrapper can’t cheat a firewall that blocks outbound traffic to the SaaS domain.
- **Instrument your CI/CD** with the diagnostic script. Treat hidden routing as a test failure, not a mystery to be debugged after the fact.

By internalizing these principles, you’ll turn the “receptionist” from a silent gatekeeper into a transparent conduit you can audit, configure, and, when needed, bypass. Your LLM workloads will stay where you expect them—on‑prem, on‑GPU, and under your full control.