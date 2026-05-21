# Reading the Source: Diagnostics When Configs Are Opaque  

You’ve probably spent a few sleepless nights staring at a log line that says *“unexpected status 500 from service‑X”* while the Swagger UI for that service is a single‑page PDF that was last updated in 2019. The reality is that **most production services are opaque**: they run code you didn’t write, you can’t see the source, and the documentation is either stale or missing altogether. When the behavior you encounter isn’t covered by the docs, the fastest way to unblock yourself is to **read the source that’s actually executing**. Think of it as lifting the hood on a car you didn’t build—not to rebuild the engine, but to locate the one loose wire that’s causing the misfire.

In this article you will:

1. Understand why “opaque service” problems are inevitable in modern microservice stacks.  
2. Acquire a **minimum skill set** that lets you pull the source from a running container, locate the routing/decision logic, and trace a single request from entry point to side effect.  
3. Learn what you **don’t need** to succeed—no full‑code‑base comprehension, no refactoring, no test‑suite authoring.  
4. Walk through a **real‑world worked example** where 30 lines of Python solved a two‑hour network‑debugging mystery.  
5. Get a cheat‑sheet for **closed‑source binaries** (strings, ltrace, strace, packet capture).  
6. Adopt a **meta‑skill**: assume every service has at least one undocumented behavior and treat source as the ultimate source of truth.  
7. Internalize the **moral**: code wins over docs, and you can train yourself to read unfamiliar languages just enough to answer the question at hand.

---

## 1. The Opaque Service Problem

### 1.1 Why “opaque” is the default, not the exception

Modern production environments are built on **layers of abstraction**:

| Layer | Typical Owner | Visibility to You |
|-------|----------------|-------------------|
| Cloud provider (e.g., AWS, GCP) | Platform team | API surface, SLA |
| Managed services (RDS, ElasticSearch) | Platform team | Config UI, limited metrics |
| Internal shared libraries (auth SDK, logging) | Platform team | Binary wheels, version pin |
| Business microservices (recommendation, billing) | Product teams | Docker images, Helm charts |
| Third‑party SaaS (payment gateway, feature flag) | Vendor | HTTP endpoints, docs |

Each layer adds a **boundary** that you can cross with configuration, but you cannot see the implementation behind it. The boundary is transparent (you can send a request and get a response), but the **inner wiring** is hidden. That’s the *opaque service*.

### 1.2 Analogy: The Plumbing Analogy

Imagine a house where you inherited a bathroom that works fine most of the time. One day the faucet drips, but the homeowner left you only a schematic that shows a “cold water line” and a “hot water line” without any pipe IDs. You could:

* **Guess** which valve is stuck (turn the main off, wait, try again).  
* **Replace the whole faucet** (costly, risky).  

Or you could **open the wall**, locate the pipe that actually supplies the faucet, and tighten the single nut that’s loose. The wall is the opaque boundary; the pipe is the source code. Opening the wall is cheap, precise, and resolves the problem without unnecessary overhaul.

### 1.3 Analogy: The Kitchen Analogy

Think of a restaurant kitchen that runs a “sauce station” you never saw being built. The menu says “spicy aioli” but the taste is sometimes sweet. You could:

* **Add more chili** (blindly adjusting the recipe).  
* **Rewrite the entire sauce** (requires a chef, time, and risk).  

Instead, you **lift the lid on the sauce pot**, skim the surface, and spot a single spoonful of honey that was added for a “special occasion”. Removing that spoonful fixes the flavor. The sauce pot is the running container; the honey is the hidden branch in the code.

---

## 2. Minimum Skill Set: From Container to Control Flow

You don’t need to become a full‑stack guru overnight. All you need is a **three‑step toolkit** that lets you:

1. **Enter the runtime environment** (container, pod, VM).  
2. **Locate the entry point and decision logic** (routing tables, conditionals).  
3. **Follow a single request** from the first line of code to the side effect (DB write, external API call).

### 2.1 Step 1 – Getting Inside the Running Image

Most services run in Docker containers orchestrated by Kubernetes, Nomad, or ECS. The most common entry point is a shell inside the container.

```bash
# 1️⃣ Find the pod/container name (Kubernetes)
kubectl get pods -n prod -l app=order-service

# Example output
# order-service-7c9d8f5c9-kw9x2   1/1   Running   0          12h

# 2️⃣ Exec into the container
kubectl exec -it order-service-7c9d8f5c9-kw9x2 -n prod -- bash

# If you’re on a plain Docker host
docker ps | grep order-service
docker exec -it <container-id> bash
```

Once inside, you have the same filesystem the process sees. The **source code is often mounted under `/app`, `/src`, or `/opt/service`**. A quick `ls` will reveal the layout:

```bash
root@order-service:/# ls -R /app | head -n 20
app/
app/__init__.py
app/main.py
app/views/
app/views/__init__.py
app/views/orders.py
app/models/
app/models/__init__.py
app/models/order.py
...
```

If the container only contains compiled bytecode (`*.pyc`) or a single binary, you can still inspect the *decompiled* version (see Section 5).

### 2.2 Step 2 – Spotting the Routing / Decision Code

Most web frameworks expose a **router** that maps HTTP verbs and paths to handler functions. The patterns you’ll look for differ by language, but the idea is the same:

| Language / Framework | Typical Router Pattern |
|----------------------|------------------------|
| Python / Flask       | `@app.route('/orders', methods=['POST'])` |
| Python / FastAPI     | `@router.post("/orders")` |
| Node.js / Express    | `app.post('/orders', handler)` |
| Go / Gin             | `router.POST("/orders", handler)` |
| Java / Spring Boot   | `@PostMapping("/orders")` |
| Ruby / Rails         | `post 'orders', to: 'orders#create'` |

**Search for the keyword** that defines the route you’re interested in. In Bash you can use `grep -R`:

```bash
# Find any route that mentions "/orders"
grep -R "orders" /app | grep -E "route|@router|@PostMapping"
```

Typical output:

```
/app/main.py: @app.post("/orders")
/app/views/orders.py: def create_order():
```

Open the file that contains the handler:

```bash
cat /app/views/orders.py | nl | sed -n '1,120p'
```

You’ll see something like:

```python
1  from fastapi import APIRouter, Request
2  from .models import Order, OrderStatus
3  from .services import payment_gateway, notification
4
5  router = APIRouter()
6
7  @router.post("/orders")
8  async def create_order(request: Request):
9      payload = await request.json()
10     # Decision point #1
11     if payload.get("model") in PREVIEW_MODELS:
12         # Hidden branch
13         response = await payment_gateway.preview_charge(payload)
14     else:
15         response = await payment_gateway.charge(payload)
16
17     # Decision point #2
18     if response.success:
19         order = Order(**payload, status=OrderStatus.CONFIRMED)
20         await order.save()
21         await notification.send_success(order)
22     else:
23         await notification.send_failure(payload, response.error)
24
25     return {"order_id": order.id if response.success else None}
```

Lines 11‑15 are a **conditional branch** that decides which payment gateway to hit. That’s the kind of logic that often diverges from the public docs.

### 2.3 Step 3 – Tracing ONE Request End‑to‑End

Now you have the entry point (`create_order`). To see the **full control flow** for a single request, you can:

1. **Add a temporary `print` or `logger.debug`** (if the container runs in a dev‑friendly mode).  
2. **Use `strace`/`ltrace`** to watch system/library calls at runtime.  
3. **Inject a request with `curl`** and watch the logs.

#### 2.3.1 Quick `print` injection

```bash
# Open the file in a minimal editor (vi, nano)
vi /app/views/orders.py
# Insert a debug line after line 11
print(">>> DEBUG: model=%s, preview=%s" % (payload.get("model"), payload.get("model") in PREVIEW_MODELS))
```

Re‑load the service (many containers auto‑reload on file change; otherwise restart):

```bash
# If the container runs a supervisor like gunicorn
pkill -HUP gunicorn   # send HUP to reload
# Or simply restart the pod
kubectl rollout restart deployment/order-service -n prod
```

Now fire a request:

```bash
curl -X POST http://order-service.prod.svc.cluster.local/orders \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt‑4‑preview","amount":100}'
```

You’ll see the debug line in the pod logs:

```bash
kubectl logs -f pod/order-service-7c9d8f5c9-kw9x2 -n prod | grep DEBUG
>>> DEBUG: model=gpt‑4‑preview, preview=True
```

That tells you the request went down the **preview** branch. If the docs said “all orders use the production gateway”, you now know there’s a hidden **preview mode**.

#### 2.3.2 Using `strace` for a non‑intrusive view

If you cannot modify the code (e.g., the container is immutable), attach `strace` to the running process:

```bash
# Find the PID of the Python worker (assuming gunicorn)
ps aux | grep gunicorn
# Example PID = 42
strace -p 42 -e trace=network,write -s 128 -ff -o /tmp/trace.out &
```

The `-e trace=network,write` flag limits output to network syscalls (`connect`, `sendto`, `recvfrom`) and file writes (useful for seeing which external endpoint is hit). After the request, inspect the trace:

```bash
grep -i "connect" /tmp/trace.out.*
# Example output
[pid 42] connect(3, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("34.195.252.100")}, 16) = 0
```

You can map the IP to a hostname:

```bash
dig -x 34.195.252.100 +short
# Output: api.preview.ollama.com.
```

Now you know the request went to *Ollama Cloud preview endpoint*—a detail not in the public API docs.

---

## 3. What You **Don’t** Need to Succeed

It’s easy to feel compelled to **understand the entire codebase**, **refactor the hidden branch**, or **write a full test suite** before you can answer the question. That’s a classic “analysis paralysis” trap. In practice you only need:

| Goal | Minimal Requirement | Why It’s Sufficient |
|------|---------------------|---------------------|
| Identify why a request fails | Locate the conditional that selects the external service and see the branch taken | The failure is usually caused by a mis‑routed call or a missing env var |
| Verify which environment variable controls a feature flag | Find the `os.getenv` call that reads the flag and inspect its value at runtime | Feature toggles are typically read once per request |
| Confirm which database table a write goes to | Follow the ORM `save()` call to the model definition | The ORM generates the SQL; the model tells you the table name |

You **don’t** need to:

* Load the entire repository into an IDE.  
* Understand every import chain.  
* Write unit tests for the hidden branch (unless you plan to ship a fix).  

The **minimum viable investigation** is a *single request trace* that yields the answer you need.

---

## 4. Worked Example: A Thin Proxy with a Secret Ollama Call

### 4.1 The Situation

Your team maintains a **text‑generation wrapper** called `gen-proxy`. It forwards requests to an internal LLM service, but the documentation says it only talks to `http://llm.internal:8080/v1/chat`. Users report that for certain payloads the response latency spikes to **12 seconds** and the output is sometimes a **different model** than requested.

You spend **2 hours**:

* Capturing traffic with `tcpdump`.  
* Comparing request/response bodies.  
* Looking at the Helm chart values (nothing unusual).  

Nothing explains the latency or the model swap.

### 4.2 The Quick Source Dive

You exec into the pod:

```bash
kubectl exec -it deploy/gen-proxy -n prod -- bash
cd /app
ls -R | grep -E "views|router|handler"
```

You find `views.py` with only **78 lines**. Open it:

```bash
cat views.py | nl | sed -n '1,120p'
```

Result:

```python
 1  import os
 2  import httpx
 3  from fastapi import APIRouter, Request
 4
 5  router = APIRouter()
 6
 7  PREVIEW_MODELS = {"gpt-4-preview", "llama-2-preview"}
 8  OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://llm.internal:8080/v1/chat")
 9
10  @router.post("/chat")
11  async def chat(request: Request):
12      payload = await request.json()
13      model = payload.get("model")
14
15      # *** Decision point ***
16      if model in PREVIEW_MODELS:
17          # Hidden branch: use Ollama Cloud
18          endpoint = os.getenv("OLLAMA_CLOUD_URL", "https://api.ollama.cloud/v1/chat")
19          api_key = os.getenv("OLLAMA_API_KEY")
20          async with httpx.AsyncClient() as client:
21              resp = await client.post(
22                  endpoint,
23                  json=payload,
24                  headers={"Authorization": f"Bearer {api_key}"}
25              )
26          return resp.json()
27
28      # Default path: internal LLM
29      async with httpx.AsyncClient() as client:
30          resp = await client.post(OLLAMA_ENDPOINT, json=payload)
31      return resp.json()
```

**Key observations:**

* Lines 16‑26 are a **hidden branch** that routes preview models to an external Ollama Cloud endpoint.  
* The branch adds **network latency** (HTTPS, external DNS).  
* The external call uses an **API key** that may be rate‑limited, causing the 12 s spikes.

### 4.3 Verifying the Hypothesis

Inject a request that triggers the preview branch:

```bash
curl -X POST http://gen-proxy.prod.svc.cluster.local/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4-preview","messages":[{"role":"user","content":"Hello"}]}'
```

Check the pod logs (the service logs the endpoint used):

```bash
kubectl logs -f pod/gen-proxy-6c9b7d9c9-8xk2q -n prod | grep "endpoint"
# Example output
2024-05-20T14:32:01Z INFO endpoint=https://api.ollama.cloud/v1/chat
```

Now send a non‑preview model:

```bash
curl -X POST http://gen-proxy.prod.svc.cluster.local/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}'
```

Log shows:

```bash
2024-05-20T14:33:12Z INFO endpoint=http://llm.internal:8080/v1/chat
```

**Result:** The mysterious latency and model swap are explained by the hidden preview branch. The fix is either:

* Remove `gpt-4-preview` from the client payload, or  
* Set `OLLAMA_ENDPOINT` to the cloud URL for all models (if that’s the desired behavior).  

All of this took **30 lines of code** and a few `grep` commands—far less than the two hours spent on network sniffing.

---

## 5. When Source Is Closed: Binary‑Level Diagnostics

Sometimes you truly have no source: a proprietary binary, a compiled Go plugin, or a closed‑source SaaS sidecar. You can still extract *behavioural* clues by treating the binary as a **black box with a transparent boundary**.

### 5.1 `strings` – Find Embedded Literals

```bash
# Locate the binary (e.g., /usr/local/bin/agent)
strings /usr/local/bin/agent | grep -i "api" | head -n 20
```

Typical output:

```
https://api.vendorcloud.com/v2/ingest
VENDOR_API_KEY
preview_mode
```

If you see a URL, you now know the external endpoint.

### 5.2 `ltrace` – Trace Library Calls

`ltrace` shows calls to shared libraries (e.g., `libcurl`, `libssl`). Install it (`apt-get install ltrace`) and run:

```bash
ltrace -p <pid> -e curl_easy_perform,curl_easy_setopt 2>&1 | grep -i "https"
```

Example:

```
curl_easy_setopt(0x7f9c3c0000, CURLOPT_URL, "https://api.vendorcloud.com/v2/ingest") = 0
curl_easy_perform(0x7f9c3c0000) = 0
```

You’ve identified the **exact function** that makes the HTTP request.

### 5.3 `strace` – System Call Level

If you need to see **file access** or **environment variable reads**, `strace` is the tool:

```bash
strace -e trace=openat,read,write,execve -p <pid> -s 256 2>&1 | grep -i "VENDOR_API_KEY"
```

Result:

```
openat(AT_FDCWD, "/etc/vendor.conf", O_RDONLY) = 3
read(3, "api_key=ABCD1234\npreview_mode=true\n", 256) = 45
```

You now know the binary reads its config from `/etc/vendor.conf`.

### 5.4 Packet Capture – `tcpdump` or `wireshark`

When you suspect network activity but can’t see the URL, capture the traffic:

```bash
tcpdump -i eth0 -nn -s 0 -w /tmp/agent.pcap host vendorcloud.com and port 443 &
# Trigger the behavior (e.g., send a request to the service)
curl -X POST http://myservice.local/do_something -d '{"foo":"bar"}'
# Stop capture after a few seconds
kill %1
```

Open the pcap in Wireshark, filter `tls.handshake.type == 1` (Client Hello) and look for the **Server Name Indication (SNI)** field, which reveals the remote hostname even if the payload is encrypted.

### 5.5 Quick Cheat‑Sheet Table

| Tool | What it reveals | Typical command | When to use |
|------|-----------------|----------------|------------|
| `strings` | Embedded literals, URLs, keys | `strings binary | grep -i "http"` | First pass, binary has no stripping |
| `ltrace` | Library calls (curl, openssl) | `ltrace -p <pid> -e curl*` | You know the binary uses a known lib |
| `strace` | Syscalls, env var reads, file I/O | `strace -e trace=open,read -p <pid>` | Need to see config loading |
| `tcpdump`/`wireshark` | Network endpoints, protocols | `tcpdump -i eth0 -w out.pcap` | Binary encrypts payload, you need SNI |

These tools let you **map the opaque boundary** without ever seeing the original source.

---

## 6. The Meta‑Skill: Assume Undocumented Behavior Exists

When you step onto a new service, **don’t start with “the docs are correct.”** Instead, adopt the mental model:

> *Every service in my stack has at least one undocumented behavior that will bite me today.*

### 6.1 A Systematic Checklist

| Step | Question | Typical Artifact |
|------|----------|------------------|
| 1️⃣ Identify the request you care about | What HTTP method, path, and payload are involved? | `curl -v …` |
| 2️⃣ Find the routing entry | Which file defines the handler? (`grep route`) | `views.py` |
| 3️⃣ Locate conditionals that depend on input | `if … in …` , `if os.getenv` | Lines 10‑20 |
| 4️⃣ Observe runtime values | What does the env var actually contain? (`print(os.getenv)`) | Logs |
| 5️⃣ Verify side‑effects | Which external endpoint or DB is hit? (`strace`, `ltrace`) | Syscall trace |
| 6️⃣ Confirm with a minimal test | Does flipping the input change the branch? (`curl` with variant) | Success/failure |

You can run through this checklist in **under 10 minutes** for most services that expose source in the container.

### 6.2 Example: “Feature Flag” Gotchas

A service claims “Feature X is disabled in production”. You check the Helm values and see `featureX.enabled: false`. Yet the feature appears for a subset of users. Running the checklist:

1. **Locate** the flag read: `os.getenv("FEATURE_X")` in `feature.py`.  
2. **Print** its value at runtime: `print("FEATURE_X:", os.getenv("FEATURE_X"))`.  
3. **Observe** that the env var is **unset**, so the code falls back to `default=True`.  

The hidden default explains the surprise. The fix: add `FEATURE_X=0` to the deployment env.

---

## 7. The Moral: Code Beats Docs, and You Can Read It

When documentation and code disagree, **code wins**. The code is the *source of truth*; the docs are a *snapshot* that may be stale. By training yourself to **read just enough source**—even in a language you’re not fluent in—you gain a decisive advantage:

* **Speed** – You resolve a two‑hour mystery in minutes.  
* **Confidence** – You know exactly which branch is executed.  
* **Safety** – You avoid speculative config changes that could break other flows.  

### 7.1 Learning to Read Unfamiliar Languages

You don’t need full fluency. Focus on **syntax patterns** that are language‑agnostic:

| Concept | Python | Go | JavaScript | Java |
|---------|--------|----|------------|------|
| Import | `import module` | `import "pkg"` | `const x = require('x')` | `import com.foo.Bar;` |
| Function definition | `def foo(...):` | `func foo(...) {}` | `function foo(...) {}` | `public void foo(...) {}` |
| Conditional | `if x in Y:` | `if _, ok := Y[x]; ok {}` | `if (Y.includes(x)) {}` | `if (Y.contains(x)) {}` |
| Environment var | `os.getenv("VAR")` | `os.Getenv("VAR")` | `process.env.VAR` | `System.getenv("VAR")` |

When you see a line that matches any of these patterns, you already know its **semantic role**. Combine that with a quick `grep` for the keyword you care about, and you can follow the flow without needing to understand the whole language.

### 7.2 A Final Thought Experiment

Imagine you’re a **plumber** called to fix a leak in a house you never built. The homeowner hands you a **blueprint** that shows a pipe from the kitchen to the bathroom, but the actual leak is at a hidden junction behind the wall. You could:

* **Study the blueprint for hours** (analogous to reading the whole repo).  
* **Replace the entire kitchen plumbing** (refactor the whole service).  

Or you could **remove the wall panel, locate the joint, tighten the nut, and close the wall**—the minimal, targeted fix. That’s exactly what reading the source in a running container does for you. The source is the *joint*; the docs are the *blueprint*.

---

## 8. Putting It All Together – A Mini‑Playbook

| Phase | Action | Command / Snippet | Expected Outcome |
|-------|--------|-------------------|------------------|
| **Enter** | `kubectl exec -it … bash` | `docker exec -it <id> bash` | Shell inside the container |
| **Locate** | `grep -R "/orders" /app` | `grep -R "router.post" /app` | Path to handler file |
| **Identify Decision** | Search for `if … in` | `grep -n "if .* in" views.py` | Line numbers of conditionals |
| **Inspect Runtime** | Add `print` or use `strace` | `strace -p <pid> -e trace=network` | See which endpoint is hit |
| **Validate** | Send two variant requests | `curl … -d '{"model":"preview"}'` and `curl … -d '{"model":"prod"}'` | Different branches observed |
| **Document** | Write a one‑sentence note in the ticket | `echo "Preview models go to Ollama Cloud (line 16‑26)"` | Knowledge captured for future |

Follow this playbook whenever you encounter an **opaque service**. You’ll spend **minutes** where you previously spent **hours**.

---

## 9. Real‑World Tips & Gotchas

| Tip | Why It Matters |
|-----|----------------|
| **Use `nl` (numbered `cat`)** when reading a file. It lets you reference line numbers in conversation (`line 16`). | Reduces back‑and‑forth with teammates. |
| **Avoid `sudo` inside the container** unless the process runs as root. Many containers drop privileges; `sudo` may not exist. | Keeps the environment faithful to production. |
| **If the container uses a read‑only filesystem**, copy the file to `/tmp` before editing (`cp /app/views.py /tmp/views.py`). | Prevents “read‑only file system” errors. |
| **When using `strace`/`ltrace`, filter by PID** to avoid noise from other processes in the pod. | Keeps output manageable. |
| **Set `PYTHONUNBUFFERED=1`** in the container to get immediate `print` output in logs. | Makes debugging interactive. |
| **If the binary is stripped**, `strings` may still reveal URLs; otherwise, use `objdump -d` for disassembly. | Gives you a fallback when `strings` is empty. |
| **Never trust a Helm value file blindly**; environment variables can be overridden at the pod level (`envFrom`, `configMap`). | Prevents chasing a phantom flag. |

---

## 10. A Quick Reference Cheat Sheet (One Page)

```
# 1. Enter container
kubectl exec -it <pod> -n <ns> -- bash

# 2. Find source root
cd /app || cd /src || cd /opt/service

# 3. Locate handler
grep -R "router.post" . | head

# 4. Open with line numbers
nl -ba path/to/file.py | less

# 5. Spot conditionals
grep -n "if .* in" path/to/file.py

# 6. Inject debug (if possible)
echo 'print("DEBUG:", var)' >> path/to/file.py

# 7. Reload (HUP or rollout restart)
pkill -HUP gunicorn
kubectl rollout restart deployment/<svc> -n <ns>

# 8. Trace network call (no code change)
strace -p <pid> -e trace=network -s 128 -ff -o /tmp/trace.out &
curl -X POST http://svc/path -d '{"model":"preview"}'
grep -i "connect" /tmp/trace.out.*

# 9. Binary analysis (closed source)
strings /usr/local/bin/agent | grep -i "api"
ltrace -p <pid> -e curl_easy_* 2>&1 | grep -i "https"
tcpdump -i eth0 -nn -s 0 -w /tmp/agent.pcap host vendorcloud.com and port 443 &
```

Keep this sheet bookmarked in your terminal. When the next opaque service bites, you’ll have a **ready‑to‑run** set of commands that turn mystery into a **single, traceable line of code**.

---

## 11. Closing the Loop – From Mystery to Mastery

You’ve now walked through the entire diagnostic workflow:

* **Lift the hood** on the container.  
* **Find the wiring diagram** (router → handler).  
* **Spot the loose wire** (conditional branch).  
* **Measure the voltage** (runtime values via `print`, `strace`).  
* **Replace the wire** (adjust config, env var, or payload).  

Whether the service is open‑source Python, compiled Go, or a closed‑source binary, the same principles apply: **the code you can see (or the system calls you can observe) is the only reliable map**. When the map disagrees with the guidebook (docs), trust the map.

You now have a repeatable, low‑overhead method to answer the **single question** that matters for any incident. The next time you see “opaque service” in a ticket, you’ll know exactly where to look, what to look for, and how to prove it—all without rewriting the whole system.