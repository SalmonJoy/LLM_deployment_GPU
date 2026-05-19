# GPU-vs-CPU Placement Guide — Gravity LLM Stack

A reference for deciding where new services, models, and containers belong
in your fleet: the H100 GPU box, or a regular Docker host.

Anchored by the current Gravity stack as the worked example, but the rules
in §6, §7, §8 and §11 are written to apply to anything you add later.

> Companion doc: [azure-llm-setup.md](azure-llm-setup.md) — point-in-time
> snapshot of the H100 host (`GPU-1`). Read that for *what is*. Read this
> for *what should be*.

---

## 1. How to use this doc

When you are about to deploy or design any new component (a model, a
service, a worker, a database), open this and answer in order:

1. §6 — does it need a GPU? (hard rule)
2. §7 — what's the placement decision flow?
3. §10 — does the sizing fit?
4. §9 — am I about to repeat a known anti-pattern?

You should reach a placement answer in under two minutes.

---

## 2. TL;DR — one-glance decision card

| If the new thing is… | Put it on |
|---|---|
| An LLM inference engine (Ollama, vLLM, TGI, llama.cpp) | **GPU host** |
| A model fine-tuning / training job | **GPU host** (and reserve a VRAM window) |
| A Python service that imports `torch` *with CUDA* / `bitsandbytes` / `flash-attn` | **GPU host** |
| A small embedding model running on CPU (e.g. `nomic-embed-text-v1.5`) | **CPU host** |
| A vector store (Qdrant, Weaviate, Milvus, Chroma) | **CPU host** with SSDs |
| A SQL/NoSQL database (Postgres, Mongo, Redis) | **CPU host** |
| A Django/FastAPI/Node API that *calls* an LLM via HTTP | **CPU host** |
| A web frontend, reverse proxy, auth gateway | **CPU host** |
| A scheduler, queue worker, batch ingestion job | **CPU host** |
| Monitoring / observability (Prometheus, Grafana, Loki) | **CPU host** |

Default: **assume CPU host until you can prove the GPU is needed**. The H100
box costs roughly 10× a comparable CPU VM per hour — every CPU service you
park on it is wasted money and shared blast radius.

---

## 3. Component catalog (current stack)

Each row is a placement profile. Use this as a template when adding new
components — fill in the same columns and apply §6.

| Component | Role | Stateful? | GPU? | RAM (working) | CPU | Disk (state) | Latency-sensitive to LLM? | Correct host |
|---|---|---|---|---|---|---|---|---|
| `ollama` (container) | LLM inference daemon | Yes (model files) | **Yes — required** | 1–4 GB host RAM + VRAM = sum of resident models + KV cache | Light during inference | Volume `ollama` (175 GB now, models swap in/out) | N/A — it *is* the GPU consumer | **GPU host** |
| `qdrant` (container) | Vector store, HTTP API on `:6333` | Yes (collections) | No | 1 GB + RAM ≈ vectors × dim × 4 bytes | Light–moderate during search | Bind `/root/qdrant_storage` (~12 GB now) | No — embeddings precomputed | **CPU host** (currently mis-placed) |
| `ollama_wrapper` (Django) | **Hybrid local/cloud router** — proxies some models to local Ollama, others to Ollama Cloud (ollama.com). See §14. | No | No (`DeviceRequests: null`) | < 1 GB | < 1 vCPU | Logs only | Yes — sits in user request path | **CPU host** (currently mis-placed) |
| `gravity-embedding-wrapper` (Django) | Embedding API on `:11436`; loads `nomic-embed-text-v1.5` from HF cache and runs it on **CPU** | Yes (HF cache mount) | No (`DeviceRequests: null`) | 1–2 GB | 1–4 vCPU per request (model is ~137 M params) | Bind `/home/gravity/embedding-wrapper/volume-mapping/cache` (~600 MB) | Moderate — see §11 if traffic grows | **CPU host** (currently mis-placed) |
| `start_containers.sh` cron | **Boot-time safety net + mistral warmup.** Recreates any container that's missing on boot (Docker's `--restart unless-stopped` handles the normal case). Per-container recreate scripts at `/home/gravity/<name>/docker-run.sh`. | — | — | — | — | — | — | Wherever Ollama lives |

Anything you add later: copy the empty row template below.

```
| <name> | <role> | yes/no | yes/no | <ram> | <cpu> | <disk> | yes/no | <decision from §6> |
```

---

## 4. Request flow

### Plain chat (current path)

```
client (e.g. ollama.gravity.ind.in:11434)
  │  HTTP
  ▼
ollama_wrapper (CPU container, :11434)
  │  HTTP (localhost on the GPU host today; could be cross-host)
  ▼
ollama (GPU container, :11433 → :11434, NVIDIA_VISIBLE_DEVICES=all)
  │  CUDA
  ▼
H100 NVL (94 GB VRAM, OLLAMA_NUM_PARALLEL=1, KEEP_ALIVE=-1)
```

`OLLAMA_NUM_PARALLEL=1` means **one inference at a time per Ollama instance**.
Concurrent client requests queue at the wrapper.

### Embeddings (current path)

```
client → :11436 gravity-embedding-wrapper
                  │
                  └─ loads nomic-embed-text-v1.5 from HF cache mount
                     and runs forward pass on CPU
                  ▼
               returns embedding vector
```

> The wrapper does **not** call Ollama for embeddings, even though Ollama
> *also* has embedding models loaded (`nomic-embed-text`, `embeddinggemma`,
> `gravity-nomic`). See §12 — this duplication is an open question.

### RAG (probable / partial — wrappers are opaque)

```
client → ollama_wrapper → ollama         (chat completion)
       → embedding-wrapper                (vectorise query/docs)
       → qdrant :6333 (API key required)  (search nearest neighbours)
```

The current wrappers don't appear to orchestrate RAG end-to-end; that
glue likely lives in a calling application. **When you add that
orchestrator, place it on a CPU host** (it makes 1× LLM call, 1× embed
call, 1× Qdrant search per user request — the latency cost of being
off-GPU-host is one cheap network hop).

---

## 5. Resource profile per component (rules of thumb)

### LLM inference engine (Ollama, vLLM, TGI, llama.cpp)

- **VRAM** = `model_weights_size + kv_cache_per_request × concurrency`.
  KV cache per request scales with context length and model dimensions —
  budget 1–10 GB extra per concurrent stream for typical 7B–70B models.
- **System RAM**: small (1–4 GB). Models live in VRAM, not host RAM.
- **CPU**: light. The GPU does the work. ~1–4 vCPU per concurrent stream
  is plenty.
- **Disk**: `sum(model files)`. Quantised weights are ~½ size of fp16;
  4-bit is ~¼.
- **Network**: tolerant to clients being remote (LAN). Intolerant of
  remote *model storage* — the model file should be on local fast disk.

### Vector store (Qdrant)

- **VRAM**: zero.
- **System RAM**: rough rule `(N × dim × 4 bytes)` for fp32 vectors in
  memory. Qdrant supports on-disk + mmap variants that bring this down
  significantly. 1 M × 768-dim ≈ 3 GB raw, less with quantisation.
- **CPU**: 1–4 cores at search time; HNSW indexing is multi-threaded but
  bursty.
- **Disk**: `vectors + payloads + index`. SSD strongly preferred for
  search latency under load.
- **Network**: tolerant. Embeddings travel as vectors (KB), not as text.

### Stateless API service (Django/FastAPI/Node)

- **VRAM/CPU/RAM**: small unless it loads models. The
  `gravity-embedding-wrapper` is the cautionary example — it looks
  stateless from the outside but ships a 547 MB model and burns CPU.
- **Determine quickly**: does its `requirements.txt` import `torch`,
  `transformers`, `sentence-transformers`, `accelerate`, `vllm`, or
  `bitsandbytes`? If yes, it loads a model. If those packages are
  installed without `+cuXXX` wheels, it runs on CPU. If with `+cu124`
  or similar, it expects a GPU.

### Database (Postgres, Mongo, Redis)

- Always CPU host. State and durability concerns dominate. Put on a VM
  with backed-up persistent disks, not on an ephemeral GPU box.

### Fine-tuning / training jobs

- GPU host. Reserve a VRAM window (e.g. evict large inference models
  before launching). Currently `OLLAMA_KEEP_ALIVE=-1` would prevent that
  — see §9.

---

## 6. Placement decision rules (the hard ones)

### MUST be on a GPU host

A component **must** run on a GPU host if any of these are true:

- It loads CUDA libraries at runtime (any `+cuXXX` PyTorch wheel,
  `bitsandbytes`, `flash-attn`, `xformers`, `vllm`, `tensorrt-llm`).
- It runs an LLM inference engine natively (Ollama, vLLM, TGI,
  llama.cpp with CUDA).
- It does fine-tuning or training of a model that's larger than what a
  CPU can complete in a useful time window.
- Its `docker run` includes `--gpus`, `NVIDIA_VISIBLE_DEVICES`, or a
  `DeviceRequests` block with `"gpu"` capability (these are the same
  thing under the hood).

### SHOULD be on the GPU host (latency / efficiency)

A component is *eligible* for the GPU host (but not required) when:

- It's an agent loop that issues many small calls (>10 LLM calls per
  user request). Network round-trips compound; co-location halves
  latency variance.
- It pre/post-processes large tensors (e.g. tokenisation of huge prompts)
  and would otherwise serialise them over HTTP.

Even then, weigh the cost — see §8.

### MUST NOT be on the GPU host

A component **must not** live on the GPU host (because it wastes the
machine and contaminates the failure domain):

- Anything stateful where data loss on reboot would hurt (databases,
  vector stores, queues). The GPU box gets rebooted for driver updates
  far more often than a regular VM.
- Anything that scales horizontally and would clash with single-GPU
  scheduling (web frontends, API gateways, background workers).
- Anything with a public attack surface that doesn't strictly need
  GPU access (auth proxies, webhooks, dashboards). Keep the GPU box's
  attack surface tiny.

### Indifferent

- Small one-off admin tools, periodic crons, log shippers. Default to
  CPU host so the GPU's vCPU and host RAM remain available for the
  inference container.

---

## 7. Decision flow for a new component

```
START
  │
  ▼
Does it import CUDA libs / run an inference engine / fine-tune?
  ├─ YES ──────────────────────────────────────────► GPU HOST
  └─ NO
       │
       ▼
   Does it hold persistent state (DB, vectors, files, queues)?
     ├─ YES ─► dedicated CPU HOST with backed-up persistent disk
     └─ NO
          │
          ▼
      Is it on the user's hot path AND issues >10 LLM calls per request?
        ├─ YES ─► consider GPU HOST (only if you have CPU/RAM headroom there)
        └─ NO  ─► CPU HOST (default)
```

Apply this to the four current containers as a sanity check:

| Container | CUDA libs? | Persistent state? | >10 LLM calls? | Verdict |
|---|---|---|---|---|
| `ollama` | yes (CUDA runtime) | yes (model volume) | n/a (it *is* the LLM) | **GPU host** ✓ |
| `qdrant` | no | yes (vector store) | no | **CPU host** ✗ (currently on GPU host) |
| `ollama_wrapper` | no | no | no | **CPU host** ✗ |
| `gravity-embedding-wrapper` | no | yes (HF cache) | no | **CPU host** ✗ |

Three out of four are mis-placed today. That's not catastrophic — single-VM
convenience has real value early on — but it's the first thing to fix when
you build out a multi-host architecture.

---

## 8. Reference topology

### Today (single-host, all-in-one)

```
┌──────────────────────── GPU-1 (Standard_NC40ads_H100_v5) ────────────────────────┐
│                                                                                  │
│   ollama (GPU)        ollama_wrapper        embedding-wrapper        qdrant      │
│   :11433 → :11434     :11434                :11436                   :6333,11435 │
│   ↑                                                                              │
│   H100 NVL 94 GB                                                                 │
│                                                                                  │
│   Cron @reboot: /home/gravity/start_containers.sh                                │
└──────────────────────────────────────────────────────────────────────────────────┘
                          ▲
                          │  ollama.gravity.ind.in
                       clients
```

Failure domain: any kernel/driver reboot, any GPU OOM, any host-side
update takes the chat API, the embedding API, *and* the vector store
down at once.

### Recommended (two-tier)

```
┌─────────────────── GPU host (NC40ads_H100_v5) ──────────────────┐
│                                                                 │
│   ollama (GPU)                                                  │
│   :11434  ◄──────── private VNet / VPC peering ─────────────┐   │
│                                                             │   │
│   (and only ollama)                                         │   │
└─────────────────────────────────────────────────────────────┼───┘
                                                              │
┌────────────── API tier (small/medium CPU VM) ──────────────┼───┐
│                                                            │   │
│   ollama_wrapper      embedding-wrapper      app servers   │   │
│   :11434              :11436                 :443 (TLS)    │   │
│        │                  │                      │         │   │
│        └──────────┬───────┘                      │         │   │
│                   │ network calls to GPU host   ─┘         │   │
│                                                            │   │
└────────────────────────────────────────────────────────────┼───┘
                                                             │
┌──────────────── Data tier (CPU VM with SSDs) ──────────────┼───┐
│                                                            │   │
│   qdrant   :6333  (private only, API-key protected)  ◄─────┘   │
│   postgres / redis / etc. as you grow                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

Trade-offs to weigh before splitting:

| Concern | Single-host (today) | Two-tier (recommended) |
|---|---|---|
| Latency wrapper → Ollama | localhost (~0.1 ms) | LAN (~0.5–2 ms) |
| Cost | one expensive box | one expensive + one cheap |
| Blast radius of GPU reboot | full outage | only LLM calls fail; Qdrant + frontends stay up |
| Operational complexity | low (one VM) | moderate (two-three VMs, network) |
| Independent scaling | impossible | yes (add API VMs without touching GPU) |
| GPU resource leakage (Qdrant RAM, wrapper CPU) | yes | no |

The latency cost is small compared to the gains. Pull the trigger when
you have *either* a second customer-facing app *or* the first time
Qdrant data is mission-critical.

---

## 9. Anti-patterns observed in the current setup

These are concrete things to avoid when designing the next iteration —
and to fix in the current one when you have the bandwidth.

| Anti-pattern | Where | Status |
|---|---|---|
| Stateful + stateless services share a GPU host | Qdrant + wrappers on `GPU-1` | **Still open.** Move Qdrant to a CPU VM with backed-up disk; move wrappers to an API-tier VM. See §13 for the latency math. |
| ~~No `--restart` policy on any container~~ | ~~All four — `RestartPolicy: "no"`~~ | ✅ **Resolved 2026-05-18.** All four now `unless-stopped`; per-container recreate scripts under `/home/gravity/<name>/docker-run.sh`. `@reboot` cron rewritten as a safety net + warmup. |
| `pip install` at container start | `gravity-embedding-wrapper`'s `Cmd` | **Still open.** Bake deps into image; container start should be milliseconds, not seconds. (The `Cmd` is now version-controlled in `docker-run.sh`, but it still runs pip every restart.) |
| Django dev `runserver` in production | both wrappers | **Still open.** Switch to `gunicorn`/`uvicorn`. Requires image rebuild — recreate scripts will Just Work with the new image once you publish it. |
| Models pinned in VRAM forever | `OLLAMA_KEEP_ALIVE=-1` | **Less urgent post-NVMe migration.** Cold load is now 6 s for gpt-oss instead of 380 s, so eviction-and-reload is cheap. Could relax to `15m` if you ever want more dynamic model swapping. |
| Single-stream Ollama | `OLLAMA_NUM_PARALLEL=1` | **Still open** but not urgent — measured load is 4 requests/hour, no contention. |
| Two embedding paths | `nomic-embed-text-v1.5` in CPU wrapper *and* `nomic-embed-text` in GPU Ollama | **Still open.** Pick one. CPU embeddings are fine for low QPS, free the GPU. GPU embeddings only pay off above a few hundred QPS. |
| No TLS at the host | All ports `0.0.0.0` | **Still open.** Terminate TLS at an upstream load balancer / App Gateway; the host should not be reachable on `:11434` from the public internet. |
| Driver version skew (550 + 570) and Docker client/server skew | `dpkg -l` and `docker version` | **Still open.** Purge 550 packages; pin Docker CLI to engine major version. |

---

## 10. Sizing ready-reckoner

Quick lookups when scoping new work.

### LLMs

| Model size | fp16 | 8-bit | 4-bit | Notes |
|---|---|---|---|---|
| 7B | ~14 GB | ~7 GB | ~4 GB | Fits on any modern GPU ≥ 16 GB |
| 13B | ~26 GB | ~13 GB | ~7 GB | A100 40 GB / RTX 6000 |
| 30B | ~60 GB | ~30 GB | ~16 GB | A100 80 GB / H100; or 2× 24 GB |
| 70B | ~140 GB | ~70 GB | ~40 GB | **bnb-4bit fits on H100 NVL 94 GB** ✓; fp16 needs 2× 80 GB |
| 120B | ~240 GB | ~120 GB | ~65 GB | `gpt-oss:120b` at 65 GB just fits in 94 GB **alone** — nothing else can be resident |

Add **1–10 GB per concurrent stream** for KV cache (more for long contexts).

Current Ollama working set example: `gpt-oss:120b` (65 GB) + KV cache
leaves only ~25 GB for any other model and KV cache. That's why
`OLLAMA_NUM_PARALLEL=1` is set.

### Qdrant / vector stores

| Vectors | Dim | RAM (fp32, in-memory) | RAM (8-bit quant) | Disk |
|---|---|---|---|---|
| 100 K | 768 | ~300 MB | ~80 MB | ~1× index size (~150 MB) |
| 1 M | 768 | ~3 GB | ~800 MB | ~1.5 GB |
| 10 M | 768 | ~30 GB | ~8 GB | ~15 GB |
| 100 M | 768 | ~300 GB (use disk + mmap) | ~80 GB | ~150 GB |

Add HNSW index overhead (~1.2× of vector size). For Qdrant specifically,
tune `on_disk: true` once the working set exceeds half the host RAM.

### Embedding throughput on CPU vs GPU

| Model | CPU (32 vCPU) | GPU (H100) |
|---|---|---|
| `nomic-embed-text-v1.5` (137 M params) | ~50–200 docs/s | ~5 000–20 000 docs/s |
| `bge-large-en` (335 M) | ~10–50 docs/s | ~2 000–10 000 docs/s |
| `e5-mistral-7b-instruct` (7 B) | unusable | ~50–200 docs/s |

If your embedding QPS is comfortably below the CPU number, leave the
embedding service on a CPU host. Only graduate to GPU embeddings when
you can saturate it.

### Ingestion / batch jobs

- Ingestion writes vectors to Qdrant → CPU host (next to Qdrant ideally).
- If ingestion *generates* embeddings: same rule as the embedding
  service — small models on CPU, big models call the GPU host's API.

---

## 11. What changes when you add a second GPU host

- Ollama has no native sharding or load balancing. You'll need a tiny
  router (the `ollama_wrapper` is well-positioned to grow into this) that:
  - Routes requests to whichever GPU host has the model already loaded
    (model-affinity / sticky routing). Cold-loading a 65 GB model to
    answer one request defeats the purpose.
  - Falls back to a global queue if all instances are busy.
- KV cache is per-instance. If you want to support long conversational
  state cheaply, keep sessions sticky to one GPU host.
- Storage: each GPU host needs its *own* copy of frequently-used models
  on local disk. Sharing the Ollama volume over NFS is a footgun
  (latency and locking).
- Cost: a second `NC40ads_H100_v5` doubles your fixed bill. Verify with
  metrics that the first one is actually saturated before adding a
  second — given `OLLAMA_NUM_PARALLEL=1` today, the bottleneck is more
  likely concurrency config than hardware.

---

## 12. Open architectural questions

These are decisions worth making explicit before the architecture grows
further:

1. **Single embedding path.** Pick whether embeddings come from the
   CPU wrapper (`nomic-embed-text-v1.5`) or from Ollama
   (`nomic-embed-text` / `embeddinggemma` / `gravity-nomic`). Running
   both creates drift between vectors stored in Qdrant and vectors
   computed at query time if anyone forgets which path they're on.
2. **Public ingress.** No nginx/caddy/traefik on `GPU-1`, yet
   `ollama.gravity.ind.in` resolves to it on a public IP. Confirm
   what's terminating TLS upstream (Azure Application Gateway? Load
   Balancer? something else?) and document it; the GPU host's listening
   ports (`11434`, `11436`, `6333`, `11433`, `11435`) are all bound to
   `0.0.0.0`.
3. **Model lifecycle.** With 11 models totaling 185 GB on disk and
   only 94 GB VRAM, Ollama is constantly evicting. Do you need all 11?
   Removing the unused ones (start with `glm-ocr` and `mistral:instruct`
   if no clients use them) reduces cold-start cost.
4. **Backup cadence.** The Qdrant tarballs (`qdrant_data.tar` /
   `qdrant_image.tar`) in `/home/gravity/` are from 2026-04-09 — almost
   a month old at the time of this writing. Define an SLO for vector
   data and automate snapshots.
5. **Wrapper authentication.** The Qdrant API is key-protected; the
   wrappers' inspect output shows no auth env vars. Are the chat and
   embedding APIs supposed to be open inside the VNet, or do they
   enforce auth at the application layer (inside the image)?
6. **Wrapper cloud-routing policy.** §14 documents that `ollama_wrapper`
   silently routes some models (e.g. `gpt-oss:120b`,
   `mistral-large-3:675b`) to Ollama Cloud while serving others locally.
   The routing rules are baked into the Django image and not visible
   from outside it. Decide whether this is intentional (and document
   it) or a leftover from earlier development (and fix it). See §14 for
   the three options.
7. ~~**Should `gpt-oss:120b` be kept warm locally?**~~ ✅ **Resolved 2026-05-18**: after NVMe migration the cold load is **6.2 s** (was 380 s), so there's no longer a strong reason to pin it permanently. `OLLAMA_KEEP_ALIVE=-1` still keeps it warm once requested, but if it gets evicted by another large model, reload is now cheap.
8. **Wrapper cloud-routing — still open.** Now that local gpt-oss cold load is 6 s, the main historical reason for routing it to Ollama Cloud (cold-load pain) is gone. Re-evaluate the cloud path against the local path on cost / data residency now that latency parity is restored. See §14 for options.

---

## 13. Latency-driven placement (with measured numbers)

§5–§8 told you *what's possible*. This section tells you *what's worth
it* through the lens of user-visible latency. Use this when a stakeholder
asks "won't moving X off the GPU box make things slower?"

### 13.1 The math

```
user_latency  ≈  Σ(service_compute)  +  N_roundtrips × RTT  +  queueing
```

The only term that changes when you move a service to a different host is
`N_roundtrips × RTT`. Decide by ratio:

| `(N_roundtrips × RTT) / total_request_time` | Decision |
|---|---|
| `< 5 %` | **Split is free.** Move freely. |
| `5–20 %` | Split if you gain isolation, scaling, or cost wins. |
| `> 20 %` | **Keep co-located** unless there's a stronger reason (security, blast radius, GPU contention). |

### 13.2 Measured numbers — `GPU-1` ↔ `dev-2` (Azure CentralIndia, 2026-05-07)

Probes ran against the live stack from both hosts, addressed by public
DNS (`ollama.gravity.ind.in` and `dev.gravity.ind.in`). VNet-internal
private IPs would be marginally faster but the picture wouldn't change.

| Endpoint | Same-host (warm) | Cross-host (warm) | Network cost | % of total |
|---|---|---|---|---|
| `wrapper :11434 /api/version` | 2.5 ms | 4.5 ms | **+2 ms** | n/a (warm-up) |
| **chat** (`mistral:instruct`, 32 tok) | 58 ms | 61 ms | **+3 ms** | **≈ 5 %** |
| chat (cold model load) | 1 470 ms | n/a | invisible | < 0.3 % |
| `embed :11436` (short text) | 1.4 ms | est. ~3.5 ms | +2 ms | high % of compute, tiny absolute |
| `qdrant :6333 /collections` | 0.5 ms | **135 s — TIMEOUT** | NSG-blocked, see §13.4 | — |
| `ollama :11433 /api/version` | 0.5 ms | n/a | — | — |

ICMP `ping` failed both ways (100 % packet loss × 30) — that's NSG
filtering ICMP, **not** broken connectivity. HTTP works fine where it's
allowed.

Three side-effects worth noting for capacity planning:

- **Django wrapper costs ~2 ms internally** (2.5 ms via wrapper vs 0.5 ms
  via Ollama directly). That's the `runserver` tax, independent of host
  placement. Replace with `gunicorn` to recover ~1 ms.
- **Cold model load is ~1.5 s** — only seen on the first chat call
  because `OLLAMA_KEEP_ALIVE=-1` keeps the model pinned. Budget 1–5 s
  per cold miss when planning multi-model workloads.
- **Embedding on CPU is fast** (~1.4 ms warm) for short text. The
  embedding wrapper genuinely does not need a GPU.

### 13.3 Per-component latency verdict

| Component | Move to dev VM? | Why |
|---|---|---|
| `ollama_wrapper` (chat proxy) | **YES — free** | +3 ms on a 60 ms warm chat = ~5 %. +3 ms on a 1.5 s cold call = 0.2 %. Invisible to users. |
| `gravity-embedding-wrapper` | **YES — free** | +2 ms over 1.4 ms compute is a high *ratio* but ~3.5 ms absolute. In any RAG pipeline with an LLM in it, this is rounding error. |
| `qdrant` | **YES — but unblock first (see §13.4)** | Same-host search 0.5 ms, cross-host ~2.5 ms expected. Stateful + cheap network = move it off the GPU box. |
| `ollama` (LLM daemon) | **NO** | GPU-bound. The whole reason `GPU-1` exists. |

### 13.4 The NSG constraint (a real blocker today)

The cross-host Qdrant probe timed out at 135 s × 5 — five identical TCP
connect timeouts. That's the Azure NSG dropping ingress on port `6333`
from outside `GPU-1`. Today, **moving the wrappers to `dev-2` would
break the RAG path** even though the latency math says it's free.

Three ways to unblock, in increasing order of "right thing to do":

1. **Open NSG ingress on `:6333`** from `dev-2`'s IP/subnet. Qdrant has
   an API key (`QDRANT__SERVICE__API_KEY=4f3e2d1c-…`); that's already
   the auth boundary. Quick fix.
2. **Move Qdrant to `dev-2`** alongside the wrappers. Then wrapper ↔
   Qdrant traffic stays on localhost (0.5 ms), only LLM calls cross
   the network. Better long-term — Qdrant is stateful and shouldn't
   share fate with the GPU host's reboot cadence.
3. **Both: Qdrant on a third dedicated CPU host (data tier) with its
   own backed-up disk, NSG opened from the API tier.** Best for
   anything that's going past prototype.

Recommendation: option 2 or 3. Don't get cute and route Qdrant through
the public DNS just to make split work — keep the data path inside the
VNet, with the NSG explicitly enumerating who can talk to whom.

### 13.5 Pattern-by-pattern math

For your three dominant request shapes (single chat, RAG, batch):

| Request shape | Round-trips after split | Network cost | % of typical request | Verdict |
|---|---|---|---|---|
| Single chat (warm) | 1 (client→wrapper→ollama is 2 hops, but the wrapper-LLM hop is *internal* to whichever host both end up on if you also move the wrapper next to ollama; if they're split, +1 hop = 2 ms) | 2–3 ms | 60 ms warm chat → 5 %; multi-second answer → < 0.5 % | **split freely** |
| RAG (embed + search + chat) | 3 hops if all split | ~6 ms | RAG total = ~ms for embed + ms for search + seconds for LLM = essentially seconds | **split freely** |
| Batch / async | 1+ hops, latency invisible to end users | n/a | n/a | **always split** to cheapest host |

If you ever introduce **agent loops with ≥ 10 LLM calls per user
request**, the math flips: 10 × 60 ms warm-chat overhead + 10 × 2 ms
network = noticeable. At that point, co-locate the orchestrator with
Ollama (or accept streaming + pipelining as a mitigation). None of
your three current patterns trigger this.

### 13.6 Anti-rule

**Do not use latency as a justification for keeping stateful services
(Qdrant, future databases) on the GPU box.** The latency saving is
single-digit milliseconds. The cost is:

- Sharing the GPU host's far more frequent reboot cadence (kernel and
  driver updates) with stateful workloads.
- Wasting expensive H100 host-RAM and vCPU on a service that needs
  neither.
- Tying disk I/O for vector search to the GPU box's storage tier
  instead of a sized-for-purpose data disk.

The doctrine: **on the GPU host, only what cannot run elsewhere.**

### 13.7 When to re-measure

Re-run §13.2's probe script and refresh this section if any of these
change:

- You add agent loops (≥ 10 LLM calls per user request).
- You introduce streaming with first-token-latency SLOs (current chat
  is non-streaming).
- The two VMs move to different regions or one moves on-prem.
- You add a load balancer / API gateway in the data path (each layer
  adds ~1–5 ms).
- You enable TLS on internal hops (~1 extra ms for the handshake;
  amortised away with keep-alive).

The script lives in `d:\Gravity\dev server\outputs.txt` lines
2232–2351 — copy from there.

---

## 14. The `ollama_wrapper` is NOT a thin proxy — it routes some models to Ollama Cloud

Discovered 2026-05-18 by running diagnostics on a `gpt-oss:120b` request that
finished in milliseconds with zero GPU activity, zero new CPU load, zero
extra runner processes, and `null` for all `*_duration` fields in the
response.

### What's actually happening

The Django app inside `gravity2023/ollama_wrapper:1.1.0.0` uses the Python
`ollama` client (`/app/OllamaWrapper/Wrapper/views.py:94`) and is configured
such that **certain models proxy to ollama.com (Ollama Cloud) instead of the
local `ollama` container** on port `11433`. Smoking gun from
`/home/gravity/ollama-wrapper/volume-mapping/logs/ollama_proxy.log`:

```
ollama._types.ResponseError: this model requires a subscription,
upgrade for access: https://ollama.com/upgrade (status code: 403)
ollama._types.ResponseError: model 'mistral-large-3:675b' is
temporarily overloaded, please retry shortly ... (status code: 503)
```

`mistral-large-3:675b` does **not** exist in your local `ollama list` — that
error came back from ollama.com. Same path is used for `gpt-oss:120b`, which
is why local inference of it appears to use no resources.

### How to tell which path a model is on

Send the same request to port `11434` (wrapper) and to port `11433` (raw
Ollama) and compare the response:

| Field | Local (`:11433`) | Cloud-routed (`:11434`) |
|---|---|---|
| `*_duration` (load/prompt_eval/eval) | populated nanoseconds | **`null`** |
| First-call latency | minutes for big models (cold load) | seconds (cloud GPU pool) |
| Local GPU/CPU during call | spikes | flat |
| Outbound TCP from container | none | yes, to ollama.com |

The `*_duration: null` signature is the cheap diagnostic — if you see it in
a response from `:11434`, that model was cloud-routed.

### Implications for placement and architecture

1. **The §3 row for `ollama_wrapper` is misleading without this footnote** —
   it's not a stateless thin proxy. It's a router with its own (currently
   opaque, image-baked) routing rules. Moving it to a CPU host is still
   fine, but be aware you're moving a *routing decision point*, not a
   passthrough.
2. **Two parallel inference paths exist today**, with different cost,
   latency, and data-residency implications:
   - `:11434` (wrapper) — cloud for some models, local for others. Subject
     to Ollama Cloud subscription and external availability.
   - `:11433` (raw Ollama) — always local, always uses the H100.
3. **Whatever's calling these APIs needs to know which port for which
   workload.** This is currently implicit — there's no documented mapping
   of "this client uses this port for this reason."

### What to do

Pick deliberately:

- **(A) Keep the split** — useful if you have an Ollama Cloud subscription
  that you're already paying for, or if some models exceed your H100's
  practical capacity. Document the per-model routing in your client
  configs. Update §3 to flag the wrapper as a router not a proxy. (Done in
  this revision.)
- **(B) Force everything local** — modify the wrapper image so the
  `ollama_client` in `views.py` always points at `http://ollama:11434`
  (the local daemon) regardless of model. Requires building a new image
  tag (e.g. `gravity2023/ollama_wrapper:1.2.0.0`).
- **(C) Make the routing explicit** — keep the wrapper but document
  exactly which models go where, and add `X-Routed-To: local|cloud` to
  the response headers so callers can verify. Best for auditability.

### Cost model implications

Local path: fixed H100 cost regardless of usage.
Cloud path: Ollama Cloud is per-request (or subscription-tiered). At
current request volumes the cloud path may be cheaper *or* much more
expensive than local — depends on your traffic. Worth pulling the actual
ollama.com billing data and comparing against a "what if it ran locally"
estimate.

### Measured 2026-05-18 — when cloud is actually faster

Full benchmark and numbers in **Appendix B**. The decision-relevant
findings:

- **Local wins single-user latency by a lot.** TTFB 198 ms vs 694 ms
  (3.5×). Warm eval rate 151 vs 70 tok/sec (2.2×). Local response
  variance ±0.5 %, cloud ±25 %.
- **The crossover is at ~2 concurrent requests.** At N=1 local wins
  by 1.4 s. At N=2 they're tied. At N=4 cloud wins by 2.7 s because
  local serializes (`OLLAMA_NUM_PARALLEL=1`) while cloud parallelizes.
- **Cloud silently truncates outputs.** Requested 200 tokens, sometimes
  got 165–192. `num_predict` is a hint to cloud, not a contract.
  Anything downstream that expects fixed-length output (templated
  responses, structured extraction, JSON-mode-style pipelines) can
  break under cloud routing in ways that never reproduce locally.
- **Cloud doesn't honor `seed` reliably.** Same `seed=42, temp=0`
  produced different opening tokens across runs. Not safe for
  reproducibility-sensitive evals or A/B comparisons.

Updated routing rule (replaces the prior "pick A/B/C" framing):

| Workload | Route | Why |
|---|---|---|
| Interactive single-user chat | **LOCAL** | 3.5× faster first-token; rock-steady |
| Templated / fixed-length outputs | **LOCAL** | Cloud ignores `num_predict` |
| Evals / reproducibility | **LOCAL** | Cloud ignores `seed` |
| Batch or ≥ 3 simultaneous requests | **CLOUD** | Local serializes; cloud has horizontal headroom |
| Burst overflow (local queue depth > 1) | **CLOUD** | Avoid queue tax |
| Disaster recovery (VM down or mid-restore) | **CLOUD** | Only option |

**Local could close the concurrency gap** by raising
`OLLAMA_NUM_PARALLEL` from 1 to 2 (or 4). Each parallel slot costs
~2–3 GB of VRAM at q8_0 + 32K context for gpt-oss-120b; current
headroom is ~16 GB. Two parallel slots fit comfortably; four is
tight. The trade-off: VRAM reserved for parallelism can't be used to
host more models. Worth testing if real traffic ever sustains N ≥ 2.

---

## 15. KV cache tuning (the easy 2–8× VRAM win on Ollama)

Discovered 2026-05-18 while looking at why `llama3.3` felt sluggish despite
having a 94 GB H100.

### What was wrong

`docker exec ollama env | grep -iE "ollama|cache|flash"` showed:

```
OLLAMA_NUM_PARALLEL=1
OLLAMA_KEEP_ALIVE=-1
# (no flash attention, no KV cache type, no context cap)
```

Ollama defaults:
- Flash attention **off** (on Linux; this is a stale default for any Hopper-class GPU).
- KV cache precision: **fp16**.
- Context length: **the model's max** (`OLLAMA_CONTEXT_LENGTH:0`).

For `llama3.3` (80 layers, 128 K max context), that produced a **40 GiB
fp16 KV cache** plus a **2 GiB CPU spill** — meaning per-token attention
was bouncing tensors over PCIe to host RAM. Hence the sluggishness.

### Fix applied 2026-05-18

Recreated the `ollama` container with three extra env vars and a proper
restart policy. The recreate script lives at
`/home/gravity/ollama/docker-run.sh` and is the canonical place to edit
Ollama runtime settings going forward.

| Setting | Before | After |
|---|---|---|
| `OLLAMA_FLASH_ATTENTION` | unset (= false) | **`1`** |
| `OLLAMA_KV_CACHE_TYPE` | unset (= fp16) | **`q8_0`** |
| `OLLAMA_CONTEXT_LENGTH` | unset (= model max) | **`32768`** |
| `--restart` | none (cron-driven) | **`unless-stopped`** |

Quantised KV cache (`q8_0` and below) requires flash attention to be
**on**. Don't set `OLLAMA_KV_CACHE_TYPE` without also setting
`OLLAMA_FLASH_ATTENTION=1`, or Ollama silently falls back to fp16.

### Measured impact

| Model | KV cache before (fp16, native ctx) | KV cache after (q8_0, 32K ctx) | Reduction |
|---|---|---|---|
| `mistral:instruct` (32 layers, 32K) | 4 096 MiB | **2 176 MiB** | 46 % |
| `llama3.3` (80 layers, 128K native) | 40 960 MiB + 2 GiB CPU spill | **5 440 MiB, all on GPU** | **87 %**, no spill |
| `gpt-oss:120b` (estimated) | ~3 GiB (MoE) | ~1.5 GiB | ~50 % |

### Co-residency before vs after

| Combination | Before (fp16, full ctx) | After (q8_0, 32K) |
|---|---|---|
| `mistral` alone | 10 GB | 8.4 GB |
| `llama3.3` alone | ~82 GB *with CPU spill* | 51 GB on GPU only |
| `mistral` + `llama3.3` | would not coexist | **60 GB**, 35 GB headroom |
| `mistral` + `gpt-oss:120b` (local) | not tested | **74.4 GB**, 20 GB headroom |
| `llama3.3` + `gpt-oss:120b` | impossible | impossible (~120 GB needed) |

### Cold-load reality check (updated post-NVMe migration)

KV cache tuning helps memory and steady-state speed; **disk speed determines
cold-load time**. On 2026-05-18 the Ollama model store moved from Premium
SSD (262 MB/s read) to local NVMe (5.1 GB/s read). See §16.

| Model | Disk size | Cold load on Premium SSD (before) | Cold load on NVMe (after) | Warm steady-state |
|---|---|---|---|---|
| `mistral:instruct` | 4.1 GB | 6 s | 6 s *(setup-bound)* | ~60 ms / chat |
| `gpt-oss:120b` | 65 GB | **380 s** measured | **6.17 s** measured | ~155 tokens/sec |

The 380 s → 6.17 s improvement on `gpt-oss:120b` removes most of the prior
argument for the wrapper's cloud route. Whether to keep it cloud-served now
depends purely on cost / data residency, not on local cold-load pain.

### When to revisit

- If you add agent loops or long-context RAG that needs >32 K tokens,
  bump `OLLAMA_CONTEXT_LENGTH` (or pass `num_ctx` per request) and
  recompute the VRAM budget.
- If quality on q8_0 KV cache turns out to be unacceptable for a specific
  workload (rare — `q8_0` is widely considered free), drop to `q4_0` for
  more VRAM savings or back to fp16 for that model.
- If you upgrade to a multi-GPU host, the spill story changes — you can
  shard KV cache across GPUs and the context cap may no longer be needed.

---

## 16. Disk-tier placement (the 60× cold-load win on local NVMe)

Discovered and acted on 2026-05-18 while looking for low-risk latency wins.

### Disk bandwidth measured on `GPU-1`

```
/dev/sdb     (Premium SSD, default disk)  262 MB/s sequential read
/dev/nvme0n1 (local NVMe, was unmounted)  5.1 GB/s sequential read
```

The local NVMe is **19.5× faster on the disk benchmark**, and Ollama's
mmap+readahead pipelining with the H100's PCIe Gen5 link pushed
**effective disk→VRAM throughput to ~10 GB/s** during real cold loads.

### Migration applied 2026-05-18

1. Formatted `/dev/nvme0n1` as ext4 with label `nvme-ephemeral`.
2. Mounted at `/mnt/nvme`, persisted via `/etc/fstab` with `nofail,x-systemd.device-timeout=5s` so a missing NVMe never blocks boot.
3. `rsync -aHAX` of `/var/lib/docker/volumes/ollama/_data/` (112 GB, 45 files) → `/mnt/nvme/ollama/`. Took 5 min 26 s (Premium SSD read-bound).
4. Recreated the `ollama` container with bind mount `/mnt/nvme/ollama:/root/.ollama` instead of the named Docker volume.
5. `docker-run.sh` now refuses to start if `/mnt/nvme` isn't mounted (`mountpoint -q /mnt/nvme`) — prevents Ollama silently launching with an empty model directory.

### Measured impact

| Operation | Premium SSD (before) | NVMe (after) | Speedup |
|---|---|---|---|
| `gpt-oss:120b` cold load (65 GB) | **380 s** | **6.17 s** | **62×** |
| `mistral:instruct` cold load (4.1 GB) | 6 s | 6 s | 1× (setup-bound) |
| Container recreate downtime | n/a | 69 s wall-clock | — |
| Effective load throughput | ~170 MB/s | ~10 GB/s | ~60× |

Small models like `mistral` don't benefit because their cold start is
dominated by Python init, tokenizer setup, and CUDA graph build —
not by reading 4 GB from disk. The win scales with model size.

### Tiering rule

**Stable storage → Premium SSD. Re-creatable storage → local NVMe.**

| Data type | Tier | Why |
|---|---|---|
| Ollama models (working copy) | **NVMe** | Re-pullable; speed-sensitive on cold load |
| Ollama models (persistent rollback) | **Premium SSD** | Source of truth — survives daily deallocate, feeds the morning restore |
| HuggingFace cache | Premium SSD currently; NVMe is eligible | Re-pullable from HF Hub |
| Docker overlayfs / image layers | Premium SSD | Re-pullable from registry but containers themselves need to survive deallocate |
| **Qdrant collections** | **Premium SSD** | **Stateful customer data — must NEVER live on ephemeral disk** |
| Postgres / future databases | Premium SSD | Same — stateful |
| Container logs | Premium SSD | Convenient to have them survive deallocate |

### ⚠️ Daily-deallocate constraint (discovered 2026-05-18)

`GPU-1` is **deallocated nightly at 8 PM and manually restarted in the
morning** for cost control. This makes the local NVMe and the Azure
resource disk (`/mnt`) **both ephemeral** — wiped every single morning.

This isn't a bug in the design; it's the operational reality of running
expensive GPU hardware part-time. But it changes the placement rule:

**Anything that takes more than 5 minutes to regenerate must NOT live
solely on ephemeral storage.** Put a persistent copy somewhere durable
(Premium SSD, blob storage, another VM) and treat the ephemeral copy as
a cache.

For Ollama models specifically, the working copy lives on NVMe (fast),
but the **source of truth is the Premium SSD rollback volume**
(`/var/lib/docker/volumes/ollama/_data` — 112 GB, 45 files). The morning
boot script auto-detects the wiped NVMe, reformats and mounts it, and
rsyncs the rollback over. Total morning warm-up: 7–10 minutes,
automated.

The auto-restore mechanism is described in `azure-llm-setup.md` §11. The
bug that motivated it (rsync silently writing to the resource disk
because of an fstab boot race) is documented in the
[learnings doc](learnings/2026-05-18-gpu-llm-optimization.md).

### Three failure modes the design now defends against

1. **NVMe wiped but device intact** (the normal morning case): boot
   script reformats with the same label, mounts, restores from rollback.
2. **NVMe mount loses a race with `waagent`'s mount of `/mnt`** (the
   first-day bug): fixed by `x-systemd.requires-mounts-for=/mnt` in
   `/etc/fstab`. The NVMe mount waits until `/mnt` is mounted, so it
   correctly nests inside instead of being hidden underneath.
3. **Mount fails because the mountpoint directory `/mnt/nvme` no
   longer exists** (the second-day bug, found in the first real
   deallocate-cycle test): fixed by `mkdir -p "$NVMe_MOUNT"` in
   `start_containers.sh` before the `mount -a` call. `waagent`
   recreates `/mnt` fresh on every Start, taking our sub-directories
   with it — so the script must recreate the mountpoint every boot.

### What if everything goes wrong

If the boot script itself fails and you need to recover by hand:

```bash
docker stop ollama
sudo mkfs.ext4 -L nvme-ephemeral -F /dev/nvme0n1
sudo mount -a
sudo rsync -aHAX /var/lib/docker/volumes/ollama/_data/ /mnt/nvme/ollama/
docker start ollama
```

(Five commands. That's exactly what the script does automatically.)

If even the rollback is gone (e.g. someone accidentally `docker volume rm
ollama` after we said not to), re-pull each model from Ollama Hub:

```bash
for m in mistral:instruct gpt-oss:120b phi4_compliance translategemma:27b glm-ocr embeddinggemma nomic-embed-text gravity-nomic; do
  docker exec ollama ollama pull "$m"
done
```

Takes 30–60 minutes for the current 8-model set depending on Hub speed.

### When to revisit

- If you add a model larger than the current largest (`gpt-oss:120b`,
  65 GB), check `df -h /mnt/nvme` first.
- If embedding-wrapper cold starts become a pain point, move
  `/root/.cache/huggingface` and the wrapper's HF cache onto NVMe too —
  but remember to keep a persistent copy too.
- If you ever migrate to an Azure SKU **without** local NVMe, the
  `mountpoint -q` check in `docker-run.sh` will refuse to start the
  container — flip back to the named volume on Premium SSD, or attach a
  fast managed disk.
- If you switch from daily-deallocate to always-on, the auto-restore
  rsync becomes wasted work — disable Phase 1 of `start_containers.sh`
  to skip it.

---

## Appendix — Quick commands you'll want when adding a new component

When you get the next "should this go on the GPU box?" question, run
these against the *image you're considering deploying* and apply §6:

```bash
# Does it ship CUDA libs?
docker run --rm <image> sh -c 'ldconfig -p | grep -i -E "cuda|cudnn|nvidia"' 2>/dev/null

# Does its requirements include GPU packages?
docker run --rm <image> sh -c 'pip list 2>/dev/null | grep -iE "torch|cuda|bitsandbytes|flash|xformers|vllm|triton"'

# Does it want a GPU device?
docker inspect <image> 2>/dev/null | grep -i -E "nvidia|gpu|cuda|DeviceRequests" -B2 -A4
```

If all three return empty/unrelated → CPU host. If any return real hits → GPU host.

---

## Appendix B — Benchmark: `gpt-oss:120b` local vs cloud (2026-05-18)

Reproducible benchmark script lives at `/tmp/benchmark.py` on `GPU-1`.
Output captured to `/tmp/benchmark_output.txt`. Re-run quarterly — cloud
capacity, queueing, and per-replica hardware drift over time.

Setup at time of measurement:
- LOCAL = `http://localhost:11433/api/chat` (direct to local Ollama on
  H100 NVL, 94 GB VRAM, gpt-oss already in VRAM from `OLLAMA_KEEP_ALIVE=-1`).
- CLOUD = `http://localhost:11434/api/chat` (via `ollama_wrapper`,
  which routes `gpt-oss:120b` to Ollama Cloud — see §14).
- Both endpoints warmed before timed runs.
- Common options: `seed=42, temperature=0`.

### B.1 — Output length sweep (medium prompt, 3 warm runs each)

| Path | 50 tok wall | 200 tok wall | 800 tok wall* | Eval rate (warm) |
|---|---|---|---|---|
| LOCAL | **518 ms** | **1571 ms** | **1692 ms** | **151 tok/sec** (±0.5 %) |
| CLOUD | 1351 ms | 2390 ms | 2710 ms | 70–85 tok/sec end-to-end (±25 %) |
| **Local advantage** | **2.6×** | **1.5×** | **1.6×** | **2.0×** |

*800-token runs hit natural EOS earlier — actually produced ~218 tok
locally, ~200 tok on cloud.

Cloud has a **~700 ms fixed overhead** per request (matches the TTFB
below). At 50 tokens that overhead is half the response time; at 800
it's amortized away. Local fixed overhead is ~150 ms (HTTP + minor
prefill).

### B.2 — Time-to-first-token (streaming, 3 runs each)

| Path | TTFB (median) | Variance |
|---|---|---|
| LOCAL | **198 ms** | ±5 ms |
| CLOUD | 694 ms | ±20 ms |
| **Local advantage** | **3.5×** | tighter |

198 ms feels instant; 694 ms feels laggy. For chat UI this matters
more than throughput.

### B.3 — Concurrency (200 tok each — the headline test)

| Concurrency N | LOCAL wall | LOCAL agg tok/s | CLOUD wall | CLOUD agg tok/s | Verdict |
|---|---|---|---|---|---|
| 1 | **1.57 s** | 127 | 2.97 s | 67 | **LOCAL wins by 1.4 s** |
| 2 | 3.00 s | 133 | **3.03 s** | 111 | dead heat |
| 4 | 5.81 s | 138 | **3.13 s** | **247** | **CLOUD wins by 2.7 s** |

LOCAL per-request walls at N=4 were `[5813, 2993, 4402, 1584] ms` —
classic serial queueing behind `OLLAMA_NUM_PARALLEL=1`.

CLOUD per-request walls at N=4 were `[3126, 2789, 2805, 2597] ms` —
all four ran in parallel.

**The crossover at N≈2 is the most important finding in this
benchmark.** It is the rule the wrapper's routing logic should
embody if it gets re-written.

### B.4 — Prompt variety (200 tok, 2 runs each)

| Prompt | LOCAL wall (mean) | LOCAL tok/s | CLOUD wall (mean) | CLOUD tok/s |
|---|---|---|---|---|
| short ("Hello, how are you?") | 759 ms | 152 | 1492 ms | 33 |
| code (Sieve of Eratosthenes) | 1601 ms | 151 | 2524 ms | 79 |
| reasoning (trains word problem) | 1620 ms | 150 | 2308 ms | 87 |

LOCAL holds 150 ± 2 tok/sec across all three prompt types. CLOUD
varies 33 – 87 tok/sec.

The "short" prompt is the worst case for cloud — fixed overhead is
~95 % of total time. **Don't route a high volume of small prompts
(classification, intent detection, short Q&A) to cloud.**

### B.5 — Caveats

- Numbers are point-in-time. Cloud capacity changes; re-run quarterly.
- Local was tested with the **only** model loaded into VRAM. Under VRAM
  pressure with multiple models swapping, local cold-load can dominate.
  This benchmark assumes the working-set fits.
- Cloud responses did not always honor `num_predict` or `seed`. If your
  application depends on exact length or reproducibility, the
  "performance" numbers above understate the real problem — cloud
  is the wrong path entirely for those workloads.
- We did not measure cost. Ollama Cloud billing data needed to close
  that side of the comparison.

### B.6 — How to re-run

```bash
python3 /tmp/benchmark.py 2>&1 | tee /tmp/benchmark_output_$(date +%F).txt
```

(Script body preserved in `/tmp/benchmark.py`. If wiped by the daily
deallocate, regenerate from the heredoc in `learnings/2026-05-18-gpu-llm-optimization.md`.)
