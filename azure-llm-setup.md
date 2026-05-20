# Azure GPU LLM Hosting — `GPU-1`

Snapshot taken **2026-05-07** from `gpu@ollama.gravity.ind.in`. All facts in
this document come from live diagnostics on the machine; figures (utilisation,
free space, etc.) are point-in-time.

---

## 1. Overview

`GPU-1` is a single-node Azure GPU VM in Central India that serves Gravity's
LLM and embedding APIs. The stack is fully containerised: an upstream
Ollama daemon (GPU-backed) sits behind two Django wrappers, with Qdrant as
the vector store. There is no native Ollama install, no nginx, and no
reverse proxy on the host — clients reach the wrappers directly on
ports `11434` (chat) and `11436` (embeddings).

```
client ──► :11434  ollama_wrapper (Django) ──► :11433 ollama (GPU)
client ──► :11436  gravity-embedding-wrapper (Django)
client ──► :6333   qdrant   (also exposed on :11435)
```

---

## 2. Azure VM

| | |
|---|---|
| VM name | `GPU-1` |
| Public DNS | `ollama.gravity.ind.in` |
| **VM SKU** | **`Standard_NC40ads_H100_v5`** |
| Region / Zone | `CentralIndia` / `2` |
| Resource group | `GPU` |
| Subscription | `28f94c9f-4d7b-47a2-affd-169087440124` |
| Image | `canonical : ubuntu-24_04-lts : server : 24.04.202509170` |
| Security type | TrustedLaunch (vTPM on, SecureBoot off) |
| Encryption at host | disabled |
| VM ID | `2c825a5e-54af-4eef-85fb-1909227ef3ee` |
| Private IP | `172.17.0.4/24` |

---

## 3. Hardware Specs

### CPU
- **AMD EPYC 9V84** (Zen 4 / Genoa, 96-core silicon; 40 vCPUs presented to the VM)
- 1 socket, 1 thread per core, NUMA node 0 covers all 40 vCPUs
- Caches (totals): L1d 1.3 MiB, L1i 1.3 MiB, L2 40 MiB, L3 160 MiB
- AVX-512 + BF16 (`avx512_bf16`), AES-NI, SHA-NI, VAES, VPCLMULQDQ available

### Memory
- **314 GiB usable** (`MemTotal: 329 973 012 kB`)
- **No swap configured**

### GPU
- **1 × NVIDIA H100 NVL** (Hopper architecture)
- **VRAM: 95 830 MiB (~94 GB)**, ECC on, MIG disabled
- Power cap 400 W (87 W observed at idle / 35 °C)
- PCIe Gen5, pass-through virtualisation
- VBIOS `96.00.9F.00.04`, GSP firmware `570.211.01`
- Topology: GPU0 ↔ NIC0 (`mlx5_0`) on the same NUMA node 0 (CPU affinity 0–39)

### Networking
- Mellanox `mlx5_0` NIC, accelerated networking interface `enP23727s1`
- `eth0` 172.17.0.4/24, default route via 172.17.0.1
- Docker bridge `docker0` 172.18.0.1/16

### Disks

| Device | Size | Mount | FS | Use |
|---|---|---|---|---|
| `sdb` (managed, Premium_LRS) | 512 GB | `/` | ext4 | OS + Docker root — was 72 % full pre-migration |
| `sda` (managed, Premium_LRS) | 512 GB | `/data` | ext4 | Empty data disk |
| `sdc` | 128 GB | `/mnt` | ext4 | Azure ephemeral resource disk |
| **`nvme0n1`** *(Microsoft NVMe Direct Disk v2)* | **3.5 TB** | **`/mnt/nvme`** *(post-migration 2026-05-18)* | **ext4 (LABEL `nvme-ephemeral`)** | **Holds Ollama model store at `/mnt/nvme/ollama` — 112 GB used. Bind-mounted into the `ollama` container as `/root/.ollama`** |

### Disk bandwidth measured 2026-05-18

| Device | Sequential read (`dd if=… of=/dev/null bs=1M count=2048 iflag=direct`) |
|---|---|
| `sdb` (Premium SSD) | **262 MB/s** |
| `nvme0n1` (local NVMe) | **5.1 GB/s** |

= **19.5× faster** on NVMe. Effective Ollama cold-load throughput hit ~10 GB/s due to mmap+readahead pipelining with the Hopper PCIe Gen5 link.

### NVMe persistence caveat — load-bearing for this deployment

`/dev/nvme0n1` is `Microsoft NVMe Direct Disk v2` — local NVMe attached to
the SKU. **Wiped on VM deallocate (full stop)**, **survives reboots /
container restarts / apt upgrades**. The Ollama models are reproducible
from a persistent rollback so this trade is acceptable for model storage;
**stateful data (Qdrant collections, customer data) must NOT live on
NVMe**.

> **⚠️ Operational reality (2026-05-18)**: `GPU-1` is **deallocated nightly
> at 8 PM and manually restarted in the morning**. Every single morning
> the NVMe is wiped along with everything else ephemeral. This means the
> Ollama model store has to be restored from a persistent source on every
> boot — see §11 for how `start_containers.sh` handles this automatically.

The fstab line for `/mnt/nvme` uses `defaults,nofail,x-systemd.device-timeout=5s,x-systemd.requires-mounts-for=/mnt`:
- `nofail` — boot doesn't hang if NVMe is missing
- `device-timeout=5s` — give up on the device quickly
- **`x-systemd.requires-mounts-for=/mnt`** — wait for `waagent` to mount
  `/mnt` (the resource disk) before processing this line. **Without this,
  there's a boot race**: if our line runs first, NVMe mounts at
  `/mnt/nvme`, then `waagent` later mounts `/mnt` and **hides** the NVMe
  mount underneath it. Anything written to `/mnt/nvme/ollama` after that
  silently lands on the **resource disk**, which is also ephemeral and
  has only 128 GB — filling it up. We hit this bug on 2026-05-18 (see
  learnings doc).

---

## 4. Operating System

- **Ubuntu 24.04.3 LTS** (Noble Numbat), kernel `6.17.0-1011-azure`
- Firmware: Hyper-V UEFI v4.1 (2025-01-28)
- Time zone `UTC`, NTP active (chrony)
- Uptime at sample: ~6 h since last reboot (2026-05-07 06:52 UTC)
- **Login banner reports `*** System restart required ***`** and **105 pending apt updates** (15 standard security, 2 ESM)

---

## 5. GPU & CUDA Stack

- **NVIDIA driver `570.211.01`** (server flavour) — active
- CUDA runtime advertised by driver: **12.8**
- No CUDA toolkit installed (`nvcc` not in PATH)
- `nvidia-container-toolkit 1.18.2`, configured in `/etc/docker/daemon.json`:

```json
{ "runtimes": { "nvidia": { "args": [], "path": "nvidia-container-runtime" } } }
```

- `nvidia-persistenced.service` is enabled and active
- Both `nvidia-driver-550-server` and `nvidia-driver-570-server` packages are installed; `libnvidia-compute-550` is in `rc` (config-files only) state — leftover from a driver upgrade

---

## 6. Python / ML Runtime (host, conda `base`)

- **Miniconda3** at `/root/miniconda3`, **Python 3.13.9**
- **PyTorch 2.6.0+cu124** — `torch.cuda.is_available() = True`, 1 device (`NVIDIA H100 NVL`)
- Other relevant libs: `transformers 4.57.3`, `accelerate 1.12.0`, `peft 0.18.1`, `sentence-transformers 5.3.0`, `triton 3.2.0`, `torchvision 0.21.0+cu124`
- HuggingFace cache `/root/.cache/huggingface` = **79 GB**, holding:
  - `unsloth/meta-llama-3.1-70b-instruct` (full precision)
  - `unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit`
  - `unsloth/phi-4` and `unsloth/phi-4-bnb-4bit`
  - `openai/gpt-oss-20b`
  - `nomic-ai/nomic-embed-text-v1.5`, `nomic-ai/nomic-bert-2048`

> Note the `+cu124` PyTorch wheel runs fine against the 12.8 driver
> (forward-compatible). No host-side CUDA toolkit is needed for runtime;
> the NVIDIA driver alone is sufficient.

---

## 7. LLM Serving (Docker)

Docker engine: **server 27.5.1**, **client 29.1.3** (skewed but compatible).
NVIDIA runtime registered. All four containers were `Up 6 hours` at sample.

| Container | Image | Host → Container ports | Role |
|---|---|---|---|
| `ollama` | `ollama/ollama:latest` | `11433 → 11434` | Ollama daemon, GPU-backed. **Container recreated 2026-05-18 with tuned KV-cache env (see below); recreate script at `/home/gravity/ollama/docker-run.sh`.** |
| `ollama_wrapper` | `gravity2023/ollama_wrapper:1.1.0.0` | `11434 → 11434` | Django chat API. **NOT a thin proxy — it routes some models (incl. `gpt-oss:120b`, `mistral-large-3:675b`) to Ollama Cloud at ollama.com; others go to local `ollama`. See architecture-and-placement.md §14.** |
| `gravity-embedding-wrapper` | `gravity2023/gravity-embedding-wrapper:1.1.0.0` | `11436 → 11436` | Django embedding API |
| `qdrant` | `qdrant/qdrant:latest` | `6333 → 6333` and `11435 → 6333` | Vector store (HTTP) |

### Ollama model storage

- **Active**: bind mount `/mnt/nvme/ollama` → `/root/.ollama` inside the
  container. **24 GB used** by **6 models** as of 2026-05-19 (down from 175 GB / 11 models — see deletions below).
- **Rollback / source of truth**: the Docker named volume `ollama` at `/var/lib/docker/volumes/ollama/_data` on Premium SSD is now the persistent source for the daily auto-restore. **DO NOT** delete — it feeds Phase 1 of `start_containers.sh` every morning.
- **HuggingFace cache** lives on Premium SSD at `/home/gravity/hf-cache` (~113 GB after pulling `openai/gpt-oss-120b` for vLLM) and at `/home/gravity/embedding-wrapper/volume-mapping/cache` for the embedding wrapper.

### Models removed

| Model | Date | Disk | Reason |
|---|---|---|---|
| `llama3.3:latest` | 2026-05-18 | 42 GB | Unused |
| `deepseek-ocr:latest` | 2026-05-18 | 6.7 GB | Unused |
| `glm-4.7-flash:latest` | 2026-05-18 | 19 GB | Unused; `glm-ocr` is sufficient |
| `phi4_compliance:latest` | 2026-05-18 | 29 GB | Unused; Modelfile preserved at `/home/gravity/saved-modelfiles/` |
| `gpt-oss:120b` | 2026-05-19 | 65 GB | **Migrated to vLLM** — see §7 cloud-routing/vLLM section |

Total freed: **~162 GB** on NVMe + matching deletion on Premium SSD rollback = **~324 GB across both tiers**.

**6 models remain on Ollama**: `mistral:instruct` (4.1 GB), `translategemma:27b` (17 GB), `glm-ocr` (2.2 GB), `embeddinggemma` (621 MB), `nomic-embed-text` (274 MB), `gravity-nomic` (274 MB). Total ~24 GB on disk. `gpt-oss:120b` lives separately on vLLM (see below).

### VRAM at sample time

| PID | Process | VRAM |
|---|---|---|
| 49123 | `/usr/bin/ollama runner --model …4824460d…` | 77 816 MiB |
| 99510 | `/usr/bin/ollama runner --ollama-engine --model …970aa74c…` | 1 130 MiB |

Total ~78 961 MiB used of 95 830 MiB (≈ 82 %).

### Wrapper internals (observed)

```
/usr/local/bin/python OllamaWrapper/manage.py runserver 0.0.0.0:11434   (uid 5678)
/usr/local/bin/python manage.py runserver 0.0.0.0:11436                  (uid 5678)
```

Both wrappers are Django dev servers (`runserver`) running as uid 5678 inside their containers. The embedding-wrapper container installs its requirements at runtime via `pip install --user -r /app/requirements.txt`.

### Ollama env (post-migration, 2026-05-18)

The `ollama` container was recreated twice on 2026-05-18: once to enable flash
attention + quantised KV cache + context cap, then again to move model storage
to local NVMe. Active env now:

| Variable | Value | Why |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | `1` | Single concurrent inference per Ollama instance |
| `OLLAMA_KEEP_ALIVE` | `-1` | Models stay in VRAM forever once loaded |
| `OLLAMA_FLASH_ATTENTION` | **`1`** *(new 2026-05-18)* | Required for q8_0 KV cache; ~1.5–2× faster attention on Hopper |
| `OLLAMA_KV_CACHE_TYPE` | **`q8_0`** *(new 2026-05-18)* | Halves KV cache vs fp16 with negligible quality loss |
| `OLLAMA_CONTEXT_LENGTH` | **`32768`** *(new 2026-05-18)* | Caps context window; before, llama3.3 used 128K → 40 GiB KV cache with 2 GiB CPU spill |
| `OLLAMA_LOAD_TIMEOUT` | **`15m`** *(new 2026-05-18)* | Default 5m used to time out on cold-loading gpt-oss; no longer a risk after NVMe move, but kept as defence-in-depth |
| `OLLAMA_HOST` | `0.0.0.0:11434` | Container's listen address |

Container also has `--restart unless-stopped` (previously `RestartPolicy: "no"`).
Model storage moved from the Docker named volume `ollama` (Premium SSD) to a
bind mount `/mnt/nvme/ollama:/root/.ollama` (local NVMe).

The recreate script is **`/home/gravity/ollama/docker-run.sh`** — edit there
and re-run to change settings. It includes a safety check that refuses to start
if `/mnt/nvme` isn't mounted, so a wiped NVMe won't silently launch Ollama with
an empty model directory.

### KV cache impact (2026-05-18, q8_0 + flash attention + 32K context)

| Model | Before (fp16, full ctx) | After (q8_0, 32K) | Savings |
|---|---|---|---|
| `mistral:instruct` | 4 096 MiB | 2 176 MiB | 46 % |
| `llama3.3` (removed since) | 40 960 MiB + 2 GiB CPU spill | 5 440 MiB, all on GPU | **87 %, no spill** |

### Cold-load impact (2026-05-18, NVMe migration)

Models now live on local NVMe (5.1 GB/s read) instead of Premium SSD (262 MB/s).

| Model | Cold load before | Cold load after | Speedup |
|---|---|---|---|
| `mistral:instruct` (4.1 GB) | 6.0 s | 6.0 s | ~1× (setup-bound, not disk-bound) |
| `gpt-oss:120b` (65 GB) | **380 s** | **6.17 s** | **62×** |

The dramatic gpt-oss improvement comes from mmap+readahead pipelining over
PCIe Gen5; effective disk→VRAM throughput was ~10 GB/s. Small models like
mistral don't benefit because their cold load is dominated by Python startup,
tokenizer init, CUDA graph build — not by reading weights.

### Co-residency capability (post-tuning)

| Combination | Total VRAM | Headroom on 94 GB H100 |
|---|---|---|
| `mistral` + `gpt-oss:120b` | 74.4 GB | 20 GB (current warm set) |
| `mistral` + `phi4_compliance` (29 GB on disk) | ~38 GB | 56 GB |
| `mistral` + `gpt-oss` + `phi4_compliance` | ~103 GB | **−9 GB — would evict** |

### Cloud-routing surprise

`ollama_wrapper` (port 11434) is **not** a transparent proxy. Diagnostics
2026-05-18 found it routes `gpt-oss:120b` and at least one other model
(`mistral-large-3:675b`, which doesn't exist locally) to **Ollama Cloud
(ollama.com)**. Tell-tale signs of a cloud-routed response from the wrapper:

- `*_duration` fields are `null` in the JSON.
- Zero local GPU/CPU activity during the call.
- Sub-second response for models that would take seconds locally.

Wrapper log `/home/gravity/ollama-wrapper/volume-mapping/logs/ollama_proxy.log`
captures historical 403 ("subscription required") and 503 ("temporarily
overloaded") errors from ollama.com.

To always hit local Ollama, bypass the wrapper and call port **11433**
directly. `gpt-oss:120b` local cold-load is **6.2 seconds** post-NVMe migration
(was 380 s on Premium SSD); warm inference is **151 tokens/sec** on the H100
(measured 2026-05-18, ±0.5 % run-to-run variance).

**Benchmark vs Ollama Cloud (2026-05-18 Ollama; 2026-05-19 vLLM)** —
full tables in `architecture-and-placement.md` Appendix B (Ollama) and
Appendix C (vLLM). Headlines from the **vLLM** path, now the
recommended one for gpt-oss:

- TTFT: **26 ms** local vLLM vs 700 ms cloud (27× faster locally).
- Warm tok/sec: **180** local vLLM vs 76 cloud (2.4× faster locally).
- Concurrent N=8 aggregate: **1138 tok/sec** local vLLM vs 249 cloud (4.6× higher).
- N=8 latency consistency: ±0.5 % local vs ±50 % cloud.
- Cloud still silently truncates and ignores `seed`.

**Practical rule (post-vLLM migration, refined 2026-05-19):**

| Workload | Route | Reason |
|---|---|---|
| Single user (any prompt size) | **local vLLM (port 8001)** | always faster, ±0.5 % vs ±100 % variance |
| Short prompts at any concurrency | **local vLLM** | stress-tested to 1024+, no errors |
| Long-prompt RAG at N ≥ 4 (mean case) | cloud wins ~4-8 s vs local 9-16 s wall | cloud's horizontal scale |
| Long-prompt RAG with SLA constraints | **local vLLM** | cloud has 10× tail-latency outliers |
| Reproducibility / fixed-length | **local vLLM** | cloud ignores `seed`, `num_predict` |
| Burst / disaster recovery | cloud fallback | only option |

**MVP wrapper change recommendation**: "all gpt-oss → local vLLM, cloud
as fallback on local error or queue overflow." This captures the
single-user, short-prompt, and SLA wins. The only workload it
deoptimizes is sustained concurrent long-prompt RAG — a workload
Gravity doesn't have at current 4 req/hr peak.

See `architecture-and-placement.md` §14 for the full reasoning and
Appendix C.11-C.13 for the benchmark data behind each row.

**Serving stack for gpt-oss as of 2026-05-19:**
- vLLM container `vllm/vllm-openai:gptoss` serving `openai/gpt-oss-120b`
- Listens on host port 8001 (container 8000)
- HF cache on Premium SSD at `/home/gravity/hf-cache` (~113 GB, persists across daily deallocate)
- VRAM footprint: 64.4 GiB weights + 12.0 GiB KV cache budget at `--gpu-memory-utilization 0.85 --max-model-len 8192`
- Cold-start: ~90-180 s (weights load + CUDA graph capture)

**Concurrent capacity (stress-tested 2026-05-19):**

| Workload pattern | Concurrent capacity | Aggregate throughput |
|---|---|---|
| **Short chat** (~250 tokens total) | 800+ confirmed, ~1500 extrapolated | ~7,900 tok/sec |
| **Long-context RAG** (~25K tokens input) | **10 simultaneous before queueing** | **~12,200 tok/sec combined in+out** |

- **Reliability: zero server-side errors observed** through N=1024 (short prompts) and N=12 (25K prompts). The 3 errors at N=1024 were client-side (`ulimit -n` exhausted). Past the comfortable ceiling, vLLM queues gracefully — it does not refuse connections.
- **Throughput is HIGHER for long-context workloads** (12,229 tok/s at 25K-context N=10) than short-prompt workloads (7,900 tok/s at short N=512). Reason: long inputs are prefill-heavy, and prefill is compute-bound parallel attention — better GPU utilization than memory-bound decode.
- **Short-request latency tiers:**
  - Real-time chat (≤ 1.2 s): up to **8 concurrent**
  - Snappy interactive (≤ 2.3 s): up to **64 concurrent**
  - Agents / RAG (≤ 5 s): up to **192 concurrent**
  - Batch (≤ 12 s): up to **512 concurrent**
- **Long-context latency**: ~3.7 s for 1 user, ~20 s for 10 users (linear scaling past N=4).

Current production traffic is ~4 req/hour — system is 3+ orders of
magnitude over-provisioned for headroom. Full stress-test results in
`architecture-and-placement.md` Appendix C.8 (short prompts), C.11
(long context), and C.12 (definitive capacity reference table).

**Context length set to 32K via `vllm/vllm-openai:latest` (2026-05-19):**

Originally ran on the `:gptoss` tag at 8K because 32K crash-looped on
that build (sampler init bug). Switching to `vllm/vllm-openai:latest`
(v0.21.0) resolved it — 32K boots cleanly. The `:gptoss` tag was an
early specialized build that has since been mainlined; the `:latest`
tag is now the recommended path.

New runtime config (post-upgrade):

| Parameter | Value |
|---|---|
| Image | `vllm/vllm-openai:latest` (v0.21.0) |
| `--max-model-len` | **32768** |
| GPU KV cache pool | **251,561 tokens** (up from 174K on the old build) |
| Max concurrency at 32K full context | **7.68×** |
| Init time | ~89 s (down from ~140 s) |
| MoE backend | TRITON Mxfp4 + FlashAttention 3 |

The KV pool grew 44 % despite the higher per-request cap — v0.21.0
manages KV cache more efficiently. Net result: at 32K cap, you have
about the same effective concurrency for typical short prompts as you
had at 8K cap on the old build, plus the ability to handle long-context
requests.

**Historical note on the crash discovered earlier in the session:**
The `:gptoss` tag crashed deterministically in `_dummy_sampler_run`
when started with `--max-model-len 32768`. We verified 15 consecutive
crashes via `RestartCount` before catching it. The diagnostic confusion
(thinking a long 22K-token request had caused the crash, when in fact
the crash was during init and the long request just happened to hit a
crash-cycle) is documented as Lessons 9 and 10 in the learnings doc.

---

## 8. Vector Store

- **Qdrant** in `qdrant/qdrant:latest` container
- HTTP API on `:6333` (also re-exposed on `:11435` for legacy clients); gRPC `6334` is container-internal only
- Backup tarball **`/home/gravity/qdrant_data.tar` (~12 GB, 2026-04-09)** indicates a manual snapshot/restore workflow

---

## 9. Networking & Endpoints

Listening on the host (from `ss -tulpn`):

| Port | Bind | Service |
|---|---|---|
| 22 | 0.0.0.0 / [::] | OpenSSH |
| 53 | 127.0.0.x | systemd-resolved (local) |
| 6333 | 0.0.0.0 / [::] | Qdrant HTTP |
| 11433 | 0.0.0.0 / [::] | Ollama daemon |
| 11434 | 0.0.0.0 / [::] | `ollama_wrapper` (chat API) |
| 11435 | 0.0.0.0 / [::] | Qdrant HTTP (alias) |
| 11436 | 0.0.0.0 / [::] | `gravity-embedding-wrapper` (embeddings) |

- Host-level firewall (`ufw`) is **inactive**; ingress filtering is therefore the responsibility of the **Azure NSG** in front of this VM.
- DNS `ollama.gravity.ind.in` resolves to this host and is the canonical client entry point.

---

## 10. Storage Layout

Post-cleanup layout (updated 2026-05-20):

| Path | Size | Disk | Notes |
|---|---|---|---|
| **`/mnt/nvme/ollama`** | **24 G** | **NVMe** | **Active Ollama model store (bind-mounted into the `ollama` container as `/root/.ollama`). 6 models** (down from 11 → 8 → 6 after llama3.3 / deepseek-ocr / glm-4.7-flash / phi4_compliance / gpt-oss-120b removals). |
| `/var/lib/docker/volumes/ollama` | 24 G | Premium SSD | **Source of truth for daily auto-restore** — kept in sync with NVMe via post-deletion `rsync --delete`. **DO NOT** `docker volume rm ollama`. |
| `/home/gravity/hf-cache` | ~113 G | Premium SSD | HuggingFace cache for vLLM. Holds `openai/gpt-oss-120b` MXFP4 weights. Persists across daily deallocate. |
| `/var/lib/docker/overlay2` | ~30 G | Premium SSD | Container layers (now includes vllm/vllm-openai:latest, ~11 G) |
| `/var/lib/docker/containers` | ~150 M | Premium SSD | Container metadata / json-file logs |
| `/root/qdrant_storage` | ~12 G | Premium SSD | Qdrant collections — **stays on Premium SSD; do NOT move to ephemeral NVMe** |
| `/home/gpu` | 36 G | Premium SSD | |
| `/home/gravity` | ~13 G | Premium SSD | Wrapper build contexts, `start_containers.sh`, Qdrant tarballs, `ollama/docker-run.sh`, `vllm/docker-run.sh`, `saved-modelfiles/` |
| `/home/gourav` | 24 K | Premium SSD | |
| `/data` (`/dev/sda`) | empty | Premium SSD | 512 GB managed disk, unused |
| `/mnt` (`/dev/sdc`) | empty | ephemeral | 128 GB Azure resource disk |
| `/mnt/nvme` (`/dev/nvme0n1`) | 112 G / 3.5 T | **NVMe (local, ephemeral)** | ext4, LABEL `nvme-ephemeral`, mounted via fstab with `nofail` |

`/home/gravity/` notable files (post-2026-05-18):

```
ollama/docker-run.sh                  recreate script for the ollama container (NVMe-backed)
qdrant/docker-run.sh                  recreate script for the qdrant container
ollama-wrapper/docker-run.sh          recreate script (replaces old parameterised version)
ollama-wrapper/docker-run.sh.original old parameterised script preserved for reference
embedding-wrapper/docker-run.sh       recreate script (replaces old parameterised version)
embedding-wrapper/docker-run.sh.original old parameterised script preserved for reference
vllm/docker-run.sh                    recreate script for vllm_gpt_oss (added 2026-05-19)
vllm/docker-run.log                   log written by vllm/docker-run.sh
hf-cache/                             HuggingFace cache (~113 GB; openai/gpt-oss-120b lives here)
hf-cache/.hf_token                    HF Hub auth token (chmod 600)
saved-modelfiles/                     Modelfiles preserved before model deletion (phi4_compliance, gpt-oss-120b)
qdrant_data.tar     12 940 216 320 B  Qdrant data backup (2026-04-09)
qdrant_image.tar       180 882 432 B  Qdrant image tarball (2026-04-09)
start_containers.sh                   Boot-time safety net + warmup (see §11)
start_containers.sh.original     881 B Old boot script preserved for reference
docker_startup.log                    Log written by start_containers.sh
```

---

## 11. Boot & Lifecycle

`docker.service` is enabled. **Container restart on boot is now handled by
Docker's `--restart unless-stopped` policy on all four containers** (changed
2026-05-18). Reboots no longer depend on the cron script.

### Per-container recreate scripts (canonical "how to (re)build the container")

| Container | Recreate script | Notes |
|---|---|---|
| `ollama` | `/home/gravity/ollama/docker-run.sh` | Refuses to start if `/mnt/nvme` isn't mounted. Bind-mounts NVMe-backed model store. |
| `qdrant` | `/home/gravity/qdrant/docker-run.sh` | Refuses to start if `/root/qdrant_storage` is missing. API key inline (move to secret store later). |
| `ollama_wrapper` | `/home/gravity/ollama-wrapper/docker-run.sh` | Old parameterised script preserved at `docker-run.sh.original`. |
| `gravity-embedding-wrapper` | `/home/gravity/embedding-wrapper/docker-run.sh` | Preserves the `pip install + runserver` cmd. Old script at `docker-run.sh.original`. |
| `vllm_gpt_oss` | `/home/gravity/vllm/docker-run.sh` | Added 2026-05-19. Reads HF token from `/home/gravity/hf-cache/.hf_token`. Bind-mounts HF cache from Premium SSD. Port 8001 → container 8000. Image: `vllm/vllm-openai:latest` (v0.21.0). `--gpu-memory-utilization 0.85 --max-model-len 32768 --max-num-batched-tokens 1024`. |

Each script is idempotent — `docker rm -f` the existing container first, then
`docker run` a fresh one with the canonical env vars, mounts, and
`--restart unless-stopped`. Each writes to a per-container
`docker-run.log` next to the script for audit.

### Boot script — auto-restore for the daily deallocate cycle

`@reboot /home/gravity/start_containers.sh` is still in root's crontab.
The script was rewritten **twice** on 2026-05-18:

1. First as a simple safety-net + warmup (when we thought `--restart unless-stopped` was enough).
2. Then expanded to handle the daily Stop(deallocate)/Start cycle — because that wipes the local NVMe every morning and the previous version had no answer for that.

The current script runs four phases on every boot:

**Phase 1 — NVMe health + restore.**
- If `/dev/nvme0n1` has no ext4 filesystem → `mkfs.ext4 -L nvme-ephemeral`.
- `mkdir -p /mnt/nvme` to ensure the mountpoint directory exists. **This is load-bearing**: `waagent` recreates `/mnt` fresh on every Start, so any sub-directory we made yesterday is gone. Without this `mkdir`, the next mount silently fails and the restore step is skipped.
- If NVMe isn't mounted → `mount -a` (the fstab line waits for `/mnt` first thanks to the `requires-mounts-for=/mnt` option, so the race that caused the first incident on 2026-05-18 is gone).
- If NVMe mounted but the model store has fewer than 5 blobs → stop ollama, `rm -rf` the partial store, rsync everything from `/var/lib/docker/volumes/ollama/_data` (the persistent rollback on Premium SSD). Takes ~1–6 minutes depending on whether the rollback files are in the OS page cache.

**Phase 2 — Container safety net.** If any of the four containers is missing entirely (rare), run its per-container `docker-run.sh`. Otherwise Docker's `--restart unless-stopped` has already handled it.

**Phase 3 — Ensure ollama is running.** Phase 1 may have stopped it for the restore. Phase 3 brings it back.

**Phase 4 — Warm core models.** Updated 2026-05-19 to warm both:
- `gpt-oss:120b` on vLLM (port 8001) — includes a wait loop polling `/v1/models` for up to 5 min, since vLLM cold-start is 2-5 minutes after a deallocate.
- `mistral:instruct` on Ollama (via wrapper at port 11434) — the fast-chat default.
This eliminates first-request cold-load latency for both the heavy model (gpt-oss via vLLM) and the chat default (mistral via Ollama).

Both the first-version (safety-net only) and the previous-original
(parameterised `docker start` loop) are preserved as
`start_containers.sh.bak.<timestamp>` and `start_containers.sh.original`.

This means:
- **Normal in-OS reboot** (rare) → disks survive, containers auto-start, script logs "healthy" and warms both vLLM gpt-oss and mistral.
- **Daily Stop(deallocate)/Start** (every morning) → NVMe is blank; script reformats, mounts, rsyncs 24 GB from Premium SSD rollback, starts ollama, waits for vLLM, warms gpt-oss-120b on vLLM, warms mistral on Ollama. **Total ~9-10 min, fully automated.** Measured 2026-05-20: 9:22 end-to-end (boot 05:57:02 → finished 06:06:24).
- **Container missing** → safety net rebuilds it from its `docker-run.sh` (5 containers in scope: ollama, qdrant, ollama_wrapper, gravity-embedding-wrapper, vllm_gpt_oss).
- **Full VM rebuild** → run `start_containers.sh` manually after attaching the Premium SSD (which holds the rollback) and re-creating `/home/gravity/hf-cache/.hf_token`.

**Production validation status (2026-05-20)**: three full Stop(deallocate)/Start cycles completed since 2026-05-18 with the auto-restore in place. All cycles: vLLM RestartCount=0 (no crashes), 6 Ollama models restored from rollback, mistral warmed, gpt-oss accessible via vLLM with real responses. The hybrid Ollama + vLLM stack is **production-validated** across the daily deallocate cycle.

### Persistent vs ephemeral storage — the rule

| Data | Lives at | Survives daily deallocate? |
|---|---|---|
| Ollama model store (working copy) | `/mnt/nvme/ollama` (local NVMe) | ❌ wiped every morning |
| **Ollama model store (rollback / source of truth)** | **`/var/lib/docker/volumes/ollama/_data` (Premium SSD)** | ✅ persists |
| Qdrant collections | `/root/qdrant_storage` (Premium SSD) | ✅ persists |
| HuggingFace cache (embedding wrapper) | `/home/gravity/embedding-wrapper/volume-mapping/cache` (Premium SSD) | ✅ persists |
| Container logs | per-container bind mounts on Premium SSD | ✅ persists |
| Docker overlayfs | `/var/lib/docker/overlay2` (Premium SSD) | ✅ persists |
| Resource disk (`/mnt`) | `/dev/sdb1` or `/dev/sdc1` (Azure ephemeral) | ❌ wiped every morning |

**Operational rule**: anything that takes more than 5 minutes to regenerate
must live on Premium SSD. Anything else can live on NVMe for speed.

Other enabled / running services of note: `containerd`, `nvidia-persistenced`, `walinuxagent`, `chrony`, `ssh`. No nginx, no Postgres/MySQL/Redis. No `cron.d` jobs beyond Ubuntu defaults (`e2scrub_all`, `sysstat`).

---

## 12. Users & SSH

- Linux accounts with login shells: `root`, `gpu` (uid 1000)
- SSH: password auth disabled, key-only; one Azure-generated public key in `/root/.ssh/authorized_keys` (and the equivalent in `/home/gpu/.ssh/`).
- Recent logins are from `182.71.53.106` and `3.6.25.144` (operator IPs).

---

## 13. Reproducing This Setup

### Required Azure SKU

- **`Standard_NC40ads_H100_v5`** (40 vCPU EPYC, 320 GiB RAM, 1 × H100 NVL 94 GB) — exact match.
- Smaller options that *won't* work for this workload as-is:
  - `NC24ads_A100_v4` (1 × A100 80 GB) — too little VRAM if you want headroom above the current 78 GB working set.
  - Any T4/L4 SKU — wrong class entirely.
- Larger drop-in options:
  - `NC80adis_H100_v5` (2 × H100 NVL, 188 GB total VRAM) — gives room for full-precision 70B inference.

### Minimum specs

| | Required | This VM |
|---|---|---|
| GPU VRAM | ≥ 80 GB to host the current Ollama working set with headroom | 94 GB ✓ |
| System RAM | ≥ 128 GB (model loading, embedding service, Qdrant) | 314 GiB ✓ |
| vCPUs | ≥ 16 | 40 ✓ |
| OS disk | ≥ 256 GB if Ollama volume stays on `/`; otherwise ≥ 100 GB | 512 GB ✓ |
| Model volume | ≥ 256 GB (Ollama 175 G + headroom) — consider local NVMe for throughput | NVMe 3.5 T available, currently unmounted |
| OS | Ubuntu 22.04 / 24.04 LTS | 24.04.3 ✓ |
| NVIDIA driver | ≥ 560 (for CUDA 12.6+); 570-server here | 570.211.01 ✓ |
| Docker + nvidia-container-toolkit | required | ✓ |

### Largest model that fits

- Llama 3.1 70B 4-bit (`unsloth/…-bnb-4bit`) is ~40 GB on disk; live VRAM with KV cache fits comfortably under 94 GB.
- **Full-precision (bf16) 70B will not fit on a single H100 NVL** — the bnb-4bit variant is the only practical 70B option here. If you need bf16 70B inference, move to a 2× H100 SKU.

### Bring-up checklist on a fresh VM

1. **Create VM**: `Standard_NC40ads_H100_v5`, Ubuntu 24.04 LTS, attach a 512 GB Premium SSD for `/`, and at least one large managed disk *or* the local NVMe of the SKU for model storage.
2. **Install NVIDIA driver 570-server** and `nvidia-container-toolkit`:
   ```bash
   sudo apt-get install -y nvidia-driver-570-server nvidia-container-toolkit
   sudo systemctl enable --now nvidia-persistenced
   ```
3. **Install Docker** and write `/etc/docker/daemon.json`:
   ```json
   { "runtimes": { "nvidia": { "args": [], "path": "nvidia-container-runtime" } } }
   ```
4. **Mount the local NVMe for the Ollama model store.**
   ```bash
   sudo mkfs.ext4 -L nvme-ephemeral /dev/nvme0n1
   sudo mkdir -p /mnt/nvme
   sudo mount /dev/nvme0n1 /mnt/nvme
   echo 'LABEL=nvme-ephemeral /mnt/nvme ext4 defaults,nofail,x-systemd.device-timeout=5s 0 0' | sudo tee -a /etc/fstab
   sudo mkdir -p /mnt/nvme/ollama
   ```
5. **Create the four containers** (the original `docker run` commands aren't in `start_containers.sh` — see Open Questions). The `ollama` container has a documented recreate script at `/home/gravity/ollama/docker-run.sh`. Approximate shape:
   ```bash
   # ollama — runs the script (env vars + bind mount + restart policy baked in)
   /home/gravity/ollama/docker-run.sh

   # qdrant — Qdrant data stays on Premium SSD (stateful, not on ephemeral NVMe)
   docker run -d --name qdrant -p 6333:6333 -p 11435:6333 \
     -v /root/qdrant_storage:/qdrant/storage \
     -e QDRANT__SERVICE__API_KEY=<your-key> \
     --restart unless-stopped qdrant/qdrant

   docker run -d --name ollama_wrapper -p 11434:11434 \
     --restart unless-stopped gravity2023/ollama_wrapper:1.1.0.0

   docker run -d --name gravity-embedding-wrapper -p 11436:11436 \
     -v /home/gravity/embedding-wrapper/volume-mapping/cache:/app/.cache/huggingface \
     --restart unless-stopped gravity2023/gravity-embedding-wrapper:1.1.0.0
   ```
6. **Restore data**:
   - Re-pull Ollama models (`docker exec ollama ollama pull <model>`) or restore the `ollama` volume from backup.
   - `docker load < /home/gravity/qdrant_image.tar` and restore `qdrant_data.tar` into the Qdrant volume.
7. **Install the boot script**: copy `start_containers.sh` to `/home/gravity/`, make it executable, add `@reboot /home/gravity/start_containers.sh` to root's crontab.
8. **Point DNS** (`ollama.gravity.ind.in` or equivalent) at the VM's public IP and configure the Azure NSG to allow only the ports that should be public (likely `:443` via a load balancer / App Gateway — ports `11433–11436` and `6333` should not be open to the internet directly).

---

## 14. Next Steps / Recommendations

Ordered roughly by impact and risk:

1. **Apply pending updates and reboot.**
   - 105 packages outstanding, 15 are security; the kernel is asking for a restart.
   - Schedule a maintenance window — every container is on this single host.

2. **Mount the 3.5 TB local NVMe and move heavy data onto it. — ✅ done 2026-05-18 (Ollama volume only).**
   - Ollama model store migrated from `/var/lib/docker/volumes/ollama` (Premium SSD, 262 MB/s) to `/mnt/nvme/ollama` (local NVMe, 5.1 GB/s).
   - Result: `gpt-oss:120b` cold load went from **380 s → 6.17 s (62× faster)**. See §7 "Cold-load impact" table.
   - NVMe is confirmed `Microsoft NVMe Direct Disk v2` — ephemeral, wiped on VM deallocate but survives reboots. Acceptable for Ollama models (re-pullable from Ollama Hub). `/etc/fstab` uses `nofail` so boot doesn't hang if NVMe is missing. `docker-run.sh` includes a `mountpoint -q /mnt/nvme` safety check.
   - **Still pending**: optionally move `/root/.cache/huggingface` (79 G) to NVMe too. Not yet done — embedding wrapper's cold-start would benefit but it's a smaller win.
   - **Do NOT move** `/root/qdrant_storage` to NVMe — Qdrant data is stateful and cannot survive a VM deallocate.

3. **Reconcile the dual NVIDIA driver install.**
   - Purge the leftover 550-series packages once you've confirmed nothing depends on them: `sudo apt purge nvidia-*-550-server libnvidia-compute-550 && sudo apt autoremove`.

4. **Replace the `@reboot` cron with proper restart policies. — ✅ fully done 2026-05-18.**
   - All four containers (`ollama`, `qdrant`, `ollama_wrapper`, `gravity-embedding-wrapper`) now run with `--restart unless-stopped`.
   - Each has a canonical recreate script — see §11.
   - The `@reboot` cron entry is kept but the script content was rewritten to act only as a safety net (recreate any missing container) plus a mistral warmup. Boot is no longer dependent on it.
   - A `docker-compose.yml` would still be a nice consolidation if you ever migrate to compose, but the per-container scripts already make every run-time argument visible and version-controllable.

5. **Lock down ingress.**
   - Confirm that Azure NSG only allows `:22` from operator IPs and `:443` from clients (presumably via an upstream load balancer that terminates TLS and forwards to `:11434`/`:11436`).
   - The host has no TLS — exposing `:11434` directly to the internet sends API calls in clear text.

6. **Stop running Django `runserver` in production.**
   - Both wrappers use `manage.py runserver`, which is explicitly not for production. Switch to `gunicorn` (or `uvicorn` if any view is async) inside the wrapper images.
   - Bake `pip install` into the image rather than running it at container start.

7. **Pin Docker client to match the server.**
   - Client 29.1.3 vs server 27.5.1 works today but invites surprises. Either upgrade the engine or downgrade the CLI to 27.x.

8. **Document the missing `docker run` invocations.**
   - Capture the exact arguments (`--gpus`, mounts, env) for each container into a compose file or a runbook so a rebuild doesn't depend on tribal knowledge or this snapshot.

9. **Establish a backup cadence.**
   - The Qdrant tarballs in `/home/gravity/` are from 2026-04-09 — almost a month old as of this snapshot. Automate snapshots of the `ollama` volume and the Qdrant data, and ship them off-box.

10. **Monitor.**
    - `nvidia-smi` and container health checks are not currently exported anywhere. Even a basic Prometheus + node-exporter + dcgm-exporter stack would give visibility into VRAM saturation, GPU temp, and container restarts.

11. **Ollama KV cache tuning — ✅ done 2026-05-18.**
    - Added `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_CONTEXT_LENGTH=32768`, and `OLLAMA_LOAD_TIMEOUT=15m` to the `ollama` container.
    - Result: `llama3.3` KV cache dropped from 40 GiB (with 2 GiB CPU spill) to 5.4 GiB, all on GPU. `mistral:instruct` from 4 GiB → 2.18 GiB. See §7 for the impact table.
    - All Ollama settings now live in `/home/gravity/ollama/docker-run.sh` — edit there to change.

13. **Unused-model cleanup — ✅ done 2026-05-18.**
    - Removed `llama3.3:latest` (42 GB), `deepseek-ocr:latest` (6.7 GB), `glm-4.7-flash:latest` (19 GB) — ~67 GB freed.
    - 8 models remain. See §7 "Models removed 2026-05-18" for the list.
    - Periodically audit — `docker exec ollama ollama ps` shows what's actively in VRAM; anything that hasn't been hit in months is a deletion candidate.

12. **Decide the wrapper cloud-routing policy.**
    - The `ollama_wrapper` silently routes `gpt-oss:120b` (and at least `mistral-large-3:675b`) to Ollama Cloud (ollama.com) rather than the local `ollama` daemon. Discovered 2026-05-18.
    - Three options (see `architecture-and-placement.md` §14): keep the split and document it; force everything local by patching the wrapper image; or expose the routing explicitly via response headers.
    - Even if you stay on cloud, audit the actual usage against your ollama.com plan — the log shows historical 403 ("subscription required") errors, meaning some requests are being rejected at the upstream.

---

## 15. Open Questions

- ~~**Exact `docker run` arguments for the three remaining containers**~~ ✅ **Resolved 2026-05-18**: every container now has a canonical recreate script under `/home/gravity/<name>/docker-run.sh`. See §11 for the table of locations.
- **Public ingress path.** No nginx/caddy/traefik on this host, so TLS termination and the `ollama.gravity.ind.in` → `:11434` mapping likely happen on an Azure Load Balancer or Application Gateway — that config is not visible from this VM.
- ~~**NVMe persistence guarantee.**~~ ✅ **Resolved 2026-05-18**: `Microsoft NVMe Direct Disk v2` is ephemeral (wiped on VM deallocate) but survives reboots. Acceptable for Ollama models (re-pullable). fstab uses `nofail`. Models took ~5 min to rsync over once; if NVMe is wiped, `ollama pull` from the Hub takes ~30–60 min for the current 8-model set.
- **Wrapper cloud-routing rules.** Confirmed 2026-05-18 that `ollama_wrapper` proxies `gpt-oss:120b` (and `mistral-large-3:675b`) to Ollama Cloud rather than local Ollama. The exact list of cloud-routed models and the routing logic live inside `gravity2023/ollama_wrapper:1.1.0.0` at `/app/OllamaWrapper/Wrapper/views.py` (not extractable without docker cp / image inspection). Until that's documented, callers can't reliably predict which path their request will take.
- **Whether `gpt-oss:120b` should be kept warm locally.** Cold load is 380 s (62 GB of weights). Steady-state warm inference is ~155 tokens/sec on the H100. With current q8_0 settings it fits alongside `mistral` (74 GB total) but blocks `llama3.3` from co-residency. Pin it (`OLLAMA_KEEP_ALIVE=-1`) if low-latency local gpt-oss matters; leave off-resident if you're happy with the cloud path through the wrapper.
