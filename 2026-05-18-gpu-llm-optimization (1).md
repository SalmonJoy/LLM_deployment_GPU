# Learnings: Optimizing a Self-Hosted LLM Stack on Azure H100

**Date:** 2026-05-18
**Host:** `GPU-1` — `Standard_NC40ads_H100_v5` (Azure CentralIndia, 1× H100 NVL 94 GB)
**Stack:** Ollama + Qdrant + two Django wrappers in Docker
**Audience:** AI/ML engineers who manage self-hosted inference; AI assistants doing similar diagnostic work in the future.

This document captures one day of diagnostic + optimization work end-to-end —
the initial state, the problems found, the diagnostic reasoning, the
fixes applied, the mistakes made, and the principles that generalise.
Read it as a case study, not as a tutorial.

---

## Table of contents

1. [Starting state](#1-starting-state)
2. [Phase 1 — building the picture](#2-phase-1--building-the-picture-from-zero-knowledge)
3. [Phase 2 — the KV cache spill](#3-phase-2--the-kv-cache-spill-an-easy-win-hiding-in-plain-sight)
4. [Phase 3 — the cloud-routing mystery](#4-phase-3--the-cloud-routing-mystery-the-best-debugging-of-the-day)
5. [Phase 4 — the NVMe migration](#5-phase-4--the-nvme-migration-the-62%C3%97-cold-load-win)
6. [Phase 5 — restart policies and recreate scripts](#6-phase-5--restart-policies-and-recreate-scripts)
7. [Mistakes I made](#7-mistakes-i-made-honest-catalog)
8. [The numbers — before and after](#8-the-numbers--before-and-after)
9. [Lessons for AI engineers](#9-lessons-for-ai-engineers)
10. [Reusable patterns and checklists](#10-reusable-patterns-and-checklists)
11. [Appendix A — the daily-deallocate disaster (added later same day)](#appendix-a-the-daily-deallocate-disaster-afternoon-of-2026-05-18)

---

## 1. Starting state

The user had a working LLM stack but had never deeply audited it. The
goal was open-ended: understand what was deployed, document it, identify
inefficiencies, and improve latency without hurting quality.

### What was on the box (point-in-time, before any changes)

- **VM**: `Standard_NC40ads_H100_v5`, Ubuntu 24.04.3, AMD EPYC 9V84 (40 vCPU), 314 GiB RAM, 1× NVIDIA H100 NVL (94 GB VRAM).
- **Disks**: 512 GB Premium SSD root, 512 GB empty data disk, **3.5 TB local NVMe completely unmounted**, 128 GB Azure ephemeral resource disk.
- **Containers** (all four with `RestartPolicy: "no"`):
  - `ollama` (`ollama/ollama:latest`) on port 11433 — the only GPU container.
  - `ollama_wrapper` (Django, `gravity2023/ollama_wrapper:1.1.0.0`) on 11434 — assumed to be a thin proxy.
  - `gravity-embedding-wrapper` (Django, `gravity2023/gravity-embedding-wrapper:1.1.0.0`) on 11436.
  - `qdrant` (`qdrant/qdrant`) on 6333 + 11435.
- **Boot recovery**: a single cron entry `@reboot /home/gravity/start_containers.sh` that ran `docker start` on each container.
- **Ollama configuration**: only `OLLAMA_NUM_PARALLEL=1` and `OLLAMA_KEEP_ALIVE=-1` set. Everything else at Ollama defaults.
- **Models**: 11 Ollama models totalling 185 GB on disk. HF cache 79 GB. `OLLAMA_NUM_PARALLEL=1` (single concurrent inference), `OLLAMA_KEEP_ALIVE=-1` (models pinned in VRAM forever once loaded).

### The implicit assumptions we walked in with

We assumed (correctly or not):
- The wrapper is a thin proxy. ❌ Wrong — see Phase 3.
- The H100 was the bottleneck. ❌ Mostly wrong — disk and config were the real bottlenecks.
- All models were being served locally. ❌ Wrong — gpt-oss was cloud-served.
- The system was already configured optimally for inference. ❌ Wrong — flash attention was off, KV cache was fp16, no context cap.

**Lesson: never trust the assumed model of an inherited system. The first hour should be measurement, not optimization.**

---

## 2. Phase 1 — building the picture from zero knowledge

We started with no access to the box (only the user could run commands).
The right move was a **single comprehensive diagnostic capture**, not
fishing for the bug we suspected.

### The capture script we sent first

15 numbered sections covering: host/OS, CPU/RAM/disks, GPU/CUDA, Azure
VM metadata, Python/ML runtime, Ollama, Docker/containers, other LLM
servers, web/reverse proxy, databases/vector stores, networking,
storage footprint, systemd, cron, users/SSH.

It returned **~830 lines of output**. Worth every line.

### Key findings that shaped everything else

| Finding | Why it mattered |
|---|---|
| `nvidia-smi` showed 78 GB VRAM in use by a single Ollama runner | First clue something big was loaded (later found to be llama3.3) |
| `ollama list` showed 11 models, 185 GB total | Cannot all fit; eviction must be happening — set up the KV cache investigation |
| Wrapper `requirements.txt` not present on host (source in image) | Couldn't read wrapper code directly; had to rely on `docker inspect` and log files |
| `OLLAMA_REMOTES:[ollama.com]` in env | Later turned out to be the smoking gun for cloud routing |
| 3.5 TB NVMe **unmounted** | Largest single missed optimization, but we didn't know it yet |
| All containers `RestartPolicy: "no"`, cron-driven boot | Boot was fragile but functional |
| Two NVIDIA driver series installed (550 + 570) | Minor mess; flagged but not fixed |

### Lesson: parallel data gathering beats serial questioning

The instinct is to ask one question, get an answer, ask the next. With a
remote operator pasting outputs, that's ~5 minutes of round-trip per
question. A pre-built capture script gets you the whole picture in one
round-trip. Build the capture from the symptoms you suspect AND the
ones you don't — the unexpected fields often turn out to matter most.

---

## 3. Phase 2 — the KV cache spill (an easy win hiding in plain sight)

User reported llama3.3 felt sluggish. We asked Ollama what its config
was. The output was the smoking gun:

```
OLLAMA_FLASH_ATTENTION:false
OLLAMA_KV_CACHE_TYPE: (empty, default = fp16)
OLLAMA_CONTEXT_LENGTH:0 (default = model max)
```

And then in the logs:

```
llama_kv_cache:        CPU KV buffer size =  2048.00 MiB
llama_kv_cache:      CUDA0 KV buffer size = 38912.00 MiB
llama_kv_cache: size = 40960.00 MiB (131072 cells, 80 layers, 1/1 seqs),
                K (f16): 20480.00 MiB, V (f16): 20480.00 MiB
```

**Translation:**
- llama3.3 has 80 transformer layers and a 128 K context window.
- At fp16 KV cache, that's 40 GiB of KV cache alone — eating almost half the H100's VRAM **for just the cache**.
- **2 GiB of that cache had spilled to system RAM** (CPU KV buffer). Every attention computation was pulling tensors back and forth over PCIe per token. That's the "sluggish" feeling.

### The fix (one-line knowledge buried in a deep doc)

1. `OLLAMA_FLASH_ATTENTION=1` — enables flash attention. On Linux, Ollama defaults to **off**, which is a stale default for any Hopper-class GPU.
2. `OLLAMA_KV_CACHE_TYPE=q8_0` — quantises KV cache to int8. Halves the cache size with effectively zero quality loss for q8_0 (this is widely-known engineering folklore at this point — drop to q4_0 if you want more savings, but quality starts to matter).
3. `OLLAMA_CONTEXT_LENGTH=32768` — caps context window. Almost nobody actually uses 128K tokens of context; capping at 32K is sane for chat/RAG and quarters the KV cache.

**Critical gotcha:** `OLLAMA_KV_CACHE_TYPE=q8_0` is **silently ignored unless `OLLAMA_FLASH_ATTENTION=1` is also set**. If you set the cache type without flash attention, Ollama falls back to fp16 without warning. Found this in the Ollama source — not in any default docs you'd find first.

### Measured impact

| Model | KV cache before (fp16, native ctx) | KV cache after (q8_0, 32K) | Reduction |
|---|---|---|---|
| `mistral:instruct` (32 layers, 32K) | 4 096 MiB | 2 176 MiB | 46 % |
| `llama3.3` (80 layers, 128K native) | 40 960 MiB + 2 GiB CPU spill | 5 440 MiB, all on GPU | **87 %**, no spill |

Total transformative effect on what could co-reside:
- Before: llama3.3 alone barely fit (with spill). mistral + llama3.3 was impossible.
- After: mistral + llama3.3 fit together (60 GB resident, 35 GB headroom). mistral + gpt-oss:120b fit together (74 GB resident, 20 GB headroom).

### Generalisable lesson

For any LLM inference engine you operate, on day 1 you should know:

```
VRAM = sum(weights of all resident models) + KV cache + activation buffers

KV cache per model = num_layers × context_cells × 2 (K and V) × bytes_per_element × num_parallel_slots
```

If you can't write that formula for your stack, you're flying blind on
sizing. The four knobs are: which models are resident, the context cap,
the KV cache precision, and the parallelism. Tune all four.

---

## 4. Phase 3 — the cloud-routing mystery (the best debugging of the day)

This was the longest debugging arc, and the most embarrassing for me
(the AI). It's worth telling in full because the diagnostic method
generalises to almost any "this works but I don't know how" situation.

### Symptom

The user wanted gpt-oss:120b loaded alongside mistral. I gave them a
warmup curl pointing at `http://localhost:11434/api/chat`. The response
came back in **0.93 seconds** with `done_reason: "stop"`.

For a 120 B parameter model on a single H100, 0.93 s for a cold load
plus inference is **impossible**. The weights alone are 65 GB on disk.

### What I assumed first (wrong)

> "Maybe gpt-oss uses Ollama's new engine, which spawns a runner that
> exits between requests. That's why it's not in `ollama ps`."

Plausible — the response had `*_duration: null` in the JSON, which I
recognised as a "different runner backend" signature. But it didn't
explain the 0.93 s. So I asked for more data.

### Second hypothesis (also wrong)

> "Maybe gpt-oss is cloud-served. The env shows `OLLAMA_REMOTES:[ollama.com]`."

I asserted this confidently and the user almost went with it. But it
turned out to be wrong because:

- `ollama show gpt-oss:120b` showed a normal local Modelfile pointing at a local blob.
- A 61 GB blob with the right hash existed on disk.

I had to walk it back and apologise. **This was a real mistake** —
`OLLAMA_REMOTES` is the registry to **pull from**, not a flag that
routes inference. I jumped on a plausible-sounding clue without
checking what the env var actually did.

### Third hypothesis (also wrong)

> "Maybe it's running on CPU. The AMD EPYC 9V84 has enough memory bandwidth that MoE inference at MXFP4 could hit ~88 tok/s on CPU."

This was numerically defensible. I asked for `top` during inference.
The CPU was **0 % busy**. Load average was 0.09. Not CPU either.

### Fourth hypothesis (correct, finally)

> "Wait — every test I've sent has used port 11434. That's the `ollama_wrapper`, not Ollama directly. Ollama is on port 11433. What if the wrapper routes some models cloud-side?"

The test that proved it: hit Ollama **directly** on port 11433:

```
gpt-oss cold load via :11433: 379.9 seconds
```

That's a real 6+ minute cold load — 65 GB of weights being read from
Premium SSD. The local model **works**, it just takes ages to load.

Then we read the wrapper log:

```
ollama._types.ResponseError: this model requires a subscription,
  upgrade for access: https://ollama.com/upgrade (status code: 403)
ollama._types.ResponseError: model 'mistral-large-3:675b' is
  temporarily overloaded ... (status code: 503)
```

`mistral-large-3:675b` doesn't exist anywhere locally. That error
**came back from ollama.com**. The wrapper's Python `ollama` client
silently proxies certain models to Ollama Cloud while serving others
locally.

### What we learned about the diagnostic method

The key shift in approach was **trace the request, don't trust the names**.
Once I asked "what port am I actually hitting?" the mystery dissolved
in one command. All the earlier hypotheses (new engine, cloud, CPU)
were guesses about *internal* behaviour. The real bug was at the
*network boundary* I hadn't questioned.

Decisive diagnostic pattern, for future reference:

```
A bug where "the thing works but produces impossible results" almost
always has one of three explanations:
  1. You're not measuring what you think you're measuring.
  2. The request is going somewhere you don't know about.
  3. A cache is returning a stale or fake result.

Test all three by:
  1. Snapshot processes AND ports AND network connections during the bug.
  2. Hit the lowest layer of the stack directly. (Port 11433 vs 11434.)
  3. Look at the request/response with -v / no -o /dev/null. Read the bytes.
```

### The fix (or non-fix)

We didn't change the routing — the user decided to keep the cloud path
for gpt-oss, since it's free of the 6-minute cold-load cost. But we
**documented it explicitly** in `architecture-and-placement.md` §14:
the wrapper is a hybrid local/cloud router, not a thin proxy. Three
options were laid out for the user (keep, force local, make explicit).

**Generalisable lesson:** **API endpoints lie.** A request to
`http://localhost:11434/api/chat` looks like it's local. But "localhost"
is just a docker-proxy that forwards to a container that has its own
internal logic. Always confirm where compute is actually happening
before optimizing.

---

## 5. Phase 4 — the NVMe migration (the 62× cold-load win)

After phase 3, we had a baseline cold-load measurement: **gpt-oss
takes 380 seconds to load from disk on Premium SSD**. That was the
biggest single latency cost in the system. The user asked: "what
can I improve next without hurting quality?"

### The hypothesis

`dd` on the two disks:

```
/dev/sdb (Premium SSD, current model location):  262 MB/s
/dev/nvme0n1 (local NVMe, unmounted):            5 100 MB/s
```

19.5× faster on the disk benchmark. If cold-load was disk-bound, NVMe
should give ~20× speedup → gpt-oss cold load 380 s → ~20 s.

### The migration (10 minutes of work)

1. `mkfs.ext4 -L nvme-ephemeral /dev/nvme0n1` — format the disk.
2. `mount /dev/nvme0n1 /mnt/nvme` + add `/etc/fstab` line with `nofail` so a missing NVMe never blocks boot.
3. `rsync -aHAX` the 112 GB Ollama volume from Premium SSD to `/mnt/nvme/ollama`. Took 5 min 26 s — bottlenecked by the Premium SSD's read speed (~349 MB/s sustained).
4. Stop ollama container, do a final incremental rsync (caught zero changes — models don't mutate during a session), update the recreate script to bind-mount NVMe instead of the named Docker volume, restart.
5. **Downtime: 69 seconds total.** Wrappers and Qdrant kept running.

### The result

| Operation | Before (Premium SSD) | After (NVMe) | Speedup |
|---|---|---|---|
| `gpt-oss:120b` cold load (65 GB) | **380 s** | **6.17 s** | **62×** |
| `mistral:instruct` cold load (4.1 GB) | 6 s | 6 s | 1× (setup-bound, not disk-bound) |
| Effective throughput during cold load | ~170 MB/s | ~10 GB/s | ~60× |

The result **beat the raw `dd` benchmark** (62× vs 19.5×). Why?

- The `dd` test was sequential read into `/dev/null` — single-threaded.
- Real cold load is **mmap with readahead, pipelined with GPU DMA copy over PCIe Gen5**. The kernel reads pages ahead while the GPU is busy copying earlier pages. They overlap.
- The bottleneck shifted from disk seek to PCIe bandwidth (which is much higher).

**This is a recurring pattern.** Real-world I/O performance with mmap +
readahead can be 2–3× the naive single-threaded benchmark when the
consumer is fast enough to keep up. PCIe Gen5 with H100 is fast enough.

### Important caveat: NVMe persistence

`Microsoft NVMe Direct Disk v2` is **ephemeral** — wiped on:
- VM deallocate (full stop)
- Some live-migrations

**Not** wiped on:
- Reboots
- Container restarts
- apt upgrades

We made the trade explicit: put **re-creatable storage** on NVMe (Ollama
models, downloadable from Ollama Hub; HF cache, downloadable from HF
Hub). Keep **stateful storage** on Premium SSD (Qdrant collections —
that's customer data and cannot be re-derived).

The recreate script for `ollama` includes a `mountpoint -q /mnt/nvme`
check that **refuses to start** if NVMe is missing. This prevents
Ollama from silently launching with an empty model directory on a
wiped NVMe — better to fail loudly than serve "model not found".

### Smaller win the same day: deleted three unused models

After understanding the model inventory, the user identified three
unused models:
- `llama3.3:latest` (42 GB)
- `deepseek-ocr:latest` (6.7 GB)
- `glm-4.7-flash:latest` (19 GB)

Total: **~67 GB freed**. Reduces the rsync to NVMe (faster), reduces
the surface area for accidental cold loads, and frees disk.

### Generalisable lesson

**Disk speed is usually under-appreciated in LLM ops.** Cold-load time
scales linearly with model size at the disk bandwidth. A 70B model at
~40 GB is 2–3 minutes on slow disk, seconds on NVMe. For a system that
operators reboot quarterly, this is the difference between "instant
recovery" and "the LLM is down for 10 minutes while it warms up."

**The single highest-ROI question to ask of any inference host:** What
disk are the weights on, and how fast is it? If the answer is
"Premium SSD" or "EBS gp3" or anything network-attached, you probably
have a 5–50× win available by moving to local NVMe.

---

## 6. Phase 5 — restart policies and recreate scripts

The boot mechanism was a cron job (`@reboot /home/gravity/start_containers.sh`)
that did `docker start` on each named container. This worked, but had
two problems:

1. **No `--restart` policy on any container.** If a container crashed,
   it stayed dead until next reboot. The cron papered over this for
   reboots only.
2. **No documented `docker run` invocations.** The current container
   config existed only as state inside `docker inspect`. Rebuild meant
   reverse-engineering from inspect output.

### The pattern we adopted

For each container, a canonical recreate script at
`/home/gravity/<container-name>/docker-run.sh` that:

1. Has a header comment explaining the container and any caveats.
2. Defines the env vars, volumes, ports, and **`--restart unless-stopped`** inline.
3. Idempotently `docker rm -f`s any existing container, then `docker run`s a fresh one.
4. Logs to a per-container `docker-run.log` for audit.
5. Includes a safety check where relevant (e.g., `ollama` script refuses to start if `/mnt/nvme` isn't mounted).

The boot cron script was rewritten as a **safety net**: it only acts if
a container is *missing entirely*, otherwise lets Docker's restart
policy handle the boot. It still warms `mistral:instruct` after every
boot.

### Why this matters beyond aesthetics

Three real benefits:

1. **Reproducibility.** The next person who needs to rebuild the host
   doesn't have to read `docker inspect`. They run four scripts.
2. **Single source of truth.** Want to change `OLLAMA_KV_CACHE_TYPE`?
   Edit the script, re-run it. No risk of state drift between "what
   the container has" and "what we think the container has."
3. **Disaster recovery.** Full VM rebuild from scratch is now one
   safety-net script call. Models re-download from Ollama Hub
   automatically if NVMe was wiped.

### What we deliberately did NOT do

A `docker-compose.yml` would consolidate all four containers into one
file. It's the canonical "next step." But:

- We didn't have time to validate it end-to-end.
- The per-container script pattern matched the user's existing site
  conventions (they already had similar scripts for the wrappers).
- Each script is independently runnable for rebuild/debugging.

Compose is a strict improvement, but each script is already a
strict improvement over the original "tribal knowledge inside inspect"
state. Take the win you can ship.

### Generalisable lesson

**Infrastructure should be expressible as code, even if it's not
running as code.** A bash script that mirrors a `docker run` is not as
good as a compose file or a Terraform spec. But it's much better than
nothing. **Whatever level of formalisation you can sustain is the
right level.**

---

## 7. Mistakes I made (honest catalog)

The AI assistant (me) made several wrong calls during this work. Worth
listing them explicitly so future AI engineers can recognise the
pattern.

### 1. Asserted cloud routing too early

When I saw `OLLAMA_REMOTES:[ollama.com]` in the env, I asserted gpt-oss
was being cloud-served. The user almost acted on that. I had to walk it
back when `ollama show` revealed a local Modelfile.

**Root cause:** I matched a pattern (env var name with `REMOTES`,
suspicious null fields) without checking what the env var actually
controlled. `OLLAMA_REMOTES` is the registry to **pull models from**,
not a flag that decides where inference runs.

**Future rule:** Before asserting a config value causes behaviour X,
read the source / docs to confirm that's what the value does. Don't
infer from name.

### 2. Assumed sub-millisecond RTT inside an Azure AZ

I said in an early plan that intra-AZ RTT would be < 1 ms. The actual
measurement showed ~2 ms over public DNS. Still negligible for this
workload, but my estimate was off by 2–4×.

**Root cause:** I assumed VNet-internal routing on public DNS, which
isn't how Azure routes by default. Traffic goes through the edge.

**Future rule:** When measuring network latency to bound a decision,
specify *which path* the traffic takes (private IP vs DNS) and measure
that exact path.

### 3. Overstated the VRAM savings for mistral

When the KV cache tuning landed, I said mistral would drop from 10 GB
to ~6 GB. It actually dropped to 8.4 GB. I conflated "KV cache halved"
with "total VRAM halved." Only the cache halved; weights and scratch
buffers stayed the same.

**Root cause:** Quick mental math without writing the formula. The
fixed costs (weights, compute buffers) don't shrink with KV cache
tuning; only the cache portion does.

**Future rule:** When reporting savings, decompose the total. Tell the
user: "KV cache went from 4 GB to 2 GB, weights stay 4 GB, total
goes from 10 GB to 8.4 GB."

### 4. Under-estimated the NVMe speedup

I predicted 5–15× cold-load improvement. Actual was 62×. I missed the
mmap + readahead + PCIe Gen5 pipelining interaction.

**Root cause:** I sized the win based on the raw `dd` benchmark.
mmap-driven I/O can dramatically outperform sequential reads when the
consumer (GPU DMA) can keep up.

**Future rule:** Disk benchmarks measure a single dimension. Real
workload performance can be higher (pipelined) or lower (random small
reads) than the benchmark. Always confirm with a real workload test.

### 5. Under-estimated the embedding wrapper cold-start

I said the embedding wrapper would take 30–60 s to come up post-recreate.
It actually took ~110 s. The `pip install` step was slower than I expected.

**Root cause:** Didn't time `pip install` separately. I knew it ran at
container start; didn't budget for its actual duration.

**Future rule:** If "X happens during container start", time X
specifically before estimating end-to-end startup time.

### 6. The "I see only one runner" snapshot timing error

When investigating the gpt-oss puzzle, I asked the user to snapshot
processes during inference. We snapshotted **at the wrong moment** —
after the request had finished — and saw "only mistral's runner." I
took that as evidence gpt-oss was cloud-served, when actually we just
missed the window.

**Root cause:** Async / time-sensitive observations need controlled
timing. We didn't pipeline the snapshot with the in-flight request
correctly.

**Future rule:** When debugging time-dependent behaviour, structure the
test as: start the work in background → sleep for a known interval →
snapshot → wait for completion. Don't snapshot first then run.

### Pattern across all six

All six errors share one shape: **I projected the cheaper-to-think
conclusion ahead of the data**. Pattern matching on partial signals
felt faster but cost more time when it was wrong.

For complex distributed systems, the disciplined approach is:
**collect data → form hypothesis → state hypothesis as a falsifiable
prediction → test → only then conclude**. I skipped steps 2–4 multiple
times in this session and paid for it.

---

## 8. The numbers — before and after

### Configuration

| Setting | Before | After |
|---|---|---|
| Flash attention | off (default) | **on** |
| KV cache precision | fp16 (default) | **q8_0** |
| Context cap | unlimited (model max) | **32 768** |
| `OLLAMA_LOAD_TIMEOUT` | 5 min (default) | **15 min** |
| Restart policy (all 4 containers) | none | **`unless-stopped`** |
| Ollama model storage | Premium SSD | **Local NVMe (5 GB/s)** |

### Performance

| Metric | Before | After | Change |
|---|---|---|---|
| `gpt-oss:120b` cold load | 380 s | **6.17 s** | **62× faster** |
| `mistral:instruct` cold load | 6 s | 6 s | (setup-bound, unchanged) |
| `llama3.3` KV cache | 40 GiB (with 2 GiB CPU spill) | 5.4 GiB on GPU only | **87 % smaller, no spill** |
| `mistral` KV cache | 4 GiB | 2.18 GiB | 46 % smaller |
| Max simultaneous models | llama3.3 alone (with spill) | mistral + gpt-oss-120b (74 GB) | qualitative win |
| Disk space used by models | 175 GB / 11 models | **112 GB / 8 models** (67 GB freed) | 36 % less |

### Operational

| Aspect | Before | After |
|---|---|---|
| Container restart on crash | manual / wait for reboot | automatic |
| `docker run` config | tribal knowledge in `docker inspect` | 4 version-controllable scripts |
| Boot dependency | cron-only | Docker restart policy + cron safety net |
| Wrapper routing path | undocumented, surprising | explicitly documented in `architecture-and-placement.md` §14 |
| NVMe utilisation | 0 GB / 3.5 TB (idle) | 112 GB / 3.5 TB (Ollama models) |

### Time spent

- Total elapsed: ~one work day (with extensive step-by-step pacing).
- Largest single time block: phase 3 (cloud routing mystery, ~90 min including wrong turns).
- Highest ROI block: NVMe migration (~15 min for 62× speedup).
- Documentation: the user has three persistent docs (`azure-llm-setup.md`, `architecture-and-placement.md`, this learnings doc) that didn't exist 24 h ago.

---

## 9. Lessons for AI engineers

These are the transferable principles, generalised beyond this stack.

### About diagnostics

1. **Build a comprehensive capture script before optimizing.** One long
   capture is worth ten serial questions. Include things you don't yet
   suspect.

2. **Trace requests end-to-end. Port numbers matter.** "localhost:X"
   means "talk to whatever's bound to X right now." That can be a
   proxy, a wrapper, a router, or the thing you actually want.
   `docker inspect` on the listening container is the lowest-cost way
   to know which.

3. **Time-sensitive observations need controlled timing.** Background
   the work, sleep a known interval, snapshot. Don't snapshot first.

4. **When something seems impossible, you're measuring the wrong
   thing.** Generation at 88 tok/s for a 120B model on idle CPU was
   impossible; we kept testing until we realised the request never
   reached the box. The first instinct should be "what am I actually
   measuring?" not "this hardware is magical."

5. **`*_duration: null` in API responses is a fingerprint.** Different
   inference backends populate telemetry differently. Watch for it.

### About inference engine tuning (Ollama-specific but mostly generalises)

6. **Flash attention should be ON on any modern GPU.** Hopper, Ada,
   Ampere — all support it. The savings are 1.5–2× attention speed
   and significantly less activation memory.

7. **Quantised KV cache (q8_0) is free quality-wise** for most
   models — adopt it. q4 cache is more aggressive and quality starts
   to matter; test it on your workload.

8. **The KV cache formula:**
   `bytes ≈ num_layers × context × 2 × bytes_per_element × parallel_slots`.
   Llama-3-70B at 128K context fp16 single-slot is ~40 GB. Drop to
   32K + q8 single-slot, you're at ~5 GB. **The defaults are often
   wrong for your workload.**

9. **Context cap is the lever nobody adjusts.** Almost no real
   workload uses the full model context. Capping at 32K or 16K can
   free 4–8× of KV cache. Pass `num_ctx` per request to override
   when you genuinely need long context.

10. **Always set a load timeout > the largest model's expected cold
    load.** Default 5 minutes can time out on big-model first-load and
    cascade into client retries.

### About storage and infrastructure

11. **Disk speed dominates cold-load time.** Premium SSD vs local NVMe
    is often 20× on the disk benchmark and 60× on real cold load
    (mmap + readahead pipelining). If your weights aren't on the
    fastest disk available, that's your first move.

12. **Tier your storage explicitly.** Re-creatable data (model
    weights, caches, container images) → ephemeral fast storage.
    Stateful data (databases, vector stores, customer data) → durable
    storage. Mixing them is the worst of both worlds.

13. **`mountpoint -q` guards** prevent silent half-empty operation.
    If your service needs a mount, refuse to start without it. Better
    to fail loudly than serve garbage.

14. **`nofail` in fstab** for ephemeral mounts prevents boot hangs
    when the disk doesn't come back. Combined with the mountpoint
    guard, you fail soft on the disk and hard on the service.

15. **`--restart unless-stopped` is the default for production
    containers.** "Restart on crash" is a lower-cost reliability
    feature than most monitoring/alerting.

### About system architecture

16. **Inferred behaviour can lie. Read the source if you can.** The
    wrapper "is a thin proxy" until you discover it routes some
    models to Ollama Cloud. The Modelfile is "local" until you
    realise localhost:11434 means localhost:11434 of the *wrapper*
    container.

17. **Stateful and stateless services should not share fate.**
    Qdrant on the same host as the GPU box means a CUDA driver
    update reboots Qdrant. Move stateful services to dedicated CPU
    hosts as soon as it's worth the operational cost.

18. **Latency cost of splitting hosts is small inside one AZ.**
    Measured ~2 ms RTT, ~3 ms on a chat call. Negligible compared
    to seconds-long LLM compute. Almost any CPU-bound service can
    move off the GPU box.

19. **Anti-rule: don't use latency as a justification for keeping
    stateful services on the expensive GPU box.** The savings are
    milliseconds; the cost is shared reboot cadence with a much
    more frequently-rebooted machine.

### About the work itself

20. **Step-by-step pacing with a human-in-the-loop is the right
    default for risky changes.** Recreating containers, formatting
    disks, modifying boot scripts — every one of these should be
    one step at a time with verification, never a "run this whole
    block."

21. **Write the doc as you go, not after.** The user asked for doc
    updates after each major phase. By the end of the day there were
    three living documents. Trying to recall "what did we do?" the
    next morning would have been much harder.

22. **Honest about wrong turns, including AI wrong turns.** This doc
    catalogs six explicit mistakes I made. The reader needs to know
    the signal-to-noise ratio of my conclusions — if I never admit
    error, they can't calibrate.

23. **Quantify the win.** "62× faster" is concrete and motivating
    in a way that "faster" never is. Always measure before-and-after
    for any optimization you're going to ship.

---

## 10. Reusable patterns and checklists

### Checklist: "Is my Ollama deployment optimal?"

Run these and check the answers:

```bash
docker exec ollama env | grep -iE "ollama|cache|flash"
# Expect: OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE=q8_0 (or fp16 if you have VRAM to spare)
# Expect: OLLAMA_CONTEXT_LENGTH set to your actual max needed context
# Expect: OLLAMA_LOAD_TIMEOUT > expected cold load of your biggest model

docker logs ollama 2>&1 | grep -iE "kv cache|kv_cache|flash" | tail -10
# Expect: K (q8_0) and V (q8_0) — not f16
# Expect: NO "CPU KV buffer" lines (means no CPU spill)

docker exec ollama ollama list
# Expect: only models you actually use; delete the rest

docker inspect ollama --format '{{.HostConfig.RestartPolicy.Name}}'
# Expect: unless-stopped (not "no")

docker inspect ollama --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}'
# Expect: the model store path is on your fastest disk
df -hT $(docker inspect ollama --format '{{range .Mounts}}{{.Source}}{{end}}' | head -1)
# Expect: ext4 or xfs on NVMe/local SSD, not on a slow managed disk
```

### Pattern: per-container recreate script

```bash
#!/bin/bash
# Recreate the <NAME> container with --restart unless-stopped.
# <One sentence about state and constraints>

set -e

LOGFILE="/home/<USER>/<NAME>/docker-run.log"
echo "=== Recreating <NAME> at $(date) ===" | tee -a "$LOGFILE"

# Refuse to start if a critical dependency is missing
if [ ! -d /path/to/required/data ]; then
    echo "ERROR: <dependency> missing. Refusing." | tee -a "$LOGFILE"
    exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q '^<NAME>$'; then
    docker rm -f <NAME> >> "$LOGFILE" 2>&1
fi

docker run -d --name <NAME> \
    --restart unless-stopped \
    -e VAR1=value1 \
    -v /host/path:/container/path \
    -p HOST:CONTAINER \
    <image:tag> | tee -a "$LOGFILE"

docker ps --filter name=^/<NAME>$
```

This pattern, applied to all four containers in this stack, replaced an
opaque "tribal knowledge inside docker inspect" system with explicit,
version-controllable, idempotent scripts.

### Pattern: latency-driven placement decision

```
user_latency ≈ Σ(service_compute) + N_roundtrips × RTT + queueing

Decide where a new service belongs:
- (N × RTT) / total_request_time < 5 %  → split is free; do it
- (N × RTT) / total_request_time 5-20 % → split if you get isolation, scaling, or cost wins
- (N × RTT) / total_request_time > 20 % → keep co-located unless there's a strong other reason

Measure all three with real probes from both candidate hosts before deciding.
```

### Pattern: diagnostic capture for an inherited inference stack

The 15-section script we used (in `outputs.txt`) is a good template.
The sections, in order:

1. Host & OS (`hostnamectl`, kernel, distro, uptime)
2. CPU, memory, disk (`lscpu`, `free`, `lsblk`, `df`)
3. GPU hardware/driver/CUDA (`nvidia-smi`, `nvidia-smi -q`, `nvcc`, `dpkg -l | grep nvidia`)
4. Cloud metadata (IMDS for Azure, instance-identity for AWS)
5. Python/ML runtime (`pip list`, HF cache contents, `torch.cuda` checks)
6. Inference engine specifics (`ollama list`, `ollama ps`, `vllm --version`, etc.)
7. Docker/containers (`docker ps`, `docker inspect`, daemon.json)
8. Other related processes (`ps | grep python`, listening ports)
9. Web/reverse proxy (nginx, caddy, traefik configs)
10. Databases / vector stores
11. Networking (`ip -br a`, `ss -tulpn`, firewall)
12. Storage footprint (`du` of model paths)
13. Systemd services enabled at boot
14. Cron jobs
15. Users & SSH access

Adapt as needed. The principle is: gather **everything that might
matter** in one round-trip.

---

## Closing note

The biggest single observation from this work: **the system was already
doing well on most axes** (the H100 was healthy, the models worked, the
boot script worked, the Django apps responded). The wins came from
*identifying which axes weren't well-tuned* — disk speed, KV cache
config, restart policies, model inventory. None of these required new
hardware. All of them were sitting in plain sight, waiting to be
checked against best practice.

The pattern: **measure first, optimize second, document third.**

When the next engineer (human or AI) inherits this box six months from
now, the three docs (`azure-llm-setup.md`, `architecture-and-placement.md`,
and this learnings doc) should let them get to "I understand this
system" in 30 minutes rather than the full day we spent. That's the
real deliverable.

---

## Appendix A: the daily-deallocate disaster (afternoon of 2026-05-18)

A few hours after writing the original version of this document above,
we discovered that the user's `GPU-1` is **deallocated every night at 8 PM
and manually restarted in the morning** — a cost-control routine that's
totally reasonable for an expensive H100 SKU, but that nobody mentioned
at the start of the session.

The morning's first user-side restart wiped the local NVMe, which is
where we'd just moved 112 GB of Ollama models. The system came back up
with empty model storage. The optimization done in §5 (the NVMe
migration) was about to look like a regression.

The recovery was instructive enough to warrant its own case study.

### What happened in order

1. **Symptom**: user wanted to run the local-vs-cloud speed test for
   `gpt-oss:120b`. Test script reported impossibly fast local responses
   (sub-millisecond) with `eval_count: NA` and `total_duration: null`.
2. **First hypothesis** (wrong, brief): "the test script has a bug."
   The response JSON had no `eval_count` field, which the script
   couldn't handle. But that wasn't the underlying cause.
3. **Right diagnostic**: just hit the endpoint with a raw curl and look
   at the response.
   ```
   {"error":"model 'gpt-oss:120b' not found"}
   http=404
   ```
   No model. Ollama itself was fine, but its model store was empty.
4. **Mystery**: how did the model store empty? We had verified
   `ollama list` showed all 8 models hours earlier. Something happened
   between then and now.
5. **The reveal**: user mentioned "I restarted the GPU machine."
   Suddenly the local NVMe being ephemeral (which we'd documented
   honestly, but treated as a "you'd notice if it happened" issue) was
   the entire story.
6. **Second reveal**: user mentioned the daily deallocate pattern.
   This wasn't a one-time disaster — it was about to happen every
   morning forever.
7. **Recovery plan**: re-format NVMe, mount it, rsync the 112 GB from
   the still-intact Premium SSD rollback volume, restart ollama.
8. **The rsync went to the wrong disk.** This is where the day got
   really interesting.

### The fstab race condition

When we mounted `/mnt/nvme` in the recovery and ran the rsync, the
verify step looked correct: 112 GB at destination, 45 files matching.

But after another mid-day restart, the diagnostics showed:
- NVMe filesystem present but `/mnt/nvme` was not a mountpoint.
- `/dev/sdb1` (the resource disk, 128 GB) was at `/mnt` and **94% full**.
- The 112 GB of model data was on the resource disk, not on NVMe.

Reconstructing the timeline:
- `mount -a` mounted NVMe at `/mnt/nvme` successfully (we saw the `df`
  output confirming `/dev/nvme0n1` at `/mnt/nvme`).
- Some time later, Azure's `waagent` agent decided to remount `/mnt`
  (the resource disk). That **hid the NVMe mount** that was nested
  underneath. Anything written under `/mnt/nvme` from that point on
  landed on the resource disk.
- The rsync was running during this transition. Part of it (or
  possibly all of it, after a remount) went to the resource disk
  instead of NVMe.

The fix: **add `x-systemd.requires-mounts-for=/mnt` to the fstab line
for `/mnt/nvme`.** This tells systemd to wait for `/mnt` to be mounted
before processing the NVMe mount. After this option is in place,
`waagent`'s remount of `/mnt` (if it happens) won't hide our mount,
because we always mount second.

### What we ultimately built

The recovery wasn't just "rsync again." It was:

1. **fstab fix** (one line edit): `x-systemd.requires-mounts-for=/mnt`.
2. **Rewrite `start_containers.sh` into an auto-restore script** with
   four phases:
   - Phase 1: NVMe health + restore. Detect wiped NVMe, reformat, mount,
     and rsync from the Premium SSD rollback if the model store has
     fewer than 5 blobs.
   - Phase 2: Container safety net (existing).
   - Phase 3: Ensure ollama is running (Phase 1 may have stopped it).
   - Phase 4: Warm mistral (existing).
3. **Dry-run validation** on the currently-healthy state — script
   correctly detected "34 blobs found, no restore needed" and did
   nothing destructive.

After the dry-run passed, we asked the user to **actually perform a
deallocate-start cycle** to confirm the end-to-end recovery works. (As
of the time this section was written, that test is in progress.)

### Lessons specific to this incident

1. **The cost-control workflow matters.** "We deallocate nightly" is the
   kind of operational detail that completely changes the design.
   Discover it before deploying ephemeral-disk optimizations, not
   after. Ask "how is this box managed day-to-day?" in the first
   diagnostic capture, not as an afterthought.

2. **`waagent` is not just a one-time-at-boot agent.** It can remount
   disks at runtime for housekeeping. Anything that nests inside a
   waagent-managed mount point needs explicit ordering.

3. **`x-systemd.requires-mounts-for=<path>`** is the standard way to
   make a systemd mount unit depend on another mount being available.
   Use it whenever your mount is nested inside a path managed by
   another service.

4. **Verify a destination is on the right filesystem after a mount-heavy
   operation.** `df -hT <path>` after every mount-relevant step would
   have caught the race immediately. We were checking byte counts ("112
   GB present"), not which disk the bytes were on.

5. **The rollback volume saved us.** We'd kept the original Docker
   volume on Premium SSD as a "rollback for one week" precaution. That
   precaution turned into the persistent source of truth for the entire
   architecture. **Don't delete the safety net just because the migration
   worked.** Make the safety net architectural.

6. **A working script is not a tested script.** The morning of 2026-05-18
   we wrote `start_containers.sh` as a "safety net + warmup" and it
   passed all the dry-runs. But it had no answer for "what if the data
   itself is missing." The afternoon incident exposed that gap. **For
   every operational script, ask: "what if X is wrong at boot?" for
   every X the script depends on.** Disk mounted, data present, network
   up, dependencies installed, etc.

7. **The AI made the same class of error twice in one day.**
   - Morning: claimed gpt-oss was cloud-served before checking the local Modelfile (§7.1 in the original mistakes catalog).
   - Afternoon: claimed "the rsync went to NVMe" when the actual destination was the resource disk.
   - Both times, the symptom + a partial signal looked like a confident
     conclusion. Both times, **verifying the actual filesystem state**
     would have prevented the wrong answer.
   - The transferable rule: **after any mount-related operation, run
     `mountpoint -q && df -hT` on the target before declaring victory.**

### The pattern, generalised

If you're running an LLM stack on ephemeral storage (local NVMe on a
cloud VM, container-local storage in Kubernetes, etc.), your design
needs **three layers**:

1. **Source of truth** — durable storage that survives the operational
   pattern (reboots, deallocates, pod restarts, etc.). For Gravity that
   is the Premium SSD rollback volume.
2. **Working copy** — fast ephemeral storage. For Gravity that is the
   NVMe-backed `/mnt/nvme/ollama`.
3. **Auto-restore mechanism** — a script or operator that detects the
   working copy is missing or empty and restores from the source of
   truth. For Gravity that is the rewritten `start_containers.sh`.

A two-layer design (just "fast ephemeral storage" with no source of
truth) is what we had in the morning. It works perfectly until the
first time the ephemeral layer is wiped, which in this case was eight
hours later. A three-layer design is robust against that and doesn't
require operator intervention.

### Numbers from the recovery

| Step | Time |
|---|---|
| First discovery (local gpt-oss returns 404) | seconds |
| Diagnosis (what state are we in?) | ~10 min, with several wrong hypotheses |
| First recovery rsync (which went to wrong disk) | 18 min |
| Second deallocate cycle | ~10 min (Azure stop/start) |
| Second recovery rsync (correct destination this time) | 65 s (Premium SSD reads were now in page cache) |
| Total elapsed from discovery to fully-recovered state | ~60 min |
| Permanent fix deploy (fstab + script) | ~5 min |
| Dry-run validation | ~1 min |
| **Total cost of building the auto-restore architecture** | **~75 min** |

The morning's NVMe migration took ~30 min and gave us a 62× cold-load
speedup. The afternoon's recovery took ~75 min and made that speedup
**survivable** across the daily deallocate cycle. Both were necessary;
neither would have been sufficient on its own.

### Follow-up: the second failure mode (next-morning test)

The first real deallocate-start cycle after the auto-restore script
went in *also* failed — but in a new place. The boot log showed:

```
NVMe has no ext4 filesystem — formatting with LABEL=nvme-ephemeral
Mounting NVMe at /mnt/nvme
mount: /mnt/nvme: mount point does not exist.
ERROR: NVMe failed to mount. Skipping restore.
```

The fstab race was fixed. The format step worked. But the **mount point
directory itself didn't exist** at the moment we tried to mount.

The reason: `waagent` doesn't just remount `/mnt`, it recreates it
fresh on every Start. Any sub-directories we'd created earlier (such
as `/mnt/nvme`) are gone with it. Our script was assuming the
mountpoint would persist; it doesn't.

When the mount silently failed, Phase 1 skipped the restore, and the
containers came up via `--restart unless-stopped` against a non-existent
bind-mount source. Docker quietly created `/mnt/nvme/ollama` as a plain
directory on the underlying `/dev/sdc1` (the resource disk), and ollama
started writing into the wrong filesystem with zero models.

The fix: **one line** added to `start_containers.sh`:

```bash
if ! mountpoint -q "$NVMe_MOUNT"; then
    mkdir -p "$NVMe_MOUNT"          # <-- the fix
    log "Mounting NVMe at $NVMe_MOUNT"
    mount -a >> "$LOGFILE" 2>&1
fi
```

Recovery was straightforward (stop ollama, clean the bogus path,
`mkdir -p`, `mount -a`, rsync the 112 GB from the rollback in 11 min,
restart ollama, all 8 models back) but the lesson is the more important
artifact:

**Lesson 8 (new): never assume a directory under a managed mount
point persists.** The `/mnt` hierarchy is owned by `waagent` on Azure
VMs (and analogous agents on other clouds). Any sub-directory we
create there is ephemeral with the parent mount. If a script
needs the sub-directory to exist, the script must create it. Don't
rely on "it was there yesterday."

This is the AI making the *same class of error a third time in one
arc* — assuming something about disk state that wasn't true. The first
two were in Appendix A's lesson 7 (cloud routing, rsync destination).
The third is this one (mountpoint directory persisting). Pattern:
**filesystem state is exactly the kind of thing that's invisible to
"obviously this works" reasoning and exactly the kind of thing that
needs explicit verification.**

The full deallocate-start cycle is now validated end-to-end: format →
`mkdir -p` → mount → rsync → start ollama → warm mistral, all
unattended.

### Closing thought

The hardest part of operating a stateful system on cloud infrastructure
isn't the steady-state performance — it's the lifecycle events. Boots,
shutdowns, deallocates, migrations, restores. Each one is a chance for
state to drift between where it should be and where it actually is.

If you're building production infrastructure with an AI assistant, ask
explicitly: **"what happens at each lifecycle event?"** Not just "does
this work right now." That single question would have caught the fstab
race, the ephemeral-disk problem, *and* the missing-mountpoint issue
before any of them bit us.

---

## Appendix B — The local-vs-cloud benchmark (after the recovery)

After the auto-restore was validated on a real deallocate-start cycle,
we ran the speed comparison that had triggered the entire NVMe disaster
in the first place: `gpt-oss:120b` on local H100 NVL vs Ollama Cloud
via the wrapper. Full numbers in `architecture-and-placement.md`
Appendix B. The findings worth remembering as lessons:

1. **The interesting result wasn't "local is faster" or "cloud is
   faster" — it was the crossover.** Local wins decisively at N=1
   (3.5× faster TTFB, 2× warm tok/sec). Cloud wins decisively at
   N=4 (2.7 s faster wall-clock). They meet at N≈2. If you'd asked
   me before the benchmark "which is faster," I'd have said "local"
   and been wrong half the time.

2. **One-axis benchmarks are misleading.** The first comparison
   (single warm request, fixed length) said "local is 2× faster"
   and stopped there. The thorough comparison surfaced the
   concurrency crossover, the cloud-truncation bug, and the
   determinism difference — each of which changes the *routing
   policy*, not just the perceived performance.

3. **`null` fields are a signal, not a bug.** Cloud returns `null`
   for `load_duration`, `prompt_eval_duration`, `eval_duration`. The
   first benchmark script crashed on those nulls; the response was
   to "fix the script." But the right interpretation is: **cloud
   reports less observability than local, and that observability
   gap is itself a placement consideration.** If you need to measure
   tokens/sec on the cloud path, you have to derive it from
   end-to-end wall time, which conflates network and queueing with
   inference.

4. **Cloud has hidden behaviour differences.** `num_predict` is a
   contract locally and a hint on cloud. `seed` is honored locally
   and ignored on cloud. These break silently — the response is
   shorter or differently-worded but nothing throws. Any pipeline
   that depends on these guarantees will fail under cloud routing in
   ways that pass code review.

5. **Re-run quarterly.** Cloud capacity changes. Today's crossover
   at N=2 may be at N=3 next quarter (or N=1 if their fleet gets
   busier). The benchmark itself is the artifact, not the numbers.

The benchmark script is preserved at `/tmp/benchmark.py` on `GPU-1`
(wiped on each deallocate; regenerate from the heredoc that produced
it). Output snapshot at `/tmp/benchmark_output.txt`.
