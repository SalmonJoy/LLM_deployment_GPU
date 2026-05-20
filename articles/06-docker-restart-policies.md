# Docker Restart Policies and Recreate Scripts for Self‑Hosted Inference  

You’ve probably spun up a container that runs a language model, hit *Ctrl‑C* to stop it, and then wondered why it didn’t automatically come back up after a crash. Or you might have seen a production node go “boom” and the container silently disappear, leaving your inference endpoint dead.  

Docker gives you a tiny switch—`--restart`—that decides exactly how a container behaves when the host process dies, when Docker itself restarts, or when you issue a manual stop. The choice of policy can be the difference between a resilient inference service and a silent failure that takes hours to notice.

In this article we will:

1. **Demystify the four Docker restart policies** (`no`, `on-failure`, `always`, `unless-stopped`) using concrete analogies.  
2. **Explain why `unless-stopped` is the production‑grade default** for stateless inference containers.  
3. **Show why `restart: always` is a foot‑gun for stateful services** (think persistent model caches, GPU‑bound processes).  
4. **Introduce the “recreate script” pattern**—a single, version‑controlled `docker‑run.sh` that captures every flag you need.  
5. **Contrast scripts with ad‑hoc `docker inspect`** and illustrate why the latter is fragile.  
6. **Add a safety‑check guard** that refuses to start a container if a critical dependency (e.g., an NVMe mount for model files) is missing.  
7. **Discuss when a boot‑time cron job still makes sense** versus relying purely on restart policies.  

By the end you’ll have a ready‑to‑copy script that logs, removes stale containers, validates the environment, and launches a reproducible inference service. You’ll also understand the trade‑offs that keep your self‑hosted stack both reliable and debuggable.

---  

## 1. The Four Docker Restart Policies, One by One  

Docker’s `--restart` flag lives on the same command line as `docker run`. The syntax is:

```bash
docker run --restart=<policy> …
```

| Policy | When does Docker restart the container? | Typical use‑case |
|--------|------------------------------------------|------------------|
| `no` | Never restart automatically. | One‑off jobs, CI pipelines. |
| `on-failure[:max-retries]` | Restart only if the container exits with a non‑zero status code. Optional `max-retries` caps attempts. | Batch workers that should retry on transient errors. |
| `always` | Restart on any exit (including `0`), and also when Docker daemon starts. | “Fire‑and‑forget” services that must always be up. |
| `unless-stopped` | Same as `always`, **except** Docker will *not* restart a container that you explicitly stopped with `docker stop`. | Production services that you may need to pause for maintenance. |

### 1.1 Analogy #1 – The Thermostat vs. Manual Fan  

Think of a container as a **room heater**.  

| Policy | Heater behavior | Analogy |
|--------|----------------|---------|
| `no` | The heater turns off and stays off, even if the room gets cold. | A **manual fan** you turn on once and then forget about. |
| `on-failure` | The heater restarts only if the temperature drops below a threshold (i.e., something went wrong). | A **thermostat** that only fires when the room is too cold. |
| `always` | The heater turns back on *every* time the power goes out, even if you deliberately unplugged it. | An **emergency generator** that never asks “Did you mean to shut me down?” |
| `unless-stopped` | The heater restarts after power loss **unless** you pressed the “off” switch yourself. | A **smart thermostat** that respects a manual “off” command but otherwise auto‑recovers. |

### 1.2 Analogy #2 – The Smoke Alarm vs. Fire Drill  

| Policy | Smoke alarm behavior | Analogy |
|--------|----------------------|---------|
| `no` | No alarm at all. | A **broken smoke detector** that never beeps. |
| `on-failure` | Beeps only when it detects smoke (non‑zero exit). | A **real smoke alarm** that only sounds on fire. |
| `always` | Beeps every time the power cycles, even if there’s no smoke. | A **fire drill alarm** that rings on any power reset, regardless of cause. |
| `unless‑stopped` | Beeps on power loss **unless** you disabled it. | A **smart alarm** that you can silence for maintenance, but otherwise stays vigilant. |

### 1.3 Analogy #3 – The Kitchen Oven vs. a Stovetop Burner  

| Policy | Kitchen appliance behavior | Analogy |
|--------|----------------------------|---------|
| `no` | You turn the oven on, it bakes, then shuts off forever. | A **single‑use oven**—good for a one‑off roast. |
| `on-failure` | Oven restarts only if the temperature sensor fails (non‑zero exit). | A **self‑cleaning oven** that retries if the cleaning cycle aborts. |
| `always` | Oven powers back on every time you flip the breaker, even if you left it empty. | A **burner left on “high”**—dangerous if you forget to turn it off. |
| `unless‑stopped` | Oven powers back on after a breaker trip **unless** you pressed the “off” knob. | A **smart oven** that knows you intentionally turned it off for cleaning. |

These analogies help you internalize *when* Docker will intervene. The next sections explain why one policy is usually the sweet spot for inference services.

---  

## 2. Why `unless-stopped` Is the Right Default for Production  

Self‑hosted inference containers are typically **stateless** from the host’s perspective: they read model files from a mounted volume, serve HTTP/gRPC requests, and keep no durable data inside the container’s writable layer. Yet they are **stateful** in the sense that a running process holds GPU memory, open sockets, and a warm cache that dramatically reduces latency.

### 2.1 Crash Recovery Without Interfering With Maintenance  

| Scenario | Desired outcome with `unless-stopped` |
|----------|----------------------------------------|
| Container crashes due to an OOM (out‑of‑memory) event | Docker restarts it automatically; inference resumes within seconds. |
| Operator runs `docker stop inference` to upgrade the model | Docker respects the stop; the container stays stopped until you run `docker start`. |
| Host reboots (e.g., kernel panic, power loss) | Docker daemon starts, sees the container marked “running”, and launches it again. |
| System admin disables the service for a scheduled maintenance window | The container remains stopped; you don’t have to remember to “un‑stop” it later. |

If you used `always`, the container would **ignore** the explicit `docker stop` you performed for a rolling upgrade. You’d have to manually intervene with `docker rm -f` or edit the policy—an extra step that can be missed in a hurry, leading to a “ghost” container that keeps restarting and blocks the port you need for the new version.

### 2.2 Quantitative Example  

Consider a GPU‑backed inference service that serves 200 requests per second (RPS) with an average latency of 45 ms. A crash occurs at 02:13 AM due to a transient driver bug.  

| Policy | Time to recovery | Additional downtime |
|--------|------------------|----------------------|
| `no` | Never restarts → manual `docker run` required (≈ 2 min). | 2 min + operator latency. |
| `on-failure` | Restarts automatically (Docker default back‑off: 10 s, 20 s, 40 s). | Up to 40 s. |
| `always` | Restarts instantly (Docker daemon sees container as “running”). | < 5 s. |
| `unless‑stopped` | Same as `always` because the crash is not a manual stop. | < 5 s. |

Both `always` and `unless‑stopped` give sub‑5‑second recovery, but `unless‑stopped` adds the safety net of honoring intentional stops. That tiny nuance translates into **zero accidental rollbacks** during a rolling deployment.

---  

## 3. The Foot‑Gun: `restart: always` With Stateful Services  

Stateful services keep data *inside* the container’s writable layer or rely on external resources that must be in a known state before the process starts. Common examples in the inference world:

| Service | Stateful artifact | Typical failure mode |
|---------|-------------------|----------------------|
| Ollama (model cache) | `/root/.ollama/models` (GPU‑resident cache) | Corrupted cache → container exits with code 0 after cleaning up. |
| Custom model server | SQLite DB in `/data/db.sqlite` | DB lock → process exits with code 0 after releasing lock. |
| GPU driver wrapper | `/dev/nvidia*` device nodes | Driver reload → container exits with code 0 because device disappears. |

When you set `restart: always`, Docker **doesn’t inspect the exit code**. It treats a clean exit (`0`) the same as a crash (`1`). This leads to **boot loops** that hide the underlying problem.

### 3.1 Real‑World Boot Loop  

```bash
docker run --restart=always \
    -v /mnt/models:/models \
    -e OLLAMA_MODELS=/models \
    myorg/ollama:latest
```

Suppose the NVMe mount `/mnt/models` becomes read‑only after a power surge. Ollama detects the error, logs “cannot write to model cache”, exits with code 0 (it thinks it finished cleanly). Docker sees a stopped container, immediately restarts it, and the same error repeats every 2 seconds.  

You end up with a **log flood**:

```
2026-05-20T03:12:01.123Z INFO Starting Ollama …
2026-05-20T03:12:01.456Z ERROR cannot write to model cache /mnt/models
2026-05-20T03:12:01.456Z INFO Exiting with code 0
2026-05-20T03:12:01.459Z WARN Restarting container (attempt 1)
```

Because the container never crashes, you might miss the fact that the underlying NVMe is read‑only. You’ll waste hours chasing a “random restart” that never shows a non‑zero exit.

### 3.2 How `unless‑stopped` Saves You  

If you switch to `unless‑stopped`, Docker will still restart on a non‑zero exit, but **won’t restart on a clean exit**. In the above scenario, Ollama exits with `0`, Docker stops restarting, and you see the error once in the container logs. The failure becomes visible, and you can act (remount the NVMe, fix permissions) instead of chasing a phantom loop.

---  

## 4. The Canonical “Recreate Script” Pattern  

### 4.1 Why a Script Beats `docker inspect`  

Many teams rely on `docker inspect` to “reverse‑engineer” the flags they used to launch a container:

```bash
docker inspect --format='{{json .HostConfig}}' inference
```

The output is a massive JSON blob that changes whenever you add a volume, a label, or a network. Storing that blob in a wiki or a ticket is fragile:

| Issue | Why it matters |
|-------|-----------------|
| **State drift** | The live container may have been edited manually (`docker exec` → `apt install`), diverging from the original launch command. |
| **No version control** | JSON snippets are rarely checked into Git, so you lose the audit trail of *who* added which flag. |
| **No review** | Changes happen on a running host; there’s no PR process to catch a missing `--restart` or a wrong port mapping. |
| **Hard to reproduce** | Re‑creating the container on a new node requires copy‑pasting a massive JSON block, then manually editing it. |

A **single, version‑controlled script** eliminates all of these problems. The script becomes the *source of truth* for:

* Image name and tag
* Environment variables
* Volume mounts (including read‑only flags)
* Port mappings
* Restart policy
* Logging options

Because the script lives in Git, any change is reviewed, tested, and can be rolled back.

### 4.2 Anatomy of a Recreate Script  

Below is a fully‑featured `docker‑run.sh` for an Ollama inference container that:

1. **Logs to a file** (`/var/log/ollama.log`) with timestamps.  
2. **Idempotently removes** any stale container named `ollama`.  
3. **Validates** that the NVMe mount point exists before launching.  
4. **Uses `unless-stopped`** as the restart policy.  
5. **Captures the exact `docker run` command** for reproducibility.  

```bash
#!/usr/bin/env bash
# ------------------------------------------------------------
# docker-run.sh – Recreate script for self‑hosted Ollama
# ------------------------------------------------------------
set -euo pipefail

# ------------------------------------------------------------------
# Configuration (edit these values, then commit the file)
# ------------------------------------------------------------------
CONTAINER_NAME="ollama"
IMAGE="ollama/ollama:0.3.5"
RESTART_POLICY="unless-stopped"
HOST_PORT=11434
MODEL_VOLUME="/mnt/models"
MODEL_MOUNT="/models"
LOG_FILE="/var/log/ollama.log"
ENV_VARS=(
    "OLLAMA_MODELS=${MODEL_MOUNT}"
    "OLLAMA_HOST=0.0.0.0"
    "OLLAMA_PORT=${HOST_PORT}"
)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Helper: write timestamped log line
# ------------------------------------------------------------------
log() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] $*" | tee -a "${LOG_FILE}"
}

# ------------------------------------------------------------------
# Safety guard: ensure the model volume is mounted and writable
# ------------------------------------------------------------------
if ! mountpoint -q "${MODEL_VOLUME}"; then
    log "❌ Critical: ${MODEL_VOLUME} is not a mountpoint. Aborting."
    exit 1
fi

if [[ ! -w "${MODEL_VOLUME}" ]]; then
    log "❌ Critical: ${MODEL_VOLUME} is not writable. Aborting."
    exit 1
fi

log "✅ Mountpoint ${MODEL_VOLUME} verified."

# ------------------------------------------------------------------
# Idempotent removal of any stale container
# ------------------------------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
    log "🔄 Removing existing container ${CONTAINER_NAME}..."
    docker rm -f "${CONTAINER_NAME}" >> "${LOG_FILE}" 2>&1
    log "✅ Removed."
else
    log "ℹ️ No existing container named ${CONTAINER_NAME}."
fi

# ------------------------------------------------------------------
# Build the docker run command (for logging / audit)
# ------------------------------------------------------------------
RUN_CMD=(
    docker run -d
    --name "${CONTAINER_NAME}"
    --restart "${RESTART_POLICY}"
    -p "${HOST_PORT}:11434"
    -v "${MODEL_VOLUME}:${MODEL_MOUNT}"
)

# Append environment variables
for env in "${ENV_VARS[@]}"; do
    RUN_CMD+=("-e" "${env}")
done

# Append logging driver (json-file with max‑size)
RUN_CMD+=(
    --log-driver json-file
    --log-opt max-size=10m
    --log-opt max-file=3
)

RUN_CMD+=("${IMAGE}")

# ------------------------------------------------------------------
# Execute the run command
# ------------------------------------------------------------------
log "🚀 Starting container with command:"
log "${RUN_CMD[*]}"
"${RUN_CMD[@]}" >> "${LOG_FILE}" 2>&1

log "✅ Container ${CONTAINER_NAME} started. PID=$(docker inspect -f '{{.State.Pid}}' ${CONTAINER_NAME})"
log "📦 Logs tailing now (press Ctrl‑C to exit):"
exec tail -F "${LOG_FILE}"
```

#### What the script does, step by step

| Step | Command | Reason |
|------|---------|--------|
| **Safety guard** | `mountpoint -q /mnt/models` | Guarantees the NVMe is present; prevents silent boot loops. |
| **Idempotent rm** | `docker rm -f ollama` (only if exists) | Guarantees a clean slate; avoids “container already exists” errors on redeploy. |
| **Restart policy** | `--restart unless-stopped` | Recovers from crashes, respects manual stops. |
| **Logging** | `--log-driver json-file --log-opt max-size=10m` | Rotates logs automatically; prevents disk exhaustion. |
| **Env vars** | `-e OLLAMA_MODELS=/models` | Explicitly injects configuration; easier to audit. |
| **Port mapping** | `-p 11434:11434` | Exposes the inference endpoint. |
| **Volume mount** | `-v /mnt/models:/models` | Binds the model store; read‑write on host side. |
| **Tail** | `tail -F /var/log/ollama.log` | Gives you live feedback after the script exits (useful for systemd services). |

All values are defined at the top of the script, making it trivial to bump the image tag or change the host port without hunting through a long command line.

### 4.3 Version‑Control Integration  

Store the script in a repository alongside your model files:

```
repo/
├─ models/
│  └─ llama-2-7b.gguf
├─ docker-run.sh
└─ README.md
```

Add a **Git hook** that lints the script (e.g., `shellcheck`) and a **CI job** that runs `docker run --rm -d … && docker logs -f` on a test node. This ensures that any change to the launch parameters is automatically validated.

---  

## 5. The Fragility of “Tribal Knowledge” in `docker inspect`  

Even with a solid script, some teams still rely on `docker inspect` to answer questions like “what volume is this container using?” or “which restart policy did we set?”. That approach suffers from three systemic problems.

### 5.1 State Drift  

A container can be **mutated after launch**:

```bash
docker exec ollama apt-get update && apt-get install -y vim
docker exec ollama mkdir /tmp/debug
```

Now `docker inspect` will show a larger `SizeRw` field, but the **launch command never reflected those changes**. If you later redeploy using a script that doesn’t include these manual steps, the container’s behavior diverges from the production baseline.

### 5.2 No Auditable History  

When you copy‑paste a JSON snippet into a ticket, you lose:

| Missing piece | Impact |
|---------------|--------|
| Who added the volume? | Hard to trace permission issues. |
| When was the restart policy changed? | Hard to correlate with a downtime event. |
| What environment variable values were used? | Hard to reproduce a bug in a staging environment. |

Git solves this by providing a **chronological log** (`git log -p docker-run.sh`) that shows exactly when `--restart` switched from `always` to `unless‑stopped`.

### 5.3 No Review Process  

Manual `docker run` commands are often typed directly on a production host. A typo like `-p 11434:11435` (note the swapped port) silently creates a mis‑routed endpoint. With a script, the change would go through a pull request, where a reviewer can spot the mistake before it lands.

---  

## 6. Safety‑Check Pattern: Guarding Critical Dependencies  

The **safety‑check** in the example script (`mountpoint -q`) is a minimal illustration. In practice you may need multiple guards:

| Dependency | Guard command | Failure mode |
|------------|---------------|--------------|
| NVMe model store (`/mnt/models`) | `mountpoint -q /mnt/models && [[ -w /mnt/models ]]` | Container would crash repeatedly on write errors. |
| GPU driver (`/dev/nvidia0`) | `[[ -c /dev/nvidia0 ]]` | Container would exit with “no device found”. |
| Redis cache (`redis:6379`) | `nc -z -w5 redis 6379` | Model server would fallback to disk, causing latency spikes. |
| License file (`/etc/ollama/license.key`) | `[[ -f /etc/ollama/license.key ]]` | Server would start in trial mode, violating compliance. |

You can wrap these checks in a reusable function:

```bash
require_file() {
    local path=$1
    if [[ ! -f "${path}" ]]; then
        log "❌ Required file ${path} missing."
        exit 1
    fi
}
require_device() {
    local dev=$1
    if [[ ! -c "${dev}" ]]; then
        log "❌ Required device ${dev} missing."
        exit 1
    fi
}
```

Then call them early in the script:

```bash
require_device "/dev/nvidia0"
require_file "/etc/ollama/license.key"
```

If any guard fails, the script aborts *before* Docker even touches the host, keeping the system in a known safe state.

---  

## 7. When a Boot‑Time Cron Job Still Makes Sense  

Docker restart policies cover **container‑level failures** (crash, host reboot). However, there are edge cases where a **system‑level watchdog** is valuable.

| Situation | Why a cron (or systemd timer) helps |
|-----------|--------------------------------------|
| **Container never created** (e.g., after a fresh OS install) | Restart policies are irrelevant because there is no container to restart. A boot‑time script that runs `docker ps -q -f name=ollama || ./docker-run.sh` guarantees the service appears after a clean OS install. |
| **Corrupt Docker daemon state** (e.g., after an unclean shutdown) | Docker may lose the container metadata. A watchdog that checks `docker inspect` and re‑creates missing containers can self‑heal. |
| **External dependency becomes unavailable** (e.g., network drive unmounted) | The guard in the script will abort, but a cron can periodically retry the mount and then re‑run the script once the dependency is back. |
| **Scheduled maintenance windows** | You can combine a cron that *stops* the container at 02:00 AM, runs a backup, then *starts* it at 02:30 AM. The `unless‑stopped` policy respects the stop, while the cron re‑starts it. |

### 7.1 Minimal Boot‑Time Wrapper  

Create `/etc/rc.local` (or a systemd unit) that runs once at boot:

```bash
#!/usr/bin/env bash
# /etc/rc.local – Ensure Ollama container exists after boot

set -euo pipefail

CONTAINER="ollama"
SCRIPT="/opt/ollama/docker-run.sh"

if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}\$"; then
    echo "$(date) – Container ${CONTAINER} missing, launching…" >> /var/log/rc-ollama.log
    "${SCRIPT}" >> /var/log/rc-ollama.log 2>&1
else
    echo "$(date) – Container ${CONTAINER} already present." >> /var/log/rc-ollama.log
fi
```

Make it executable and enable it (`chmod +x /etc/rc.local`). This tiny wrapper runs **once** after the kernel and Docker daemon are up, guaranteeing the container exists even if the host was freshly provisioned.

### 7.2 When to Skip the Cron  

If you **always** deploy containers via the recreate script (e.g., as part of a CI/CD pipeline) and you never rely on manual `docker run` commands, you can safely omit the boot‑time cron. The `unless‑stopped` policy already covers:

* Host reboots
* Docker daemon restarts
* Container crashes

The only remaining gap is the *initial* creation, which is handled by your provisioning tool (Terraform, Ansible, cloud‑init). In that case, the cron becomes redundant noise.

---  

## 8. Advanced Tips: Going Beyond a Single Script  

### 8.1 Docker Compose as a Structured Alternative  

If you have **multiple containers** (model server, Redis cache, Prometheus exporter), a `docker-compose.yml` can encode the same information in a declarative format:

```yaml
version: "3.9"
services:
  ollama:
    image: ollama/ollama:0.3.5
    container_name: ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - /mnt/models:/models
    environment:
      OLLAMA_MODELS: /models
      OLLAMA_HOST: 0.0.0.0
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

You can still keep the **safety guard** in a wrapper script that runs `docker compose up -d` *after* verifying the mount. The advantage is that Compose automatically creates a network, resolves service dependencies (`depends_on`), and can be version‑controlled as a single YAML file.

### 8.2 Using `docker run --init` for Signal Forwarding  

Inference containers often need to respond to `SIGTERM` (e.g., to gracefully unload a model from GPU memory). Adding `--init` injects a tiny `tini` init process that forwards signals correctly:

```bash
docker run -d --init --restart unless-stopped …
```

If you forget this flag, `docker stop` sends `SIGTERM` to the **PID 1** process inside the container, which may ignore it (common in Python scripts that don’t install a signal handler). The container then gets a hard `SIGKILL` after the default 10‑second timeout, potentially corrupting in‑memory caches.  

### 8.3 Pinning Image Digests for Immutable Deployments  

Never rely on the `latest` tag in production. Use an immutable digest:

```bash
IMAGE="ollama/ollama@sha256:9c5b2f1e5d8a3f7e4b6c9d2e..."
```

When you need to upgrade, you change the digest in the script, open a PR, and merge. This guarantees that the exact binary you tested is the one that runs on every node.

### 8.4 Healthchecks to Complement Restart Policies  

Docker’s restart policies react to **process exit**, not to a hung service. Add a `HEALTHCHECK` that probes the HTTP endpoint:

```bash
docker run -d \
    --health-cmd="curl -f http://localhost:11434/health || exit 1" \
    --health-interval=30s \
    --health-timeout=5s \
    --health-retries=3 \
    …
```

When the healthcheck fails three times, Docker marks the container `unhealthy`. You can then use an external watchdog (e.g., a systemd service with `Restart=on-failure`) to **restart the container** based on health status, providing a second layer of resilience.

---  

## 9. Putting It All Together: A Full Production Blueprint  

Below is a **complete, production‑ready layout** for a self‑hosted inference node that serves an Ollama model, caches results in Redis, and exports metrics to Prometheus.

```
/opt/inference/
├─ docker-compose.yml
├─ .env                # environment variables, git‑ignored
├─ scripts/
│  ├─ ollama-run.sh    # wrapper with safety checks
│  └─ redis-run.sh
└─ rc.local            # boot‑time guard
```

### 9.1 `.env` (git‑ignored)

```dotenv
# .env – keep secrets out of version control
OLLAMA_LICENSE_KEY=/etc/ollama/license.key
REDIS_PASSWORD=SuperSecret123
```

### 9.2 `docker-compose.yml`

```yaml
version: "3.9"
services:
  ollama:
    image: ollama/ollama@sha256:9c5b2f1e5d8a3f7e4b6c9d2e...
    container_name: ollama
    restart: unless-stopped
    init: true
    ports:
      - "11434:11434"
    volumes:
      - /mnt/models:/models
      - ${OLLAMA_LICENSE_KEY}:/etc/ollama/license.key:ro
    environment:
      OLLAMA_MODELS: /models
      OLLAMA_HOST: 0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    logging:
      driver: json-file
      options:
        max-size: "20m"
        max-file: "5"

  redis:
    image: redis:7-alpine
    container_name: redis
    restart: unless-stopped
    command: ["redis-server", "--requirepass", "${REDIS_PASSWORD}"]
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 15s
      timeout: 3s
      retries: 5
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

### 9.3 `scripts/ollama-run.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
source /opt/inference/.env

# Safety guard – NVMe mount
if ! mountpoint -q /mnt/models; then
    echo "❌ /mnt/models is not mounted. Abort."
    exit 1
fi

# Guard – license file
if [[ ! -f "${OLLAMA_LICENSE_KEY}" ]]; then
    echo "❌ License key missing at ${OLLAMA_LICENSE_KEY}"
    exit 1
fi

# Run compose (idempotent)
docker compose -f /opt/inference/docker-compose.yml up -d ollama
```

### 9.4 `rc.local` (boot‑time guard)

```bash
#!/usr/bin/env bash
# Ensure the inference stack is up after a fresh boot

set -euo pipefail
logfile="/var/log/inference-boot.log"

{
    echo "$(date) – Boot guard start"
    /opt/inference/scripts/ollama-run.sh
    /opt/inference/scripts/redis-run.sh
    echo "$(date) – Boot guard completed"
} >> "${logfile}" 2>&1
```

With this blueprint:

* **All launch flags live in `docker‑compose.yml`** (volumes, ports, restart policy).  
* **Safety checks live in the wrapper scripts**, keeping the compose file clean.  
* **Boot‑time `rc.local` guarantees the stack appears after a fresh OS install**.  
* **Version control** (Git) tracks every change to the compose file, scripts, and environment variables.  

---  

## 10. Takeaways for the Senior Engineer  

1. **Pick `unless‑stopped` as your default**. It gives you crash‑recovery *and* respects intentional pauses, which is exactly what a production inference service needs.  
2. **Avoid `restart: always` for any container that can exit cleanly on error** (most stateful services). The silent restart loop hides the root cause and can fill your logs with noise.  
3. **Encapsulate the full `docker run` command in a version‑controlled script**. This eliminates reliance on `docker inspect`, prevents state drift, and makes peer review trivial.  
4. **Add explicit safety guards** for critical resources (mount points, device nodes, license files). A single `mountpoint -q` can save you from a 30‑minute boot‑loop debugging session.  
5. **Use a boot‑time watchdog only for “container‑does‑not‑exist” scenarios**. Once the container is created, let Docker’s restart policies and healthchecks do the heavy lifting.  
6. **Combine restart policies with healthchecks and `--init`** to handle both crashes and hung processes.  
7. **Keep your images immutable** (digests) and your environment variables out of the repo (`.env` + Git‑ignore).  

You now have a concrete, reproducible pattern that scales from a single‑container Ollama deployment to a multi‑service inference stack. The next time you spin up a new model or upgrade the GPU driver, you’ll edit a line in a script, open a PR, and let Docker do the rest—without worrying that a hidden boot loop will silently eat your compute cycles.  

Happy containerizing!