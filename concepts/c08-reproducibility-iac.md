# Reproducibility for ML Infrastructure: From “It Worked on the Box” to “Anyone Can Rebuild It”

---

## 1. Why “It Worked on My Laptop” Is a Red Flag, Not a Badge

You’ve probably heard the classic line, *“It works on my machine.”* In the world of machine‑learning (ML) serving, that line usually means **the container is running, but nobody knows how it got there**.  

Imagine you inherit a kitchen where the chef left a half‑scribbled recipe on the back of a napkin: “2 cups flour, a pinch of salt, add *something* from the pantry, bake 12 min.” The dish tastes great, but the next cook can’t reproduce it without guessing. The same thing happens with containers: the image is running, but the exact `docker run` command, the environment variables, the volume mounts, the GPU flags—everything that made the dish tasty—is lost in the fog.

In a healthy ML infra, the “recipe” lives in a version‑controlled cookbook, not on a napkin. The cookbook tells you **exactly** how to spin up the same serving container on any host, at any time, with the same performance characteristics.

---

## 2. The Symptom: State Drift and the Illusion of “Running”

When you `docker inspect <container>` you get a snapshot of the *current* state. That snapshot is useful for debugging, but it is **not** a source of truth for recreation. The moment you destroy the container and run `docker run` again—*without* the original command—you will likely see:

| What you lose | Why it matters | Typical cause |
|---------------|----------------|---------------|
| `-e` environment variables (e.g., `MODEL_PATH=/models/v1`) | Model loading fails | Ad‑hoc `docker exec` to set env |
| Volume mounts (`-v /data:/data`) | Data not found, training restarts | Manual `docker cp` instead |
| `--gpus all` flag | No GPU, inference latency spikes | Forgetting to copy the flag |
| `--restart unless-stopped` | Container dies on reboot | Relying on host’s init system |
| Network mode (`--network host`) | Port collisions, firewall issues | Temporary `--net bridge` for debugging |

Every time a teammate runs a quick `docker exec -it <c> bash` and tweaks a config file inside the container, they are **mutating the running state**. Those changes are invisible to `docker ps` and `docker inspect`. Over weeks, the container drifts farther from the original recipe, and the next person who tries to “re‑create it from scratch” ends up with a broken stack.

### Real‑world example

```bash
# Original (lost) command
docker run -d \
  --name sentiment-svc \
  -e MODEL_PATH=/opt/models/sentiment-1.2.0.pt \
  -v /mnt/models:/opt/models \
  --gpus all \
  --restart unless-stopped \
  -p 8080:8080 \
  myorg/sentiment:1.2.0
```

A teammate later does:

```bash
docker exec -it sentiment-svc bash
# Inside container, they edit /app/config.yaml and add a new env var:
export LOG_LEVEL=debug
```

Now the container is **different** from the original, but `docker inspect sentiment-svc` still shows the original command line. If the container crashes and the host restarts, the new `LOG_LEVEL` disappears, and the service behaves differently—exactly the kind of nondeterminism that kills reproducibility.

---

## 3. The Fix: One Versioned `docker‑run.sh` per Service

The simplest, most auditable way to lock down the recipe is to **store a single, version‑controlled shell script** that contains the full `docker run` invocation. Treat that script as the *only* legitimate way to start the container.

### 3.1 Canonical script template

```bash
#!/usr/bin/env bash
# ------------------------------------------------------------
# sentiment-svc – reproducible launch script
# Commit: 9f2c3e1 (2024‑02‑12)
# Purpose: Serve sentiment analysis model v1.2.0
# ------------------------------------------------------------
set -euo pipefail   # fail fast, treat unset vars as errors

# ----------------------------------------------------------------
# 1️⃣  Define immutable configuration
# ----------------------------------------------------------------
IMAGE="myorg/sentiment:1.2.0"          # pinned tag, never :latest
CONTAINER_NAME="sentiment-svc"
HOST_PORT=8080
HOST_MODEL_DIR="/mnt/models"
CONTAINER_MODEL_DIR="/opt/models"
GPU_FLAGS="--gpus all"
RESTART_POLICY="--restart unless-stopped"
LOG_FILE="/var/log/sentiment-svc.log"

# ----------------------------------------------------------------
# 2️⃣  Clean up any stale instance (idempotent)
# ----------------------------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "🧹 Removing stale container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}"
fi

# ----------------------------------------------------------------
# 3️⃣  Run the container with explicit, reproducible flags
# ----------------------------------------------------------------
docker run -d \
  --name "${CONTAINER_NAME}" \
  -e MODEL_PATH="${CONTAINER_MODEL_DIR}/sentiment-1.2.0.pt" \
  -e LOG_LEVEL="info" \
  -v "${HOST_MODEL_DIR}:${CONTAINER_MODEL_DIR}:ro" \
  ${GPU_FLAGS} \
  ${RESTART_POLICY} \
  -p "${HOST_PORT}:8080" \
  "${IMAGE}" \
  >> "${LOG_FILE}" 2>&1 &

echo "🚀 ${CONTAINER_NAME} started, logs → ${LOG_FILE}"
```

#### Why this script solves the pain points

| Problem | Script feature | How it prevents drift |
|---------|----------------|-----------------------|
| Lost env vars | All `-e` declarations are in the script | No hidden state |
| Missing volume mounts | `-v` line is explicit, read‑only (`:ro`) | Guarantees data consistency |
| GPU flag forgotten | `GPU_FLAGS` variable defined once | Central source of truth |
| Restart policy disappears after host reboot | `--restart unless-stopped` baked in | Container survives reboots |
| Unclear logging location | `LOG_FILE` defined, all stdout/stderr redirected | Auditable output |

### 3.2 Version control matters

Commit the script to a Git repository **alongside** the Dockerfile that builds the image. Tag the repository with the same version number you use for the image (`v1.2.0`). When you need to rebuild the image, you check out the tag, run the CI pipeline, and the script will still point to the exact image you just built.

```bash
git checkout tags/v1.2.0
docker build -t myorg/sentiment:1.2.0 .
git push origin v1.2.0
```

Now you have a *closed loop*: the script points to a tag, the tag points to a Dockerfile commit, and the Dockerfile commit produces the same image every time (provided the build context is deterministic).

---

## 4. When Do You Need Ansible, Terraform, or Helm?  

For a **single inference host** that runs three services (e.g., a model server, a monitoring sidecar, and a log forwarder), pulling in a full‑blown configuration‑management stack can be overkill. The shell‑script approach gives you:

| Criterion | Full IaC (Ansible/Terraform/Helm) | Minimal Script‑Only |
|-----------|-----------------------------------|---------------------|
| **Setup time** | 1‑2 days (write playbooks, test) | < 2 hours (write 4 scripts) |
| **Learning curve** | Requires YAML, Jinja, module knowledge | Bash basics + Docker CLI |
| **Audit surface** | Multiple layers, templating variables | One file, line‑by‑line review |
| **Change latency** | Push → run → wait for idempotence | Edit script → run (seconds) |
| **Portability** | Tied to specific orchestrator version | Works on any Linux host with Docker |

That’s not to say Ansible or Terraform are bad—they excel when you need to provision VMs, configure network ACLs, or orchestrate dozens of services across many clusters. But for the **“four‑script” pattern** described below, the trade‑off leans heavily toward simplicity and traceability.

### 4.1 The “Four‑Script” Pattern

| Script | Purpose |
|--------|---------|
| `build-image.sh` | `docker build` with pinned base images |
| `docker-run.sh` | Starts the service (shown above) |
| `update-config.sh` | Generates a config file from a small JSON/YAML template |
| `health-check.sh` | `curl` / `docker exec` sanity check, returns non‑zero on failure |

All four live in the same directory, are versioned together, and can be invoked manually or from a CI pipeline. The entire stack can be rebuilt on a fresh VM with:

```bash
git clone https://github.com/myorg/sentiment-svc.git
cd sentiment-svc
./build-image.sh
./docker-run.sh
./health-check.sh   # exits 0 on success
```

That sequence takes **≈ 8 minutes** on a modest `c5.large` (2 vCPU, 4 GiB) instance.

---

## 5. Auditing Made Easy: One Script, One Story

Imagine it’s 02:13 UTC, your on‑call engineer gets a PagerDuty alert: *“sentiment‑svc latency > 5 s.”* The engineer logs in, runs `docker logs sentiment-svc`, sees a stack trace, and needs to know **whether the container was launched with the correct GPU flag**.

With a multi‑layered stack (Helm chart → values.yaml → Ansible role → Dockerfile), the engineer must:

1. Pull the Helm release version.
2. Locate the corresponding `values.yaml` in Git.
3. Render the chart (Helm templating).
4. Find the generated `docker run` line inside a ConfigMap.
5. Verify the flag.

That is **five separate artefacts** to chase down, each potentially out of sync.

With the script‑only approach, the engineer opens **one file**:

```bash
less docker-run.sh
```

Scrolling to the `GPU_FLAGS` line instantly confirms whether `--gpus all` is present. No templating, no hidden defaults, no guesswork. The audit trail is a single commit diff, which can be reviewed in under a minute.

### Real‑world audit snippet

```diff
diff --git a/docker-run.sh b/docker-run.sh
index 3e7b9c1..a4f2d8e 100755
--- a/docker-run.sh
+++ b/docker-run.sh
@@
- GPU_FLAGS="--gpus all"
+ # Updated to limit GPU memory to 8 GiB after OOMs on 2024‑04‑03
+ GPU_FLAGS="--gpus 'device=0,capabilities=compute,utility,mem=8g'"
```

The diff tells you *exactly* why the service started failing: a recent change reduced the memory allocation, causing the model to be evicted from the GPU. The fix is now a one‑line edit and a redeploy.

---

## 6. Patterns That Keep Drift at Bay

### 6.1 Pin Tags Rigorously

Never use `:latest`. Always reference an immutable tag that includes **major.minor.patch.build** (e.g., `1.2.0.20240401`). This eliminates the “my image changed overnight” surprise.

```bash
# Bad
docker pull myorg/sentiment:latest

# Good
docker pull myorg/sentiment:1.2.0.20240401
```

### 6.2 Commit the Build Artifact Tag

When your CI pipeline builds an image, have it output the full SHA‑256 digest and write that digest into a `VERSION` file committed alongside the launch script.

```bash
# CI step
IMAGE="myorg/sentiment:1.2.0"
docker build -t "${IMAGE}" .
DIGEST=$(docker images --digests "${IMAGE}" | awk 'NR==2 {print $3}')
echo "${DIGEST}" > VERSION
git add VERSION docker-run.sh
git commit -m "Release 1.2.0 – lock to ${DIGEST}"
```

Now the script can read the digest:

```bash
IMAGE="myorg/sentiment@$(cat VERSION)"
```

Even if the tag `1.2.0` is later retagged (a mistake many teams make), the script still points to the exact image you tested.

### 6.3 Never Edit a Running Container

If you need to change an environment variable or mount a new volume, **edit the script** and run it again. Docker will automatically replace the old container because the script removes any stale instance first.

```bash
# BAD: docker exec sentiment-svc export NEW_VAR=foo
# GOOD: edit docker-run.sh, add -e NEW_VAR=foo, then:
./docker-run.sh   # recreates container with NEW_VAR
```

### 6.4 Enforce Idempotence

The `if docker ps -a …` block in the template guarantees that running the script twice does not spawn duplicate containers. This property is essential for automated deployments (e.g., a nightly cron that guarantees the service is up).

---

## 7. Bonus: Rebuilding on a Fresh VM in Ten Minutes

Let’s walk through a realistic migration. You have a `c5.large` inference host that is about to be decommissioned. You spin up a new `c5.large` in a different AZ and need the stack up and running **fast**.

### 7.1 Prerequisites on the new VM

```bash
# Install Docker (official script)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER   # optional, for passwordless docker
```

### 7.2 Clone the repo and run the four scripts

```bash
git clone https://github.com/myorg/sentiment-svc.git
cd sentiment-svc

# 1️⃣ Build the image (uses pinned base layers)
./build-image.sh

# 2️⃣ Start the service
./docker-run.sh

# 3️⃣ Verify health
./health-check.sh
```

**Timing breakdown (average on AWS `c5.large`):**

| Step | Duration |
|------|----------|
| Docker install | 1 min |
| Git clone (≈ 5 MB) | 30 s |
| `build-image.sh` (pull base, compile) | 4 min |
| `docker-run.sh` (pulls image if missing) | 30 s |
| `health-check.sh` | 10 s |
| **Total** | **≈ 6 min** (including manual verification) |

Even if you add a monitoring sidecar, the same pattern applies: a separate `docker-run-sidecar.sh` script, identical idempotent cleanup, and a shared `VERSION` file.

### 7.3 Scaling to Multiple Hosts

When you need the same service on **N** hosts, you can wrap the scripts in a tiny Makefile:

```makefile
HOSTS = host-a host-b host-c

build:
	for h in $(HOSTS); do \
	  ssh $$h 'cd ~/sentiment-svc && ./build-image.sh'; \
	done

run:
	for h in $(HOSTS); do \
	  ssh $$h 'cd ~/sentiment-svc && ./docker-run.sh'; \
	done

check:
	for h in $(HOSTS); do \
	  ssh $$h 'cd ~/sentiment-svc && ./health-check.sh'; \
	done
```

Running `make run` launches the service on all three hosts in parallel (or sequentially, depending on your SSH configuration). No need for a full orchestration engine; the Makefile is versioned alongside the scripts, so the deployment logic stays transparent.

---

## 8. Advanced Tips for Senior Engineers

You’ve now got a rock‑solid baseline. Here are a few refinements you can adopt once the fundamentals are in place.

| Feature | Why it matters | Minimal implementation |
|---------|----------------|------------------------|
| **Immutable config files** | Guarantees that the container sees the same JSON/YAML each run | `update-config.sh` writes to `/etc/svc/config.yaml` on the host, then mounts `-v /etc/svc:/etc/svc:ro` |
| **SHA‑256 digests in `docker‑run.sh`** | Even if a tag is overwritten, the digest never changes | `IMAGE="myorg/sentiment@sha256:$(cat DIGEST)"` |
| **CI gate that diffs `docker‑run.sh` against live container** | Detects drift automatically | `docker inspect` → compare env vars → fail CI if mismatch |
| **Resource‑capped logs** | Prevents log‑disk exhaustion on long‑running services | `--log-opt max-size=10m --log-opt max-file=3` in `docker‑run.sh` |
| **Namespace isolation** | Avoids port collisions when multiple services share a host | Use Docker user‑defined bridge network: `docker network create ml-net` and add `--network ml-net` |

### Example: Adding immutable config generation

```bash
#!/usr/bin/env bash
# update-config.sh – generate config.yaml from a template

set -euo pipefail

TEMPLATE="templates/config.yaml.tmpl"
OUTPUT="/etc/svc/config.yaml"

# Simple envsubst substitution
export MODEL_PATH="/opt/models/sentiment-1.2.0.pt"
export LOG_LEVEL="info"

envsubst < "${TEMPLATE}" > "${OUTPUT}"
chmod 644 "${OUTPUT}"
```

`templates/config.yaml.tmpl`:

```yaml
model_path: "${MODEL_PATH}"
log_level: "${LOG_LEVEL}"
batch_size: 64
```

Now `docker-run.sh` adds the mount:

```bash
-v /etc/svc:/etc/svc:ro \
```

The config is **generated once**, versioned, and never edited inside the container.

---

## 9. Checklist – Make Your ML Service Reproducible Today

| ✅ Item | How to verify |
|--------|---------------|
| **All `docker run` flags are explicit** | Open `docker-run.sh`; no hidden defaults. |
| **Image tag is pinned (no `latest`)** | `grep ':' docker-run.sh` → shows `1.2.0.20240401`. |
| **Digest stored in `VERSION`** | `cat VERSION` prints a SHA‑256 string. |
| **No manual `docker exec` edits** | `git log -p docker-run.sh` shows the only source of changes. |
| **Idempotent cleanup block present** | `if docker ps -a … rm -f` exists. |
| **Health‑check script exits 0 on success** | `./health-check.sh; echo $?` → `0`. |
| **All scripts are executable and versioned** | `git ls-files -m` shows none untracked. |
| **Documentation of each env var** | Header comment block lists all `-e` variables. |
| **Rebuild test on a fresh VM** | Follow the “Bonus” steps on a disposable EC2 instance; total time < 10 min. |

If you can tick every box without digging through multiple repos, you have achieved the “Anyone can rebuild it” state.

---

## 10. From Box‑Bound to Team‑Bound

Reproducibility isn’t a luxury; it’s a safety net that turns a fragile, “it works on my box” setup into a **team‑owned, auditable, and portable** service. By treating the `docker run` command as code—versioned, reviewed, and immutable—you eliminate the most common source of drift: undocumented ad‑hoc tweaks.

You don’t need a heavyweight orchestrator for a handful of inference containers. A handful of well‑named shell scripts, a Git repo, and a disciplined workflow give you the same guarantees that larger IaC tools provide, but with **far less cognitive overhead**. When the next on‑call shift rolls around, the engineer will spend minutes reading a single script instead of hunting through Helm values, Ansible roles, and generated manifests.

The payoff is immediate: faster incident response, smoother migrations, and confidence that the model you trained yesterday will run exactly the same way on a brand‑new VM tomorrow. That is the essence of reproducibility for ML infrastructure—turning a scribbled napkin recipe into a cookbook that anyone on your team can follow, any time, on any machine.