# Building a Diagnostic Capture Script for Inherited LLM Infrastructure

## Why “One‑Question‑At‑A‑Time” Breaks Down in Real‑World Ops  

You’ve probably been in a support call where the remote operator asks you to run a single command, copy‑paste the result, wait for a reply, then move on to the next item. On paper it looks tidy: “run `uname -a`, send back, then `lscpu`, then `docker ps`…”. In practice each round‑trip costs **≈5 minutes** (typing, scrolling, copy‑pasting, network latency, waiting for the other side to read).  

If you need to answer **dozens** of such questions, the time balloons:

| # of questions | Time per question | Cumulative time |
|----------------|-------------------|-----------------|
| 10             | 5 min             | 50 min          |
| 20             | 5 min             | 1 h 40 min      |
| 40             | 5 min             | 3 h 20 min      |
| 80             | 5 min             | 6 h 40 min      |

A half‑day of pure “ping‑pong” is a lost opportunity to actually **debug** the model serving pipeline.  

### Analogy: The Doctor’s One‑Test‑At‑A‑Time Clinic  

Imagine you walk into a clinic where the physician orders **one blood test per visit**. You leave, come back the next day, wait another hour, get the next test, and so on. After a week you finally have a panel of 40 markers, but you’ve already missed the window for early treatment.  

Contrast that with a modern lab that draws a single vial and runs a **40‑marker panel** in one go. The physician gets the whole picture instantly and can act decisively.  

Your diagnostic capture script is the “single‑vial panel” for LLM infrastructure. It gathers everything you might need—*even the things you don’t suspect*—in a single, reproducible run, and writes a **structured, line‑by‑line report** that you can hand off to anyone (on‑call, security, compliance) without a back‑and‑forth.

### Analogy: Plumbing a Multi‑Story Building in One Sweep  

Think of a 10‑story building with separate shut‑off valves for each floor. If you need to check water pressure, you climb to floor 1, test, go back down, then repeat for floor 2, etc. It’s inefficient and you might miss a leak on floor 7 because you’re exhausted.  

Instead, you install a **central manifold** that lets you read pressure, flow, and temperature for every floor from a single control room. The manifold is analogous to a script that queries the OS, container runtime, GPU driver, cloud metadata, and all ancillary services in one pass, presenting a unified view.

## The 15‑Section Diagnostic Checklist  

Below is the **canonical checklist** that has proven reliable across on‑prem, cloud, and hybrid LLM deployments. Treat each bullet as a *section* in your script; the script will emit a header, run the commands, and dump the raw output. The order is intentional: start with the host, then move outward to the services that sit on top of it.

| # | Section | Why it matters | Typical command(s) |
|---|---------|----------------|--------------------|
| 1 | **Host & OS** | Baseline for kernel, distro, uptime. | `cat /etc/os-release`, `uname -r`, `uptime -p` |
| 2 | **CPU / RAM / Disks** | Resource headroom, NUMA layout, I/O bottlenecks. | `lscpu`, `free -m`, `lsblk -o NAME,SIZE,TYPE,MOUNTPOINT` |
| 3 | **GPU / CUDA** | LLM inference is GPU‑bound; driver / toolkit mismatches kill performance. | `nvidia-smi`, `nvcc --version`, `cat /proc/driver/nvidia/version` |
| 4 | **Cloud Metadata (IMDS)** | Instance identity, tags, IAM role, and custom user data that may affect runtime. | `curl -s http://169.254.169.254/latest/meta-data/instance-id` (AWS) |
| 5 | **Python / ML Runtime** | Python version, virtualenv, pip packages, torch/cuda builds. | `python3 -V`, `pip freeze | grep -E 'torch|transformers|accelerate'` |
| 6 | **Inference Engine Specifics** | Server (vLLM, Text Generation Inference, etc.) config, model path, quantization flags. | `cat /etc/vllm/config.yaml`, `ps -eo pid,cmd | grep -i vllm` |
| 7 | **Docker / Containers** | Container IDs, image digests, runtime options (restart policies, resource limits). | `docker ps --format "{{.ID}} {{.Image}} {{.Status}}"`, `docker inspect <id>` |
| 8 | **Other Related Processes** | Side‑car processes (model cache daemons, monitoring agents). | `ps -eo pid,cmd --sort=-%mem | head -n 20` |
| 9 | **Web / Reverse Proxy** | Nginx/Traefik/Envoy configs, TLS certs, upstream health. | `nginx -T`, `systemctl status traefik` |
|10 | **Databases / Vector Stores** | Persistence layer health, version, connection strings. | `pg_isready -d $POSTGRES_DB`, `redis-cli INFO` |
|11 | **Networking** | Interface stats, firewall rules, MTU, DNS resolver. | `ip -br a`, `iptables -L -n -v`, `cat /etc/resolv.conf` |
|12 | **Storage Footprint** | Disk usage per mount, model cache size, log rotation. | `du -sh /var/lib/models`, `df -h` |
|13 | **systemd Services** | Service definitions, enable/disable status, recent logs. | `systemctl list-units --type=service --state=running`, `journalctl -u vllm -n 50 --no-pager` |
|14 | **Cron Jobs** | Periodic jobs that may purge caches or rotate logs. | `crontab -l`, `ls -l /etc/cron.d` |
|15 | **Users / SSH** | Authorized keys, sudoers, recent login attempts. | `cat /etc/passwd | grep -E '^llm'`, `grep -i 'ssh' /var/log/auth.log | tail -n 20` |

> **Key principle:** *Capture everything, even if you think it’s irrelevant.* The next section explains why.

## Collecting the “Noise” – Why You Gather What You Don’t Suspect  

In one production incident we inherited a fleet of inference nodes from a previous team. The obvious suspects were:

* GPU driver version mismatch  
* Wrong `torch` CUDA build  

Both were **correct**. The real culprit was an environment variable that the previous team had added to a **systemd drop‑in**:  

```bash
# /etc/systemd/system/vllm.service.d/override.conf
[Service]
Environment="OLLAMA_REMOTES=10.0.0.5:11434"
```

The variable forced the vLLM process to route all outbound HTTP traffic through an internal proxy that had been decommissioned. The model server kept timing out, but the logs only showed “connection refused”. Because we had **captured every environment block** (section 13) and **the full list of systemd drop‑ins** (section 1), we spotted the stray line within minutes.

### The “Unexpected Field” Effect  

| Symptom | Suspected cause | Unexpected field that mattered |
|---------|----------------|--------------------------------|
| High latency (30 s) | GPU throttling | `CUDA_VISIBLE_DEVICES=""` set in a user’s `.bashrc` |
| Model not loading | Disk full | `OLLAMA_REMOTES` pointing to dead proxy |
| Random crashes | Out‑of‑memory | `ulimit -n` (open file limit) set to 1024 in `/etc/security/limits.conf` |
| 502 from reverse proxy | Nginx config | `proxy_read_timeout` overridden in a per‑service include |

If you only ask “what’s the driver version?” you’ll miss the *proxy* that silently reroutes traffic. The diagnostic script is your **full‑body scan**; the “unexpected fields” are the hidden tumors that only a comprehensive panel can reveal.

## A Worked Example – Building the Capture Script  

Below is a **complete Bash script** (`capture_diagnostics.sh`) that implements the 15‑section checklist. It writes a **single, timestamped file** (`diagnostics-$(hostname)-$(date +%Y%m%d%H%M%S).log`) that typically exceeds **800 lines** on a typical LLM node.

```bash
#!/usr/bin/env bash
# capture_diagnostics.sh – one‑shot diagnostic panel for LLM infra
set -euo pipefail

HOST=$(hostname)
TS=$(date +%Y%m%d%H%M%S)
OUT="diagnostics-${HOST}-${TS}.log"

# Helper to print a section header
section() {
    echo -e "\n===== $1 =====\n" >> "$OUT"
}

# 1. Host & OS
section "01 – Host & OS"
{
    echo "Hostname: $(hostname)"
    cat /etc/os-release
    uname -a
    uptime -p
} >> "$OUT"

# 2. CPU / RAM / Disks
section "02 – CPU / RAM / Disks"
{
    lscpu
    free -m
    lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE,UUID
} >> "$OUT"

# 3. GPU / CUDA
section "03 – GPU / CUDA"
{
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi -L
        nvidia-smi
    else
        echo "No NVIDIA driver detected"
    fi
    if command -v nvcc &>/dev/null; then
        nvcc --version
    else
        echo "nvcc not installed"
    fi
    cat /proc/driver/nvidia/version || true
} >> "$OUT"

# 4. Cloud Metadata (IMDS)
section "04 – Cloud Metadata (IMDS)"
{
    # Detect AWS, Azure, GCP heuristically
    if curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id &>/dev/null; then
        echo "AWS detected"
        curl -s http://169.254.169.254/latest/meta-data/instance-id
        curl -s http://169.254.169.254/latest/meta-data/instance-type
        curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone
    elif curl -s --connect-timeout 2 http://169.254.169.254/metadata/instance?api-version=2021-02-01 &>/dev/null; then
        echo "Azure detected"
        curl -s "http://169.254.169.254/metadata/instance?api-version=2021-02-01" -H Metadata:true
    elif curl -s --connect-timeout 2 http://metadata.google.internal/computeMetadata/v1/instance/id -H "Metadata-Flavor: Google" &>/dev/null; then
        echo "GCP detected"
        curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/id
    else
        echo "No known cloud metadata endpoint reachable"
    fi
} >> "$OUT"

# 5. Python / ML Runtime
section "05 – Python / ML Runtime"
{
    python3 -V
    pip freeze | grep -E 'torch|transformers|accelerate|vllm|sentencepiece'
    # Show compiled CUDA version for torch
    python3 - <<'PY'
import torch, sys
print(f"torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
PY
} >> "$OUT"

# 6. Inference Engine Specifics
section "06 – Inference Engine Specifics"
{
    # Example for vLLM
    if pgrep -f vllm &>/dev/null; then
        ps -eo pid,cmd | grep -i vllm
        # Assume config lives in /etc/vllm/
        if [ -f /etc/vllm/config.yaml ]; then
            cat /etc/vllm/config.yaml
        else
            echo "No vLLM config file found"
        fi
    else
        echo "vLLM process not running"
    fi
} >> "$OUT"

# 7. Docker / Containers
section "07 – Docker / Containers"
{
    docker version || echo "Docker CLI not installed"
    docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}"
    # Dump inspect for each running container (limit to 5 for brevity)
    for cid in $(docker ps -q | head -n 5); do
        echo "--- docker inspect $cid ---"
        docker inspect "$cid"
    done
} >> "$OUT"

# 8. Other Related Processes
section "08 – Other Related Processes"
{
    ps -eo pid,user,%cpu,%mem,cmd --sort=-%mem | head -n 20
} >> "$OUT"

# 9. Web / Reverse Proxy
section "09 – Web / Reverse Proxy"
{
    if command -v nginx &>/dev/null; then
        echo "=== nginx -T (full config) ==="
        nginx -T
        systemctl status nginx
    elif command -v traefik &>/dev/null; then
        traefik version
        cat /etc/traefik/traefik.yaml
    else
        echo "No known reverse proxy detected"
    fi
} >> "$OUT"

#10. Databases / Vector Stores
section "10 – Databases / Vector Stores"
{
    # PostgreSQL
    if pg_isready -q; then
        echo "PostgreSQL reachable"
        pg_isready -d "$POSTGRES_DB" -U "$POSTGRES_USER"
        psql -c "\l" -U "$POSTGRES_USER"
    else
        echo "PostgreSQL not reachable"
    fi
    # Redis
    if redis-cli ping &>/dev/null; then
        echo "Redis reachable"
        redis-cli INFO | head -n 20
    else
        echo "Redis not reachable"
    fi
    # Milvus (example vector store)
    if curl -s http://localhost:19530/api/v1/health | grep -q 'healthy'; then
        echo "Milvus health: healthy"
    else
        echo "Milvus not responding"
    fi
} >> "$OUT"

#11. Networking
section "11 – Networking"
{
    ip -br a
    ip -br r
    ss -tulpan | head -n 20
    iptables -L -n -v
    cat /etc/resolv.conf
} >> "$OUT"

#12. Storage Footprint
section "12 – Storage Footprint"
{
    df -hT
    du -sh /var/lib/models || echo "Model dir missing"
    du -sh /var/log | sort -h
    # Show logrotate config
    cat /etc/logrotate.conf
    ls -l /etc/logrotate.d
} >> "$OUT"

#13. systemd Services
section "13 – systemd Services"
{
    systemctl list-units --type=service --state=running | wc -l
    systemctl list-units --type=service --state=running | head -n 20
    # Show overrides for vllm (if any)
    if [ -d /etc/systemd/system/vllm.service.d ]; then
        echo "=== vllm.service.d overrides ==="
        cat /etc/systemd/system/vllm.service.d/*.conf
    fi
    # Recent logs (last 200 lines)
    journalctl -u vllm -n 200 --no-pager
} >> "$OUT"

#14. Cron Jobs
section "14 – Cron Jobs"
{
    echo "=== root crontab ==="
    crontab -l || echo "No root crontab"
    echo "=== /etc/cron.d ==="
    ls -l /etc/cron.d
    cat /etc/cron.d/* 2>/dev/null || echo "No files in /etc/cron.d"
} >> "$OUT"

#15. Users / SSH
section "15 – Users / SSH"
{
    echo "=== /etc/passwd (llm‑related users) ==="
    grep -E 'llm|inference|model' /etc/passwd || echo "No special users found"
    echo "=== Authorized keys for llm user ==="
    if id -u llm &>/dev/null; then
        cat ~llm/.ssh/authorized_keys || echo "No keys"
    fi
    echo "=== Recent SSH logins ==="
    grep -i 'sshd' /var/log/auth.log | tail -n 20 || echo "auth.log not present"
} >> "$OUT"

# Final summary line
echo -e "\n=== Capture complete: $OUT ===\n"
```

### How the Script Generates 800+ Lines  

Running the script on a typical 8‑GPU inference node yields output similar to the excerpt below (full file: **822 lines**). The line count comes from:

* `lsblk` (≈30 lines for each block device)
* `docker inspect` for every container (≈15 lines per container × 4 containers)
* `nginx -T` (≈120 lines of config)
* `journalctl -u vllm -n 200` (200 lines of logs)
* `ps -eo …` (≈25 lines)
* `systemctl list‑units` (≈30 lines)

> **Tip:** If you need a deterministic line count for compliance, add `wc -l "$OUT"` at the end of the script.

#### Sample excerpt (first 30 lines)

```text
===== 01 – Host & OS =====

Hostname: llm-infer-01
NAME="Ubuntu"
VERSION="22.04.3 LTS (Jammy Jellyfish)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 22.04.3 LTS"
VERSION_ID="22.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=jammy
UBUNTU_CODENAME=jammy
Linux llm-infer-01 5.15.0-1063-aws #63~22.04.1-Ubuntu SMP Thu Sep 28 12:04:27 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
Uptime: 3 days, 4 hours, 12 minutes

===== 02 – CPU / RAM / Disks =====

Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              96
On-line CPU(s) list: 0-95
Thread(s) per core:  2
Core(s) per socket:  24
Socket(s):           2
Vendor ID:           GenuineIntel
Model name:          Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
CPU MHz:             2800.000
L1d cache:           32K
L1i cache:           32K
L2 cache:            1024K
L3 cache:            33792K
NUMA node0 CPU(s):   0-47
NUMA node1 CPU(s):   48-95

MemTotal:        251658240 kB
MemFree:          12345678 kB
MemAvailable:    21098765 kB
Buffers:           123456 kB
Cached:           3456789 kB
SwapTotal:        0 kB
SwapFree:         0 kB

NAME   SIZE TYPE MOUNTPOINT FSTYPE UUID
nvme0n1  1.8T disk 
├─nvme0n1p1  512M part /boot/efi ext4 1234-ABCD
├─nvme0n1p2  128M part /boot     ext4 5678-EFGH
└─nvme0n1p3  1.8T part /          xfs 9ABC-DEF0

...
```

The rest of the file continues through sections 3‑15, each clearly delimited by the `===== N – Section Name =====` header. Because the script **never prompts** and writes directly to a file, you can run it on dozens of hosts in parallel and collect a uniform set of artifacts.

## Making Sense of the Output  

### 1. First Pass – “Read Top‑to‑Bottom for the Big Picture”

Open the file in a pager (`less -S diagnostics-*.log`) and scroll **slowly**. The header order is intentional:

* **Host & OS** tells you the baseline (e.g., Ubuntu 22.04 on AWS m6i.24xlarge).  
* **CPU / RAM / Disks** reveals whether you have enough memory to hold the model weights (e.g., 250 GiB RAM, 2 TiB NVMe).  
* **GPU / CUDA** confirms driver‑toolkit compatibility (e.g., driver 525.85, CUDA 12.1).  

If any of these three top sections show a red flag (kernel older than 5.15, driver version mismatched to CUDA toolkit), you can stop early—most LLM failures trace back to these layers.

### 2. Second Pass – “Scan for Anomalies”

Now that you have the context, look for **outliers**. Below is a checklist of common anomalies and the sections where they appear.

| Anomaly | Where to look | Example detection |
|---------|----------------|-------------------|
| Unmounted model cache disk | Section 12 (Storage Footprint) | `du -sh /var/lib/models` shows 0 B, but `df -h` shows a 2 TiB mount point not listed |
| Container restart policy set to `no` | Section 7 (Docker) | `docker inspect <id>` → `"RestartPolicy":{"Name":"no"}` |
| FlashAttention disabled via env var | Section 13 (systemd) | `Environment="FLASH_ATTENTION=0"` in `vllm.service.d/override.conf` |
| Two different NVIDIA driver versions reported | Section 3 (GPU) | `nvidia-smi` shows driver 525.85, `cat /proc/driver/nvidia/version` shows 525.84 |
| Unexpected proxy variable (`OLLAMA_REMOTES`) | Section 13 (systemd) or Section 5 (Python env) | `grep -R OLLAMA_REMOTES /etc/systemd /etc/profile*` |
| Open file limit too low | Section 13 (systemd) or Section 15 (Users) | `ulimit -n` (via `systemctl show vllm.service -p LimitNOFILE`) returns 1024 |
| Stale IAM role attached to instance | Section 4 (IMDS) | Instance metadata shows role `llm-old-role` that no longer has S3 read permission |
| Cron job that wipes `/var/lib/models` nightly | Section 14 (Cron) | `cat /etc/cron.d/model-cleanup` contains `rm -rf /var/lib/models/*` |
| Reverse proxy TLS cert expired | Section 9 (Web) | `nginx -T` shows `ssl_certificate` file with `Not After : Dec 31 23:59:59 2023 GMT` |

When you spot an anomaly, **copy the relevant snippet** into a separate “investigation” file and dig deeper. For example, if you see a restart policy of `no`, you might change it to `on-failure` and reload the daemon:

```bash
docker update --restart=on-failure <container-id>
systemctl daemon-reload
systemctl restart vllm
```

### 3. Correlating Across Sections  

Sometimes the root cause lives in the intersection of two sections. A classic pattern:

* **Section 3** reports driver 525.85, but **Section 5** shows `torch` compiled against CUDA 11.8 → **Mismatch** leads to “illegal memory access” errors.  

Resolution: reinstall `torch` with the same CUDA version as the driver, e.g.:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

Another cross‑section case:

* **Section 11** shows a firewall rule dropping outbound TCP 443, while **Section 6** logs “failed to fetch model from huggingface.co”.  

Fix: adjust `iptables`:

```bash
iptables -I OUTPUT -p tcp --dport 443 -j ACCEPT
service iptables save
```

## Live Capture vs. Snapshot During Reproduction  

### When to Use a Live Capture  

* **Intermittent failures** (e.g., occasional OOM) that only appear under load.  
* **Resource contention** that only manifests when the system is busy (GPU memory fragmentation).  
* **Security incidents** where you need to see active network connections and open file descriptors at the exact moment of the breach.

**How to do it:** Run the script **in the background** while you trigger the failure (e.g., send a request to the inference endpoint). Append a timestamp to each command output for later correlation.

```bash
# In one terminal
./capture_diagnostics.sh &> diagnostics-live-$(date +%s).log &
CAPTURE_PID=$!

# In another terminal, reproduce the issue
curl -X POST http://localhost:8000/generate -d '{"prompt":"Explain quantum"}'

# After reproduction, stop the capture
kill $CAPTURE_PID
```

You may also add `watch -n 5 "nvidia-smi"` to a separate log file to capture GPU utilization over time.

### When a Snapshot Suffices  

* **Boot‑time misconfigurations** (wrong systemd drop‑ins, missing environment variables).  
* **Version drift** (kernel, driver, library versions) that does not change during runtime.  
* **Post‑mortem analysis** after a crash where the host is still reachable but the service is down.

In these cases, a **single run** of the script (as shown above) gives you everything you need. No need to keep the process alive.

### Hybrid Approach  

For critical production clusters, schedule a **cron job** that runs the script every 6 hours and stores the output in a centralized S3 bucket. When an incident occurs, you can compare the latest snapshot with the one taken *just before* the failure (if you have a live capture). The diff often points directly to the change that triggered the problem.

```bash
# /etc/cron.d/diagnostic_capture
0 */6 * * * root /opt/diag/capture_diagnostics.sh >> /var/log/diagnostics/$(hostname)-$(date +\%Y\%m\%d\%H\%M).log 2>&1
```

## Packaging the Script for Team Consumption  

### 1. Make It Self‑Contained  

Bundle the script with a small wrapper that checks for required binaries (`docker`, `nvidia-smi`, `curl`) and prints a friendly warning if any are missing.

```bash
#!/usr/bin/env bash
# wrapper.sh – ensures environment before running capture_diagnostics.sh

REQUIRED=("docker" "nvidia-smi" "curl" "ps" "systemctl")
for cmd in "${REQUIRED[@]}"; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found in PATH. Install it before running the diagnostic script."
        exit 1
    fi
done

# Execute the real script
exec /opt/diag/capture_diagnostics.sh "$@"
```

### 2. Version the Output  

Add a header with script version, git commit hash, and a UUID. This makes every capture traceable back to the exact script revision.

```bash
echo "Script version: 1.3.2"
echo "Git commit: $(git -C /opt/diag rev-parse HEAD)"
echo "Capture UUID: $(uuidgen)"
```

### 3. Secure the Artifact  

The log contains **secrets** (API keys, SSH authorized keys). Store it in an encrypted bucket (e.g., AWS S3 with SSE‑KMS) and set a retention policy of 30 days.

```bash
aws s3 cp "$OUT" s3://company-ops/diagnostics/$HOST/$OUT --sse aws:kms --sse-kms-key-id alias/ops-key
```

## Extending the Checklist for Future LLM Stack Additions  

Your LLM stack will evolve—vector databases, LoRA adapters, custom tokenizers, and emerging hardware (e.g., AWS Trainium). The checklist is **extensible**:

| New component | Where to add | Example command |
|---------------|--------------|-----------------|
| **AWS Trainium (Neuron)** | After GPU section (3) | `neuron-ls` |
| **LoRA adapter files** | Inference Engine (6) | `find /opt/llm/adapter -type f -name "*.pt"` |
| **LangChain server** | Web/Reverse Proxy (9) | `ps -eo pid,cmd | grep -i langchain` |
| **Prometheus exporter** | Systemd Services (13) | `systemctl status prometheus-node-exporter` |
| **GPU temperature monitoring** | GPU (3) | `nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader` |

When you add a new section, keep the **header format** (`===== NN – Section Name =====`) so downstream parsers (Python, Go, or even simple `grep`) continue to work.

## Turning the Raw Log into Actionable Tickets  

After you have the 800‑plus line file, the next step is to **convert findings into tickets** for the appropriate owners (GPU team, infra, security). Here’s a lightweight Bash‑Python pipeline:

```bash
#!/usr/bin/env python3
import re, sys, json
log = sys.argv[1]

def extract(section, pattern):
    """Return first matching line in a given section."""
    sec_pat = re.compile(rf"===== {section:02d} – .*?=====\n(.*?)(?=\n=====|\Z)", re.S)
    sec_match = sec_pat.search(open(log).read())
    if not sec_match:
        return None
    return re.search(pattern, sec_match.group(1), re.M)

tickets = []

# Example: driver / torch mismatch
driver = extract(3, r"NVIDIA-SMI.*Driver Version:\s+(\S+)")
torch = extract(5, r"torch version:\s+(\S+).*CUDA runtime:\s+(\S+)",)
if driver and torch and driver.group(1) not in torch.group(2):
    tickets.append({
        "title": "CUDA driver / torch version mismatch",
        "severity": "high",
        "details": f"Driver {driver.group(1)} vs torch compiled for {torch.group(2)}"
    })

# Example: OLLAMA_REMOTES env var
ollama = extract(13, r"OLLAMA_REMOTES=(\S+)")
if ollama:
    tickets.append({
        "title": "Unexpected OLLAMA_REMOTES environment variable",
        "severity": "medium",
        "details": f"Value: {ollama.group(1)} – verify proxy is still valid"
    })

print(json.dumps(tickets, indent=2))
```

Running `python3 parse_log.py diagnostics-llm-infer-01-20240520103045.log` yields a JSON array that can be fed into your ticketing system (Jira, ServiceNow) via API. This automates the “scan for anomalies” step and ensures nothing slips through the cracks.

## TL;DR Cheat Sheet (for the busy engineer)

| Goal | Command | Output file | When to run |
|------|---------|-------------|-------------|
| Full diagnostic panel (baseline) | `./wrapper.sh` | `diagnostics-$(hostname)-$(date +%Y%m%d%H%M%S).log` | Routine health check, after any config change |
| Live capture during load test | `./wrapper.sh &> live-$(date +%s).log &` then trigger load | same file | Intermittent OOM / latency spikes |
| Periodic snapshot for audit | `0 */6 * * * root /opt/diag/wrapper.sh` (cron) | timestamped file in S3 | Compliance, drift detection |
| Automated anomaly extraction | `python3 parse_log.py <logfile>` | JSON tickets | Post‑mortem, escalation |

---

By turning the **one‑question‑at‑a‑time** nightmare into a **single, reproducible, 800‑line capture**, you give yourself and every downstream stakeholder a complete, searchable picture of the LLM host. The script’s modular sections make it easy to extend as the stack evolves, and the accompanying parsing snippet bridges the gap from raw text to actionable work items.  

Now you can stop spending half a day on ping‑pong and start spending that time **fixing** the real issues that keep your LLM services reliable, performant, and secure.