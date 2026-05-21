# Failure Modes of Self‑Hosted Inference: A Field Guide

You’ve just spun up a GPU‑backed inference service on‑prem or in the cloud. The model loads, the API endpoint appears, and you fire the first request. Then the logs start spitting out errors, latency spikes, or the container simply restarts. In a production environment those moments feel like a pilot’s “engine fire” alarm at 3 a.m.—you need a quick, reliable reference that tells you what’s wrong and how to bring the aircraft back to the runway.

This guide is that reference. We start with the building blocks that make a self‑hosted inference engine tick, then walk through ten of the most common failure modes. For each mode you’ll see:

* **Symptom** – what you actually observe (log line, HTTP status, metric spike).  
* **Root cause** – the underlying technical condition.  
* **How to diagnose** – concrete steps, commands, and log snippets that confirm the cause.  
* **How to fix** – a concise, reproducible remediation plan (including exact CLI flags, config changes, or system‑admin actions).

Think of this as the “troubleshooting chapter” of an aircraft manual: you locate the symptom, flip to the indexed entry, and follow the checklist. Keep it bookmarked; you’ll reference it many times.

---

## Foundations: How a Self‑Hosted Inference Engine Works

Before diving into the failure catalog, let’s make sure you understand the three layers that most inference stacks sit on:

| Layer | What it does | Typical component |
|-------|--------------|-------------------|
| **Hardware allocation** | Reserves GPU memory, creates CUDA streams, binds devices to containers. | NVIDIA driver, Docker `--gpus`, `nvidia-container-runtime`. |
| **Model runtime** | Loads model weights, builds attention KV cache, serves request‑level inference. | vLLM, TensorRT‑LLM, Ollama, DeepSpeed‑Inference. |
| **Serving façade** | Exposes HTTP/REST or OpenAI‑compatible endpoints, handles request routing, health checks. | FastAPI, gRPC server, OpenAI‑compatible gateway. |

A useful analogy is a **kitchen**:

* **Hardware allocation** is the **counter space** and **appliances** you have. If the counter is too small, you’ll keep bumping dishes off the edge.
* **Model runtime** is the **chef** preparing the dish. The chef needs the right ingredients (weights) and a clean workspace (KV cache) to work efficiently.
* **Serving façade** is the **waitstaff** taking orders and delivering plates. If the waitstaff can’t reach the kitchen (network block), diners get angry.

When any of these layers mis‑behave, the whole service stalls. The catalog below isolates the most frequent “kitchen disasters” you’ll encounter.

---

## 1. VRAM OOM (Out‑of‑Memory)

### Symptom
```
Traceback (most recent call last):
  File "/app/vllm/engine.py", line 412, in forward
RuntimeError: CUDA out of memory. Tried to allocate 2.34 GiB.
```
GPU utilization in `nvidia-smi` hovers at 99 % with **Memory‑Used** equal to **Memory‑Total**.

### Root Cause
The sum of **model weights** + **KV cache** (key/value tensors for each token in the current context) exceeds the available VRAM. The KV cache grows linearly with the context length and the number of parallel beams. For a 70 B model (≈ 140 GiB FP16 weights) you typically need at least 2× that VRAM for a full‑precision KV cache. If you try to run a 32‑k token prompt with 4‑way beam search on a single 24 GiB RTX 3090, the cache alone can demand > 30 GiB, pushing you over the limit.

### How to Diagnose
1. **Inspect GPU memory breakdown**:
   ```bash
   nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu \
              --format=csv,noheader,nounits
   ```
   Example output: `24576, 24200, 376, 99`. The free memory is only a few hundred MB.

2. **Check model size** (vLLM command):
   ```bash
   vllm list-models | grep -i 70b
   # Output: openai/gpt-70b-fp16 140GiB
   ```

3. **Log the KV cache footprint** (vLLM flag `--log-kv-size`):
   ```bash
   vllm serve openai/gpt-70b-fp16 --log-kv-size
   ```
   You’ll see lines like:
   ```
   [2024-05-20 12:01:03] KV cache size: 28.4 GiB (max_context_len=32768, num_kv_heads=128)
   ```

4. **Confirm that the failure occurs on the first request** (cold‑load) or after a few generations (cache buildup).

### How to Fix
| Fix | When to use | Command / Config |
|-----|-------------|------------------|
| **Quantize KV cache** (e.g., 8‑bit) | Large context, modest GPU | `vllm serve model_id --kv-cache-dtype=uint8` |
| **Cap the context length** | Prompt length controllable | `vllm serve model_id --max-model-len=8192` |
| **Reduce parallelism / beams** | Beam search not required | `vllm serve model_id --tensor-parallel-size=1 --max-num-batched-token=256` |
| **Swap to a smaller model** | Budget GPU, no quantization | `vllm serve openai/gpt-2-7b` |
| **Add more VRAM** (multi‑GPU, NVLink) | Scale‑out possible | `docker run --gpus '"device=0,1"' ...` |

**Example remediation** – you have a 24 GiB RTX 3090 and need to serve a 70 B model with a 4 k token context:

```bash
docker run -d \
  --gpus all \
  -e VLLM_KV_CACHE_DTYPE=uint8 \
  -e VLLM_MAX_MODEL_LEN=4096 \
  -p 8000:8000 \
  ghcr.io/vllm-project/vllm:latest \
  vllm serve openai/gpt-70b-fp16
```

After the restart, `nvidia-smi` shows **Memory‑Used** ≈ 19 GiB, leaving headroom for the cache.

---

## 2. KV Cache Spill to CPU

### Symptom
* Decoding latency jumps from ~30 ms/token to > 200 ms/token.  
* `nvidia-smi` shows **Memory‑Used** well below the GPU total (e.g., 12 GiB/24 GiB).  
* Engine logs contain: `KV cache spilled to host memory (size=8.3 GiB)`.

### Root Cause
When the KV cache grows beyond the pre‑allocated GPU pool, the runtime falls back to **host RAM**. Every attention operation now has to copy keys and values across the PCIe bus, turning a fast in‑GPU memory access into a slow “cross‑road” transfer. Think of it like a **water pipe that overflows into a garden hose**: the pressure drops dramatically, and the flow becomes sluggish.

### How to Diagnose
1. **Search logs for “spill”**:
   ```bash
   grep -i "spill" /var/log/vllm/*.log
   ```
2. **Measure PCIe throughput** (using `nvidia-smi dmon`):
   ```bash
   nvidia-smi dmon -s p
   # Look for high "PCIe Rx/Tx" values during generation.
   ```
3. **Confirm that VRAM is not fully used** (see previous OOM diagnosis).  

4. **Check the runtime’s cache policy** (vLLM flag `--kv-cache-dtype` defaults to `fp16`):
   ```bash
   vllm serve model_id --kv-cache-dtype=fp16 --log-kv-size
   ```

### How to Fix
| Fix | Rationale | Command |
|-----|-----------|---------|
| **Reduce max context** | Less KV data stays on GPU | `vllm serve model_id --max-model-len=8192` |
| **Enable KV cache quantization** | Smaller per‑token footprint | `vllm serve model_id --kv-cache-dtype=uint8` |
| **Increase GPU memory pool** (if multi‑GPU) | More room for cache | `docker run --gpus '"device=0,1"' ...` |
| **Explicitly set a GPU‑only cache limit** | Prevent fallback | `vllm serve model_id --gpu-cache-size=16GiB` |
| **Upgrade PCIe bandwidth** (e.g., move from PCIe 3.0 x8 to PCIe 4.0 x16) | Mitigates spill impact | Hardware change; not a config fix. |

**Example** – you’re running a 13 B model with a 16 k token context on a single RTX 4090 (24 GiB). The cache spills at ~12 k tokens. Apply an 8‑bit KV cache:

```bash
docker exec -it inference_container bash
vllm serve openai/gpt-13b-fp16 \
  --max-model-len=16384 \
  --kv-cache-dtype=uint8 \
  --gpu-cache-size=20GiB
```

Latency drops back to ~45 ms/token, and the logs no longer mention “spilled”.

---

## 3. Cold‑Load Timeout

### Symptom
* Client receives HTTP 504 or “request timed out” on the **first** request after a container restart.  
* Subsequent requests succeed instantly.  
* Engine logs show `Model loading took 124.6s, exceeding OLLAMA_LOAD_TIMEOUT=60`.

### Root Cause
The inference engine needs to **deserialize weights** and **build CUDA kernels** before it can answer any request. If the client’s HTTP timeout (often 60 s) is shorter than the model load time, the client aborts the connection. The engine is still loading in the background, so the service appears “down” only for the initial call.

### How to Diagnose
1. **Measure model load time**:
   ```bash
   time docker logs inference_container 2>&1 | grep "Model loading took"
   ```
2. **Check client timeout** (e.g., `curl -m 30`, Python `requests.timeout`).  
3. **Inspect health‑check endpoint** (if you have one). It should return “loading” rather than “unhealthy”.

### How to Fix
| Fix | When to use | Implementation |
|-----|-------------|----------------|
| **Warm the model after boot** | You control the container lifecycle | Add an entrypoint script that sends a dummy request (`curl http://localhost:8000/v1/models`) after the service starts. |
| **Increase the client timeout** | Client is under your control | `curl -m 180 http://host:8000/v1/completions` |
| **Raise the engine’s load timeout** (Ollama example) | Engine exposes a flag | `OLLAMA_LOAD_TIMEOUT=300 ollama serve` |
| **Persist a serialized checkpoint** (e.g., `torch.save` with `torch.load` on GPU) | Model supports it | `torch.save(model.state_dict(), "model.pt")` and load with `torch.load(..., map_location='cuda')` |
| **Use a smaller model for warm‑up** | You have a multi‑model deployment | Load a lightweight “warm‑up” model first, then swap to the heavy model. |

**Example** – you run Ollama in a Docker container:

```Dockerfile
FROM ollama/ollama:latest
ENV OLLAMA_LOAD_TIMEOUT=300
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

`entrypoint.sh`:

```bash
#!/usr/bin/env bash
set -e
ollama serve &
SERVER_PID=$!
# Wait for the HTTP port to be reachable
until curl -s http://localhost:11434/api/health | grep -q "ready"; do
  echo "Waiting for Ollama to be ready..."
  sleep 2
done
# Warm up with a tiny prompt
curl -s -X POST http://localhost:11434/api/generate \
  -d '{"model":"llama2","prompt":"Hello"}' > /dev/null
wait $SERVER_PID
```

Now the first real client request never times out.

---

## 4. Model Not Found (404)

### Symptom
```
GET /v1/models/gpt-oss:120b HTTP/1.1
...
HTTP/1.1 404 Not Found
{
  "error": "model gpt-oss:120b not found"
}
```
The server’s health endpoint (`/v1/models`) lists `openai/gpt-oss-120b`, but the client uses `gpt-oss:120b`.

### Root Cause
A mismatch between the **served model name** (the identifier the engine registers) and the **client’s request tag**. Many runtimes (vLLM, Ollama, OpenAI‑compatible servers) automatically derive the served name from the repository path. If you mount a model at `/models/openai/gpt-oss-120b` but start the server with `--served-model-name=gpt-oss:120b`, the engine will expose `gpt-oss:120b`. Conversely, omitting `--served-model-name` leaves the default `openai/gpt-oss-120b`. The client’s 404 is simply a “name‑lookup” failure, not a missing file.

### How to Diagnose
1. **List served models** (vLLM example):
   ```bash
   curl -s http://localhost:8000/v1/models | jq .
   ```
   Output:
   ```json
   {"data":[{"id":"openai/gpt-oss-120b","object":"model"}]}
   ```

2. **Inspect the server start command**:
   ```bash
   docker inspect inference_container --format '{{.Config.Env}}'
   ```
   Look for `VLLM_SERVED_MODEL_NAME` or `--served-model-name`.

3. **Check client request** – ensure the `model` field matches exactly.

### How to Fix
| Fix | Command |
|-----|---------|
| **Explicitly set the served name** (vLLM) | `vllm serve openai/gpt-oss-120b --served-model-name=gpt-oss:120b` |
| **Adjust client request** | `"model": "openai/gpt-oss-120b"` |
| **Rename the model directory** (if using file‑system mount) | `mv /models/openai/gpt-oss-120b /models/gpt-oss:120b` |
| **Add an alias mapping** (some servers support a mapping file) | `model_aliases.yaml: {"gpt-oss:120b": "openai/gpt-oss-120b"}` |

**Example fix** – you’re using Ollama with a custom model folder:

```bash
docker run -d \
  -v /data/models:/root/.ollama/models \
  -e OLLAMA_SERVE_MODEL_NAME="gpt-oss:120b" \
  ollama/ollama:latest \
  ollama serve
```

Now a client request with `"model":"gpt-oss:120b"` succeeds.

---

## 5. Crash Loop After Image Upgrade

### Symptom
```
2024-05-20T12:45:01.123Z  INFO  Container started
2024-05-20T12:45:02.456Z  ERROR Failed to load weights: FileNotFoundError: 'model.safetensors' not found
2024-05-20T12:45:02.457Z  INFO  Container exiting with code 1
```
Docker shows `RestartCount: 5` and the container keeps cycling.

### Root Cause
An **image upgrade** introduced a new runtime version that expects a different directory layout, a newer PyTorch version, or a different CUDA ABI. Common triggers:

* **Config incompatibility** – new flag defaults that clash with existing environment variables.  
* **Missing weight file** – the upgraded image changed the expected file extension (`.pt` → `.safetensors`).  
* **Init‑time bug** – a regression in the entrypoint script that fails when a certain environment variable is absent.

The crash loop is analogous to a **car that won’t start after you replace the battery but forget to reconnect the ground wire** – the engine turns over, immediately stalls, and the starter retries.

### How to Diagnose
1. **Compare the old and new image tags**:
   ```bash
   docker images | grep inference
   # old: myorg/inference:1.4.0   new: myorg/inference:1.5.0
   ```

2. **Read the container’s entrypoint logs** (often `/app/startup.log`):
   ```bash
   docker exec -it inference_container cat /app/startup.log
   ```

3. **Check the runtime’s expected model path**:
   ```bash
   docker exec -it inference_container bash -c 'python -c "import vllm; print(vllm.__file__)"'
   ```

4. **Validate the presence of weight files** inside the mounted volume:
   ```bash
   ls -l /mnt/models/gpt-2-7b/
   ```

### How to Fix
| Fix | When applicable | Steps |
|-----|-----------------|-------|
| **Rollback to previous image** | Immediate recovery needed | `docker pull myorg/inference:1.4.0 && docker compose up -d` |
| **Adjust volume mount** (e.g., mount the correct subdirectory) | Layout change | `-v /data/models/gpt-2-7b:/models` |
| **Set required env vars** (e.g., `VLLM_MODEL_PATH`) | Config mismatch | `-e VLLM_MODEL_PATH=/models/gpt-2-7b` |
| **Install missing dependencies** (e.g., `pip install safetensors`) | New runtime requirement | `docker exec inference_container pip install safetensors` |
| **Patch entrypoint** (add a guard for missing files) | Bug in new version | Edit `/app/entrypoint.sh` to `[[ -f $MODEL_PATH ]] || exit 1` |

**Example rollback**:

```bash
docker compose down
docker pull myorg/inference:1.4.0
docker compose up -d
```

After the rollback, `docker ps` shows `RestartCount: 0` and the service is healthy again.

---

## 6. NSG/Firewall Blocking Inter‑Host Traffic

### Symptom
* From a downstream microservice you see `curl: (7) Failed to connect to 10.2.5.12 port 8000: Connection refused`.  
* `telnet 10.2.5.12 8000` times out.  
* Inside the inference container, `netstat -tlnp` shows the service listening on `0.0.0.0:8000`.

### Root Cause
Network Security Groups (NSG) or host‑level firewalls (iptables, Azure NSG, AWS Security Group) deny traffic on the inference port. The container itself is fine; the **network perimeter** is blocking the packets. This is similar to a **gatekeeper at a building’s front door** who only lets in people with a badge—your service has a badge, but the gate’s rules don’t list the badge’s ID.

### How to Diagnose
1. **Check the host’s firewall**:
   ```bash
   sudo iptables -L -n | grep 8000
   ```
2. **Inspect cloud‑provider security groups** (AWS example):
   ```bash
   aws ec2 describe-security-groups --group-ids sg-0abc1234
   ```
3. **Run a traceroute from the client**:
   ```bash
   traceroute -T -p 8000 10.2.5.12
   ```
   If the trace stops at the first hop, the block is at the host or NSG level.

4. **Verify the container’s network mode**:
   ```bash
   docker inspect inference_container --format '{{.HostConfig.NetworkMode}}'
   ```

### How to Fix
| Fix | Command / Action |
|-----|------------------|
| **Open the port in the cloud NSG** | AWS: `aws ec2 authorize-security-group-ingress --group-id sg-0abc1234 --protocol tcp --port 8000 --cidr 10.2.0.0/16` |
| **Add an iptables rule** | `sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT` |
| **Co‑locate dependent services** (same VPC/subnet) | Redesign architecture to avoid cross‑subnet traffic |
| **Use a Service Mesh (Istio) with proper `DestinationRule`** | `istioctl create -f dr.yaml` |
| **Expose the port via a Load Balancer** | Create an Azure Load Balancer rule forwarding 80 → 8000 |

**Example fix on Azure**:

```bash
az network nsg rule create \
  --resource-group rg-inference \
  --nsg-name nsg-inference \
  --name AllowInferencePort \
  --protocol Tcp \
  --direction Inbound \
  --priority 100 \
  --source-address-prefixes 10.0.0.0/16 \
  --destination-port-ranges 8000 \
  --access Allow
```

After the rule propagates (usually < 30 s), the downstream service can reach the inference endpoint.

---

## 7. `fstab` Race on Boot

### Symptom
* After a VM restart, the container fails with `FileNotFoundError: '/models/gpt-2-7b/weight.safetensors'`.  
* Manually running `mount -a` and then restarting the container makes it work.

### Root Cause
The **mount point for the model storage** (often an NFS or network‑attached volume) is declared in `/etc/fstab` without proper ordering. During boot, the container starts before the mount is ready, leading to a “missing directory” error. This is akin to a **restaurant opening before the pantry doors are unlocked**—the chef can’t find ingredients and aborts.

### How to Diagnose
1. **Check `systemd-analyze blame`** for mount times:
   ```bash
   systemd-analyze blame | grep mnt
   ```
2. **Inspect the container’s start order**:
   ```bash
   docker ps -a --filter "status=exited"
   ```
3. **Look at `journalctl -u docker.service`** for “Mount point not found” messages.

### How to Fix
| Fix | Explanation |
|-----|-------------|
| **Add `x-systemd.requires-mounts-for=/models`** to the service unit | Guarantees Docker starts after the mount. |
| **Add `nofail` to the fstab entry** | Allows boot to continue even if the mount is delayed; container can retry. |
| **Use a systemd mount unit** (`/etc/systemd/system/mnt-models.mount`) with `WantedBy=local-fs.target`. |
| **Add a health‑check script** that loops until the directory exists before launching the inference server. |

**Example `fstab` entry with proper ordering**:

```
# /etc/fstab
nfs-server:/export/models  /mnt/models  nfs4  _netdev,x-systemd.requires=network-online.target,x-systemd.after=network-online.target,noauto  0 0
```

**Systemd unit for Docker** (`/etc/systemd/system/docker.service.d/override.conf`):

```ini
[Unit]
RequiresMountsFor=/mnt/models
```

Run `systemctl daemon-reload && systemctl restart docker` to apply. After reboot, the container sees the model directory immediately.

---

## 8. Ephemeral Disk Wiped on Deallocate‑Restart

### Symptom
* After stopping a cloud VM (deallocate) and starting it again, the inference container logs `Model directory empty, expected /data/models`.  
* A manual `ls /data/models` shows the directory is empty.

### Root Cause
Many cloud providers attach **ephemeral NVMe disks** that are cleared when the VM is deallocated. If your model resides on that disk, the data disappears on every stop/start cycle. The situation mirrors a **pop‑up food truck that loses its pantry each night because the trailer is emptied**. The service works while the truck is running, but after you park it, the pantry is gone.

### How to Diagnose
1. **Identify the disk type**:
   ```bash
   lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
   ```
   Look for `disk` entries with `TYPE=ephemeral` or `nvme0n1p1` mounted at `/data`.

2. **Check the VM’s lifecycle events** in the cloud console (e.g., “Deallocated” vs “Stopped”).

3. **Confirm model persistence on a separate volume**:
   ```bash
   df -h /data/models
   ```

### How to Fix
| Fix | Implementation |
|-----|----------------|
| **Store models on a durable volume** (e.g., Azure Managed Disk, AWS EBS) | `az vm disk attach -g rg -n vm --new --size-gb 200 --sku Premium_LRS` |
| **Copy models at boot** (rsync from durable bucket) | Add a systemd service that runs `rsync -a /mnt/persistent/models/ /data/models/` |
| **Use a container image that bundles the model** (if size permits) | `docker build -t inference:with-model .` |
| **Configure the cloud provider to retain local SSD on stop** (some providers offer “stop‑preserve‑disk”) | Set `preserveDataOnStop=true` in the VM definition. |

**Example boot‑time sync script** (`/etc/systemd/system/model-sync.service`):

```ini
[Unit]
Description=Sync inference models from durable storage
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/bin/rsync -a --delete /mnt/persistent/models/ /data/models/
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable it with `systemctl enable model-sync.service`. Now each start repopulates `/data/models` automatically.

---

## 9. Driver / CUDA Mismatch

### Symptom
* Inference runs, but latency is 5× higher than expected.  
* `nvidia-smi` shows driver version **525.85.12**, while the container’s `nvcc --version` reports **CUDA 12.2**.  
* Occasionally you see `CUDA error: unknown error` or `CUDA capability mismatch`.

### Root Cause
The **host NVIDIA driver** must be **newer or equal** to the CUDA runtime baked into the container. If the container expects a newer driver (e.g., CUDA 12.2) but the host only provides 525 (CUDA 12.0), the kernel falls back to a compatibility shim, dramatically reducing performance and sometimes emitting errors. Think of it as a **newer car model (runtime) trying to run on an older gasoline pump (driver)** – the engine can start, but it sputters.

### How to Diagnose
1. **Check driver version**:
   ```bash
   nvidia-smi --query-gpu=driver_version --format=csv,noheader
   ```
2. **Inspect container’s CUDA version**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi
   ```
   Look for the `CUDA Version` line.

3. **Run a small benchmark** (e.g., `torch.cuda.is_available()` and a matrix multiply) and compare to baseline numbers.

4. **Search logs for “CUDA capability mismatch”**.

### How to Fix
| Fix | Command |
|-----|---------|
| **Upgrade host driver** (Ubuntu) | `sudo apt-get update && sudo apt-get install -y nvidia-driver-560` |
| **Downgrade container CUDA** (use older base image) | `FROM nvidia/cuda:12.0-base-ubuntu22.04` |
| **Pin the runtime to a compatible driver** (Docker `--gpus all` automatically picks the host driver) | No action needed if driver ≥ runtime. |
| **Validate after change** | Re‑run `nvidia-smi` inside container; latency should drop to expected range. |

**Example driver upgrade on Ubuntu 22.04**:

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt-get update
sudo apt-get install -y nvidia-driver-560
sudo reboot
```

After reboot, `nvidia-smi` reports `Driver Version: 560.35.03`. Re‑launch the inference container; latency returns to baseline (~30 ms/token).

---

## 10. CRLF in Shell Scripts

### Symptom
```
docker: Error response from daemon: failed to create task for container: exec /app/startup.sh: no such file or directory
```
The file `/app/startup.sh` clearly exists (`ls -l /app/startup.sh` shows it).

### Root Cause
The script was edited on Windows and contains **CRLF line endings** (`\r\n`). The shebang line (`#!/usr/bin/env bash\r`) points the kernel to an interpreter named `/usr/bin/env bash\r`, which does not exist, so the exec fails. This is the same as trying to open a door with a key that has an extra tooth stuck on it—the key is there, but it won’t fit.

### How to Diagnose
1. **Inspect the file with `od`**:
   ```bash
   od -c /app/startup.sh | head
   ```
   Look for `\r` characters after `#!/usr/bin/env bash`.

2. **Run `file` utility**:
   ```bash
   file /app/startup.sh
   # Output: ASCII text, with CRLF line terminators
   ```

3. **Check Docker logs** for the exact exec error (as shown above).

### How to Fix
| Fix | Command |
|-----|---------|
| **Convert to LF** using `dos2unix` | `dos2unix /app/startup.sh` |
| **Re‑create the script with a Linux editor** (vim, nano) | Open and save; ensure `:set ff=unix` in vim. |
| **Add a build step that normalizes line endings** | `RUN find /app -name "*.sh" -exec dos2unix {} +` in Dockerfile |
| **Enforce Git’s `core.autocrlf`** | `git config --global core.autocrlf input` |

**Dockerfile snippet** that guarantees LF scripts:

```Dockerfile
FROM python:3.11-slim
COPY . /app
RUN apt-get update && apt-get install -y dos2unix && \
    find /app -name "*.sh" -exec dos2unix {} + && \
    chmod +x /app/startup.sh
ENTRYPOINT ["/app/startup.sh"]
```

After rebuilding and redeploying, the container starts cleanly.

---

## Putting It All Together: The Meta‑Skill of Symptom Matching

You now have ten indexed entries, each with a clear symptom → cause → diagnosis → fix pathway. The real power of this field guide lies in **pattern recognition**:

1. **Observe the first clue** (error message, metric spike, log line).  
2. **Map it to the closest catalog entry** – think of the symptom column as a set of “search keywords”.  
3. **Run the diagnostic checklist** verbatim; the commands are designed to be copy‑paste ready.  
4. **Apply the fix** that matches the root cause, then re‑run the original test (e.g., a curl request or a benchmark) to confirm resolution.

Because each entry is self‑contained, you can treat the guide like a **cheat sheet** you keep open in a split‑screen while you SSH into the host. When a new failure mode surfaces (say, a future “Tensor Core throttling” bug), you can add a new H2 section following the same template, and the guide will continue to grow organically.

---

### Quick Reference Table

| # | Failure Mode | Typical Symptom | Primary Fix |
|---|--------------|----------------|-------------|
| 1 | VRAM OOM | `CUDA out of memory` | Quantize KV cache, cap context, reduce parallelism |
| 2 | KV Cache Spill | Latency ↑, VRAM not maxed | Reduce context, enable KV quantization |
| 3 | Cold‑Load Timeout | First request 504 | Warm‑up request, raise `OLLAMA_LOAD_TIMEOUT` |
| 4 | Model Not Found | HTTP 404 | Align served model name with client tag |
| 5 | Crash Loop (Upgrade) | Container restarts, missing weight | Rollback image, fix mount/ENV, install deps |
| 6 | NSG/Firewall Block | `Connection refused` | Open port in security group / iptables |
| 7 | `fstab` Race | Missing files after boot | `RequiresMountsFor`, `nofail`, proper unit ordering |
| 8 | Ephemeral Disk Wiped | Model dir empty after deallocate | Store on durable disk, rsync at boot |
| 9 | Driver/CUDA Mismatch | 5× latency, `CUDA capability mismatch` | Upgrade host driver or downgrade container CUDA |
|10 | CRLF Scripts | `exec ... no such file` | Convert to LF (`dos2unix`) |

Keep this table handy for a **first‑pass triage**; then dive into the detailed sections for the exact commands you need.

--- 

You now have a field‑ready, catalog‑style troubleshooting manual for self‑hosted inference. When the night‑shift alarm sounds, flip to the relevant heading, run the diagnostic checklist, apply the fix, and get your model back to serving predictions at production speed. Happy debugging!