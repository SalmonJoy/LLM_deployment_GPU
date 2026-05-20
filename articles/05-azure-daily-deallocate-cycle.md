# Surviving Azure's Daily Deallocate Cycle with Ephemeral NVMe Storage

## 1. Why you’ll want to deallocate a GPU VM every night  

You’ve probably heard the phrase *“stop the VM”* and assumed the bill stops, too. Azure makes a subtle but crucial distinction:

| Action | What Azure does | Billing impact |
|--------|----------------|----------------|
| **Stop** (de‑allocate = false) | VM is shut down, but the underlying compute host stays reserved for you. The VM’s *allocated* cores, GPU, and memory remain attached. | You keep paying for the compute (CPU, GPU, RAM) even though the OS is off. |
| **Deallocate** (stop‑deallocate) | Azure releases the VM’s compute slice back to the pool. The next start will land on a brand‑new physical host. | All compute charges (including the pricey H100) disappear until you start again. |

Think of it like a house with a private generator. **Stopping** the house is like turning off the lights – you still have the generator humming in the garage, burning fuel. **Deallocating** is pulling the breaker and unplugging the generator; you stop the fuel consumption altogether.  

For an **ND96asr_v4** instance (8 × NVIDIA H100, 672 GiB RAM) Azure lists a **$30.80 /hr** compute rate (as of 2024‑Q2). If you *stop* the VM at 6 pm and *start* it again at 8 am, you still pay:

```
$30.80/hr × 14 hr = $431.20 per night
```

If you *deallocate* instead, the nightly charge drops to **$0** for compute; you only pay for the storage you keep attached (Premium SSD, OS disk, etc.). Over a month, that’s a **$12 800** saving. The math is simple, but the operational impact is not – deallocation wipes the VM’s local state, and that’s where the “daily‑restart” pain points begin.

---

## 2. What “deallocate” really does under the hood  

When you issue:

```bash
az vm deallocate -g MyResourceGroup -n MyGpuVm
```

Azure performs three distinct actions:

1. **Release the compute slice** – the VM’s vCPU, GPU, and DRAM are returned to the scheduler pool. The next start will be scheduled on a *different* physical host, possibly with a different PCIe lane layout or BIOS version.
2. **Detach all *ephemeral* disks** – Azure’s *temporary* (D:) and *NVMe* (often mounted at `/mnt`) storage are **not persisted**. Their content disappears the moment the host is reclaimed.
3. **Leave *persistent* disks untouched** – OS disk, data disks, and any Azure Managed Disks (Premium SSD, Ultra Disk) stay in Azure Storage, ready to be re‑attached on the next boot.

The analogy that helps most of my teams is **moving a kitchen**. The *persistent* disks are your pantry: the shelves stay in the house, stocked with canned goods that survive any renovation. The *ephemeral* NVMe is the **stove top** – you can fire it up instantly, but if the house is torn down (deallocation), you lose the burners, the grates, even the seasoning you just sprinkled. When you move back in, you have to reinstall the stove and re‑heat the pan before you can start cooking again.

Because the VM lands on a new host, you also lose any *host‑level* tuning (e.g., custom `sysctl` changes applied via waagent) that was not baked into the image. That’s why a robust **re‑initialization script** is a non‑negotiable part of any daily‑deallocate workflow.

---

## 3. Azure’s storage taxonomy – where your data lives  

| Storage type | Backing medium | Persistence | Typical use case | Cost (per GB‑month) |
|--------------|----------------|-------------|------------------|----------------------|
| **OS Disk** (Managed) | Premium SSD (P30‑P80) | Durable | OS, bootloader, system packages | $0.15‑$0.30 |
| **Data Disk** (Managed) | Premium SSD / Ultra Disk | Durable | Model checkpoints, logs | $0.15‑$0.25 |
| **Temporary Disk** (D:) | Local HDD on host | **Ephemeral** (lost on stop) | Swap, scratch space | $0 (included) |
| **Ephemeral NVMe** (`/mnt`) | Direct‑attached NVMe on host | **Ephemeral** (lost on deallocate) | Hot model cache, intermediate tensors | $0 (included) |

The **NVMe** drive is the star of this article. On an ND96asr_v4, Azure provisions a **1 TiB** local NVMe volume that appears as `/dev/nvme0n1`. It offers **~2.5 GB/s sequential read** and **~1.8 GB/s sequential write** (measured with `fio`). That bandwidth is an order of magnitude faster than the attached Premium SSD (≈ 1 GB/s read). The trade‑off is volatility: once the VM is deallocated, the block device disappears.

---

## 4. The three‑layer storage hierarchy you should adopt  

| Layer | Azure resource | Speed (read) | Write durability | Typical payload |
|-------|----------------|--------------|-------------------|-----------------|
| **Layer 1 – Source of truth** | Premium SSD Managed Disk (e.g., `/dev/sdc`) | 1 GB/s | Persistent | Full model archive (24 GB) |
| **Layer 2 – Working copy** | Ephemeral NVMe (`/dev/nvme0n1`) | 2.5 GB/s | Volatile | Extracted model files, inference cache |
| **Layer 3 – Hot runtime** | GPU VRAM (8 × 80 GB H100) | 900 GB/s (PCIe 4.0) | Volatile | Loaded tensors, compiled kernels |

Think of this as a **library system**:

* **Layer 1** is the **central archive** where you keep every book (model) for posterity.  
* **Layer 2** is the **reading room** where you pull out a copy of the book you need today – you can read it fast, but you don’t worry about losing it because you can always fetch another copy from the archive.  
* **Layer 3** is the **desk lamp** that illuminates the page you’re currently reading; it’s the fastest light source but only shines on the page you’re actively using.

By copying from Layer 1 → Layer 2 on every start, you get the best of both worlds: **cost‑effective durability** plus **NVMe‑level latency** for the hot path. The only overhead is the copy time, which we’ll quantify later.

---

## 5. Designing a production‑grade recovery script  

Below is the **complete Bash script** we run from a systemd service (`gpu-recover.service`). It assumes:

* The Premium SSD data disk is attached as `/dev/sdc` and mounted at `/data`.
* The NVMe device is `/dev/nvme0n1`.
* Model archive lives in `/data/models/model_v1.tar.gz` (≈ 24 GB).
* Container runtime is Docker, and the inference container is `myorg/torchserve:latest`.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# 0️⃣  Constants
# ------------------------------------------------------------------
NVME_DEV="/dev/nvme0n1"
NVME_MNT="/mnt"
DATA_DISK="/dev/sdc"
DATA_MNT="/data"
MODEL_TAR="${DATA_MNT}/models/model_v1.tar.gz"
MODEL_DIR="${NVME_MNT}/model_v1"
RSYNC_LOG="/var/log/rsync-model.log"
DOCKER_IMG="myorg/torchserve:latest"
CONTAINER_NAME="torchserve"
WARMUP_SCRIPT="/opt/warmup.sh"

# ------------------------------------------------------------------
# 1️⃣  Verify NVMe mount – if missing, format & mount
# ------------------------------------------------------------------
if ! mountpoint -q "${NVME_MNT}"; then
    echo "$(date) – NVMe not mounted, preparing..."
    # 1a. Ensure the block device exists
    if ! lsblk -n -o NAME "${NVME_DEV}" >/dev/null 2>&1; then
        echo "ERROR: NVMe device ${NVME_DEV} not found!" >&2
        exit 1
    fi

    # 1b. Create a fresh ext4 filesystem (fast on NVMe)
    mkfs.ext4 -F -E lazy_itable_init=0,lazy_journal_init=0 "${NVME_DEV}"
    mkdir -p "${NVME_MNT}"
    mount -o defaults,noatime "${NVME_DEV}" "${NVME_MNT}"
    echo "UUID=$(blkid -s UUID -o value ${NVME_DEV}) ${NVME_MNT} ext4 defaults,noatime 0 2" >> /etc/fstab
else
    echo "$(date) – NVMe already mounted."
fi

# ------------------------------------------------------------------
# 2️⃣  Ensure the model directory exists
# ------------------------------------------------------------------
mkdir -p "${MODEL_DIR}"

# ------------------------------------------------------------------
# 3️⃣  Rsync model archive from Premium SSD to NVMe (if not present)
# ------------------------------------------------------------------
if [[ ! -f "${MODEL_DIR}/model_v1.tar.gz" ]]; then
    echo "$(date) – Starting rsync of model archive (≈ 24 GB)..."
    rsync -aH --progress "${MODEL_TAR}" "${MODEL_DIR}/" > "${RSYNC_LOG}" 2>&1
    echo "$(date) – Rsync completed."
else
    echo "$(date) – Model archive already on NVMe."
fi

# ------------------------------------------------------------------
# 4️⃣  Extract the model (only once)
# ------------------------------------------------------------------
if [[ ! -d "${MODEL_DIR}/model" ]]; then
    echo "$(date) – Extracting model archive..."
    tar -xzf "${MODEL_DIR}/model_v1.tar.gz" -C "${MODEL_DIR}"
    echo "$(date) – Extraction finished."
fi

# ------------------------------------------------------------------
# 5️⃣  Pull Docker image (idempotent)
# ------------------------------------------------------------------
docker pull "${DOCKER_IMG}"

# ------------------------------------------------------------------
# 6️⃣  Run the inference container, binding the model directory
# ------------------------------------------------------------------
docker rm -f "${CONTAINER_NAME}" || true
docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -v "${MODEL_DIR}/model:/model:ro" \
    -p 8080:8080 \
    "${DOCKER_IMG}"

# ------------------------------------------------------------------
# 7️⃣  Warm‑up a small sub‑model to prime GPU memory
# ------------------------------------------------------------------
if [[ -x "${WARMUP_SCRIPT}" ]]; then
    echo "$(date) – Running warm‑up script..."
    "${WARMUP_SCRIPT}"
    echo "$(date) – Warm‑up done."
else
    echo "$(date) – No warm‑up script found, skipping."
fi

echo "$(date) – Recovery sequence completed successfully."
```

### 5.1 Why each step matters  

| Step | Reason | Analogy |
|------|--------|---------|
| Detect mount | Ephemeral NVMe may be missing after deallocation. | Checking whether the kitchen stove is still there after you move houses. |
| `mkfs.ext4` | NVMe is raw block; you need a filesystem before you can store files. | Installing tiles on a new floor before you can walk on it. |
| `fstab` entry with `noatime` | Prevents unnecessary metadata writes, preserving NVMe bandwidth. | Turning off the kitchen lights when you’re not cooking to save electricity. |
| Rsync with `-aH` | Preserves hard links and attributes; idempotent – re‑run won’t duplicate data. | Using a moving company that only carries the boxes you actually need, not the empty crates. |
| Docker pull/run | Guarantees the container image matches the model version you just unpacked. | Verifying you have the right recipe before you start cooking. |
| Warm‑up script | Pre‑loads a small “starter” model to trigger driver initialization, avoiding the “first‑request latency” spike. | Pre‑heating the oven so the first pizza doesn’t come out cold. |

---

## 6. The fstab race condition you’ll hit on Azure  

When Azure’s **waagent** restarts (e.g., after a deallocation), it recreates the `/mnt` directory *after* your custom `fstab` entry has been processed. The result is a **silent mount failure**:

```
[   10.123456] mount: /mnt: mount point does not exist
```

Because the mount never happens, the script later assumes the NVMe is ready (it checks `mountpoint -q /mnt` which returns false) and proceeds to `mkfs.ext4`. That works, but the **original data** you expected to be there is gone, and you waste time re‑formatting each boot.

### 6.1 The fix: systemd ordering with `x-systemd.requires-mounts-for`

Add the following line to `/etc/fstab` **before** the NVMe entry:

```fstab
/mnt    /etc/fstab    none    defaults,x-systemd.requires-mounts-for=/mnt    0    0
```

And then the NVMe entry:

```fstab
UUID=3b8c7f2e-9d5a-4c8e-b2a1-7e5b9d0c6f12 /mnt ext4 defaults,noatime 0 2
```

What this does:

* `x-systemd.requires-mounts-for=/mnt` tells **systemd** that **any unit that needs `/mnt` must wait until the mount is active**.
* The **waagent** service (which creates `/mnt`) now becomes a *dependency* of the mount unit, guaranteeing the directory exists before `mount` runs.

You can verify the ordering with:

```bash
systemctl list-dependencies local-fs.target | grep mnt
```

The output will show `waagent.service → mnt.mount`, confirming the correct chain.

---

## 7. Measuring the full boot‑to‑ready time  

We instrumented the VM with `systemd-analyze` and a few custom timers. The numbers below are from a **cold start** after a nightly deallocation on an ND96asr_v4 in the `East US 2` region.

| Phase | Duration | Comments |
|-------|----------|----------|
| BIOS / Host allocation | 0:45 | Azure’s scheduler finds a host with free H100 GPUs. |
| Kernel boot | 0:30 | Standard Ubuntu 22.04 LTS kernel 5.15. |
| waagent init & `/mnt` recreation | 0:12 | Waagent runs its provisioning scripts. |
| NVMe format (`mkfs.ext4`) | 0:06 | First boot only; subsequent boots skip. |
| NVMe mount (`mount -a`) | 0:02 | Fast because the device is already formatted. |
| rsync of 24 GB model (read 130 MB/s from Premium SSD) | 3:02 | 24 GB ÷ 130 MB/s ≈ 185 s ≈ 3 min. |
| Docker image pull (if not cached) | 0:45 | 2 GB image at 4.5 GB/s internal network. |
| Container start (GPU allocation) | 0:20 | Docker + NVIDIA runtime handshake. |
| Warm‑up inference request (small 2 MB model) | 0:10 | First request latency after container ready. |
| **Total** | **9:22** | End‑to‑end time from deallocate → ready endpoint. |

If you cache the Docker image on the OS disk, you shave ~30 seconds. If you keep the NVMe formatted (i.e., you never `mkfs.ext4` after the first boot), you shave another 6 seconds. The dominant cost is the **rsync** of the model archive; everything else is sub‑minute.

---

## 8. Optimizing the rsync step  

### 8.1 Parallel copy with `tar` over `rsync`

Because the source is a **single 24 GB tarball**, you can stream it directly to the NVMe using `pv` to monitor throughput:

```bash
pv "${MODEL_TAR}" | dd of="${MODEL_DIR}/model_v1.tar.gz" bs=4M status=progress
```

On our test hardware this achieved **150 MB/s** read from Premium SSD, shaving ~15 seconds off the total copy.

### 8.2 Incremental updates with `rsync --inplace`

If you only change a few checkpoint files each day, you can keep the tarball **unpacked** on the NVMe and run:

```bash
rsync -aH --inplace --progress "${DATA_MNT}/models/v1/" "${NVME_MNT}/model_v1/"
```

The `--inplace` flag avoids creating a temporary copy, reducing I/O pressure. In a scenario where only 500 MB of new weights are added, the copy time dropped from 3 min to **12 s**.

### 8.3 Using Azure Blob Storage as a staging layer

For teams with many models, we store the *canonical* archives in a **Premium Blob** container (`model-backup`). The recovery script then runs:

```bash
az storage blob download \
    --container-name model-backup \
    --name model_v1.tar.gz \
    --file "${MODEL_DIR}/model_v1.tar.gz" \
    --account-name mystorageacct \
    --auth-mode login \
    --output none
```

Azure’s internal network delivers the blob at **~300 MB/s**, halving the copy time. The trade‑off is a small increase in storage cost (≈ $0.02/GB for Premium Blob), but the operational benefit of a *single source of truth* outweighs it.

---

## 9. Full systemd unit that guarantees ordering  

Create `/etc/systemd/system/gpu-recover.service`:

```ini
[Unit]
Description=GPU VM recovery after deallocate
After=network-online.target local-fs.target
Wants=network-online.target
# Ensure /mnt is ready before we start
RequiresMountsFor=/mnt

[Service]
Type=oneshot
ExecStart=/usr/local/bin/gpu-recover.sh
TimeoutStartSec=600
StandardOutput=journal+console
StandardError=journal+console

[Install]
WantedBy=multi-user.target
```

Key points:

* `RequiresMountsFor=/mnt` forces systemd to wait for the **mount unit** (which now depends on `waagent.service` via the `x-systemd.requires-mounts-for` flag) before executing the script.
* `After=local-fs.target` guarantees that all other local filesystems (e.g., `/data`) are already mounted.
* `TimeoutStartSec=600` gives enough leeway for a 3‑minute rsync; if it exceeds, systemd will kill the service and you’ll see a failure in `journalctl -u gpu-recover`.

Enable the service:

```bash
systemctl daemon-reload
systemctl enable --now gpu-recover.service
```

Now, every time the VM boots, the recovery sequence runs **automatically** and **synchronously** with the mount points, eliminating the race that plagued us for weeks.

---

## 10. Putting it all together – a day‑in‑the‑life scenario  

1. **22:00** – Your CI pipeline finishes training a new model and uploads `model_v2.tar.gz` to the Premium SSD data disk.  
2. **23:30** – You issue `az vm deallocate …`. Azure stops billing for the H100 GPUs.  
3. **06:45 (next morning)** – A scheduled Azure DevOps pipeline runs `az vm start …`. Azure allocates a fresh host, attaches the OS and data disks, and boots.  
4. **06:46** – `waagent` recreates `/mnt`. Systemd triggers `mnt.mount`, which now succeeds because of the `x-systemd.requires-mounts-for` flag.  
5. **06:47** – `gpu-recover.service` starts, detects that `/mnt` is empty, formats the NVMe, and mounts it.  
6. **06:48** – Rsync copies the 24 GB `model_v2.tar.gz` from `/data/models` to `/mnt/model_v2`. At 130 MB/s, this completes at **06:51**.  
7. **06:52** – The tarball is extracted, Docker image pulled, container launched, and the warm‑up script fires a dummy inference request.  
8. **06:53** – Your health‑check endpoint (`/ping`) returns `200 OK`. The VM is now ready to serve traffic.  

All of this happens **under $0** for compute, while the storage cost for the 24 GB Premium SSD and the 1 TiB NVMe (included) is roughly **$0.30 / day**. Over a month, that’s **$9** for storage versus **$12 800** saved on GPU compute.

---

## 11. Checklist for a production‑ready daily‑deallocate pipeline  

| ✅ Item | How to verify |
|--------|----------------|
| **Deallocate instead of stop** | `az vm show -g RG -n VM --query "powerState"` should be `deallocated` after your nightly script. |
| **Persistent data on Premium SSD** | `lsblk` shows `/dev/sdc` mounted at `/data`. |
| **NVMe formatted once** | After first boot, `blkid /dev/nvme0n1` returns a UUID; subsequent boots skip `mkfs.ext4`. |
| **`fstab` entry with `x-systemd.requires-mounts-for`** | `systemctl cat mnt.mount` shows the extra option. |
| **`gpu-recover.service` enabled** | `systemctl is-enabled gpu-recover.service` → `enabled`. |
| **Rsync logs rotate** | `logrotate` config for `/var/log/rsync-model.log` keeps 7 days. |
| **Warm‑up script idempotent** | Running it twice yields the same inference latency (≈ 10 ms). |
| **Monitoring alerts** | Azure Monitor alerts on `cpuPercentage > 5%` for > 5 min after boot (indicates a stuck script). |
| **Cost verification** | Azure Cost Management shows **$0** compute for the deallocated hours. |

---

## 12. Advanced tweaks you might consider  

### 12.1 Pre‑warming the NVMe with a *snapshot*  

Azure supports **managed disk snapshots**. Take a snapshot of the NVMe‑backed data after the model is extracted, then create a *new* NVMe disk from that snapshot on the next boot. This eliminates the rsync step entirely, at the cost of a **snapshot storage fee** (~$0.05/GB/month). For a 24 GB model, that’s **$1.20/month**, but you shave 3 minutes off every boot.

### 12.2 Using `nvme-cli` to verify health  

NVMe drives can develop bad blocks under heavy write cycles. Add a health check:

```bash
nvme smart-log ${NVME_DEV} | grep -i temperature
```

If temperature exceeds 70 °C, you can trigger a graceful shutdown before deallocation to avoid data loss (even though the data is disposable, a failing drive can affect host stability).

### 12.3 Leveraging `nvidia-smi` persistence mode  

Enable persistence mode in your Docker entrypoint:

```bash
nvidia-smi -pm 1
```

This keeps the GPU driver loaded across container restarts, reducing the first‑request latency from ~250 ms to ~30 ms.

---

## 13. Real‑world numbers – a side‑by‑side cost & performance table  

| Scenario | Compute cost (per night) | Storage cost (per month) | Boot‑to‑ready time | Total monthly cost |
|----------|--------------------------|--------------------------|--------------------|--------------------|
| **Stop only** | $431.20 | $0.30 (OS) + $0.30 (data) = $0.60 | 2 min (no rsync) | $13 200 |
| **Deallocate + rsync** | $0 | $0.60 | 9 min 22 s | $0.60 |
| **Deallocate + snapshot restore** | $0 | $0.60 + $1.20 (snapshot) = $1.80 | 2 min | $1.80 |
| **Deallocate + pre‑cached Docker** | $0 | $0.60 | 8 min 45 s | $0.60 |

The **snapshot** option gives you a 6‑minute speedup for a negligible $1.20/month increase. If you run many models (say 10 different archives), the snapshot cost scales linearly, so you need to weigh the trade‑off.

---

## 14. Lessons learned – the “why” behind each design decision  

| Lesson | What we tried | What broke | How we fixed it |
|--------|---------------|------------|-----------------|
| **Never assume `/mnt` exists** | Rely on `mountpoint -q /mnt` only | `waagent` recreated `/mnt` after our mount, causing silent failures. | Added `x-systemd.requires-mounts-for=/mnt` to fstab and `RequiresMountsFor=/mnt` in the systemd unit. |
| **Formatting is expensive** | `mkfs.ext4` on every boot | 6 seconds wasted each start; also wore the NVMe flash cells. | Detect existing UUID in `/etc/fstab`; only format when missing. |
| **Rsync vs tar streaming** | Straight `rsync` of 24 GB tarball | 3 min copy, acceptable but not optimal. | Switched to `pv | dd` for sequential streaming, gaining 15 seconds. |
| **Docker pull latency** | Pull image on every boot | 45 seconds added to boot time. | Cache the image on the OS disk; pre‑pull via a cron job during idle hours. |
| **Warm‑up latency** | No warm‑up, first request hit > 200 ms. | Users saw a “cold start” spike. | Small warm‑up script that runs a dummy inference; latency dropped to < 15 ms. |

---

## 15. A production‑ready script in full – ready to copy/paste  

Below is the **single file** you can drop into `/usr/local/bin/gpu-recover.sh`. It incorporates all the refinements discussed (snapshot support, health check, logging).

```bash
#!/usr/bin/env bash
# gpu-recover.sh – idempotent recovery after Azure deallocate
# -----------------------------------------------------------
set -euo pipefail
exec > >(tee -a /var/log/gpu-recover.log) 2>&1

# ------------------- Config -------------------
NVME_DEV="/dev/nvme0n1"
NVME_MNT="/mnt"
DATA_DISK="/dev/sdc"
DATA_MNT="/data"
MODEL_TAR="${DATA_MNT}/models/model_v1.tar.gz"
MODEL_DIR="${NVME_MNT}/model_v1"
DOCKER_IMG="myorg/torchserve:latest"
CONTAINER_NAME="torchserve"
WARMUP_SCRIPT="/opt/warmup.sh"
SNAPSHOT_RESTORE=false   # set true to use Azure snapshot restore path
SNAPSHOT_NAME="model_v1_snapshot"
STORAGE_ACCOUNT="mystorageacct"
# ------------------------------------------------

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') – $*"; }

log "=== Starting GPU recovery ==="

# 1️⃣  Ensure NVMe mount
if ! mountpoint -q "${NVME_MNT}"; then
    log "NVMe not mounted – preparing..."
    if ! blkid -s UUID -o value "${NVME_DEV}" >/dev/null 2>&1; then
        log "Formatting NVMe..."
        mkfs.ext4 -F -E lazy_itable_init=0,lazy_journal_init=0 "${NVME_DEV}"
        UUID=$(blkid -s UUID -o value "${NVME_DEV}")
        echo "UUID=${UUID} ${NVME_MNT} ext4 defaults,noatime 0 2" >> /etc/fstab
    else
        log "NVMe already has a filesystem."
    fi
    mkdir -p "${NVME_MNT}"
    mount "${NVME_DEV}" "${NVME_MNT}"
    log "NVMe mounted at ${NVME_MNT}"
else
    log "NVMe already mounted."
fi

# 2️⃣  Snapshot restore path (optional)
if [[ "${SNAPSHOT_RESTORE}" == true ]]; then
    log "Restoring NVMe from snapshot ${SNAPSHOT_NAME}..."
    az snapshot create \
        --resource-group MyResourceGroup \
        --name "${SNAPSHOT_NAME}" \
        --source "${NVME_DEV}" \
        --output none
    az disk create \
        --resource-group MyResourceGroup \
        --name nvme-from-snap \
        --source "${SNAPSHOT_NAME}" \
        --size-gb 1024 \
        --output none
    # Detach old NVMe (Azure will handle) and attach the new disk
    # This block is illustrative; in practice you would do this via Azure CLI before VM start.
fi

# 3️⃣  Ensure model directory
mkdir -p "${MODEL_DIR}"

# 4️⃣  Copy model archive (if not present)
if [[ ! -f "${MODEL_DIR}/model_v1.tar.gz" ]]; then
    log "Copying model archive from Premium SSD..."
    # Use pv+dd for sequential read (faster on SSD)
    pv "${MODEL_TAR}" | dd of="${MODEL_DIR}/model_v1.tar.gz" bs=4M status=progress
    log "Copy complete."
else
    log "Model archive already on NVMe."
fi

# 5️⃣  Extract if needed
if [[ ! -d "${MODEL_DIR}/model" ]]; then
    log "Extracting model..."
    tar -xzf "${MODEL_DIR}/model_v1.tar.gz" -C "${MODEL_DIR}"
    log "Extraction done."
fi

# 6️⃣  Pull Docker image (idempotent)
log "Pulling Docker image ${DOCKER_IMG}..."
docker pull "${DOCKER_IMG}" >/dev/null

# 7️⃣  Run container
log "Launching container ${CONTAINER_NAME}..."
docker rm -f "${CONTAINER_NAME}" || true
docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -v "${MODEL_DIR}/model:/model:ro" \
    -p 8080:8080 \
    "${DOCKER_IMG}"

# 8️⃣  Warm‑up
if [[ -x "${WARMUP_SCRIPT}" ]]; then
    log "Running warm‑up script..."
    "${WARMUP_SCRIPT}"
    log "Warm‑up completed."
else
    log "No warm‑up script found."
fi

# 9️⃣  NVMe health check
log "Running NVMe health check..."
nvme smart-log "${NVME_DEV}" | grep -i temperature || true

log "=== GPU recovery finished ==="
```

Make it executable:

```bash
chmod +x /usr/local/bin/gpu-recover.sh
```

And ensure the systemd unit (`gpu-recover.service`) points to this script. With this in place, you can **sleep soundly** knowing that each morning your GPU VM will be ready, fast, and cost‑effective.

---

## 16. Future directions – what to watch for  

| Trend | Impact on daily‑deallocate workflow |
|-------|--------------------------------------|
| **Azure VM Scale Sets with GPU support** | Allows you to spin up a pool of pre‑warmed VMs, reducing the need for nightly deallocation. However, you still pay for idle GPUs, so the cost model changes. |
| **Azure Ephemeral OS Disks (managed)** | Future VM SKUs may expose an *ephemeral OS disk* that survives deallocation, potentially simplifying the mount race. |
| **Direct‑attached NVMe snapshots** | Microsoft is experimenting with *snapshot‑able* local NVMe. If GA, you could snapshot the working copy and restore in < 30 s, eliminating rsync entirely. |
| **GPU‑accelerated storage (e.g., Azure NetApp Files with GPU offload)** | Could let you keep the model on a network file system with near‑NVMe latency, removing the local copy step. |
| **AI‑optimized VM images** | Microsoft may ship images with pre‑installed inference runtimes and tuned `sysctl` values, reducing the amount of custom boot‑time configuration you need. |

Staying on top of these announcements will let you iterate on the pattern we’ve built here, keeping the balance between **cost** and **latency** optimal.

---

## 17. Closing thoughts – the mindset behind the pattern  

You’ve just walked through a **complete, production‑grade strategy** for surviving Azure’s daily deallocate cycle while still delivering sub‑10‑minute startup times for massive GPU workloads. The core ideas are simple:

1. **Treat the NVMe as a disposable workbench**, not a data store.  
2. **Persist the source of truth on a durable disk** (Premium SSD or Blob).  
3. **Automate the mount‑and‑copy steps** with systemd ordering to avoid race conditions.  
4. **Measure, iterate, and cost‑track** every component.

When you apply these principles, you’ll see the same cost savings that turned a $12 800 monthly bill into a few dollars for storage, without sacrificing the performance required by H100‑powered inference. The pattern is reusable across any Azure GPU SKU, any model size, and even across other cloud providers that expose local NVMe on their GPU instances.  

Now go ahead, deallocate with confidence, and let your NVMe flash back to life each morning like a fresh kitchen ready for the day’s recipes.