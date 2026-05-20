# Storage Tiering for Self‑Hosted LLM Stacks: Ephemeral vs Persistent  

You are about to spin up a self‑hosted large language model (LLM) stack—think `vLLM` + `Qdrant` + `PostgreSQL` + `FastAPI`. The compute side (your H100 GPU) gets most of the spotlight, but the storage layout is the silent plumbing that determines whether your service stays up at 8 pm or silently loses a customer’s vector index. This article walks you through **three storage tiers**, a **decision rule** that separates “re‑creatable” from “stateful” data, the **failure modes** when you cross the line, and the **guard patterns** that keep your stack honest. We’ll end with a **real‑world Azure H100 host layout**, complete with cost numbers, mount scripts, and systemd units.  

---

## 1. The Three Storage Tiers – From Kitchen Counter to Warehouse  

| Tier | Typical Azure Offering | Latency (read) | Throughput (sequential) | Durability | Cost / GB‑month* |
|------|------------------------|----------------|--------------------------|------------|-------------------|
| **Ephemeral Local NVMe** | Local SSD (NVMe) attached to the VM | ~10 µs | 3–5 GB/s (read) | **Zero** after VM de‑allocate | Included in VM price |
| **Premium / Managed SSD** | Premium P30‑P80 (managed) | ~150 µs | 2 GB/s (read) | 99.9999999 % (11 9’s) | $0.12 – $0.25 |
| **Object Storage** | Azure Blob (Hot / Cool / Archive) | 5–15 ms (Hot) | 500 MB/s (Hot) | 99.9999999 % (11 9’s) | $0.018 (H) / $0.01 (C) / $0.001 (A) |

\*Prices are Azure 2024 spot rates for the West Europe region; they fluctuate, but the ratios stay roughly the same.

### Analogy #1 – The Kitchen Counter, the Pantry, and the Warehouse  

* **Ephemeral NVMe = Kitchen Counter** – You place the ingredients you are actively chopping on the counter. It’s fast, but if you leave the kitchen (de‑allocate the VM) the counter is wiped clean.  
* **Premium SSD = Pantry** – The pantry stores cans, jars, and spices you’ll need repeatedly. It’s slower to reach than the counter, but the food stays fresh for months.  
* **Object Storage = Warehouse** – The warehouse holds bulk shipments—think pallets of rice or flour. You pull a box only when you need to restock the pantry or the counter. It’s cheap, but the walk from the dock takes time.

When you design an LLM stack, you decide which “ingredient” lives on which surface. The decision rule below makes that choice systematic.

---

## 2. Decision Rule – Re‑creatable vs Stateful Data  

| Data Type | Re‑creatable? | Typical Size | Where it belongs |
|-----------|--------------|--------------|------------------|
| **Model weights** (e.g., `meta-llama/Meta-Llama-3-70B`) | ✅ (downloadable from HF, can be re‑packed) | 30‑100 GB | **Ephemeral NVMe** (working copy) + **Premium SSD** (source archive) |
| **HF cache (`~/.cache/huggingface`)** | ✅ (can be re‑downloaded) | 5‑20 GB | **Ephemeral NVMe** |
| **Container images / OCI layers** | ✅ (pullable from registry) | 1‑5 GB per image | **Ephemeral NVMe** (docker overlay) |
| **Vector store (Qdrant, Milvus)** | ❌ (contains user‑generated embeddings) | 10‑500 GB | **Premium SSD** |
| **Relational DB (PostgreSQL, MySQL)** | ❌ (transactions, user data) | 5‑200 GB | **Premium SSD** |
| **Customer documents / PDFs** | ❌ (source material) | 1‑50 GB | **Premium SSD** (or Blob for archival) |
| **Logs & telemetry** | ✅ (re‑generable) | 0.5‑5 GB/day | **Ephemeral NVMe** (rotate) |
| **Backups / snapshots** | ✅ (copy of stateful data) | 1‑2× source size | **Object Storage** (Hot for recent, Cool/Archive for older) |

**Rule of thumb:**  

> **If you can rebuild the data from an external source (model hub, package registry, public repo) → Ephemeral.**  
> **If the data is the result of your users’ actions or a legal record → Durable (Premium SSD or Object).**

### Analogy #2 – The “Make‑It‑From‑Scratch” vs “Family‑Heirloom” Kitchen  

* **Make‑It‑From‑Scratch** ingredients (flour, eggs) are cheap to restock; you keep them on the counter while you bake. If the counter is cleared, you can fetch more from the pantry.  
* **Family‑Heirloom** items (a grandmother’s secret sauce) cannot be recreated; you store them in a locked pantry drawer. Losing that drawer means the recipe is gone forever.

---

## 3. What Happens When You Mix Tiers – The Daily‑Deallocate Disaster  

Imagine you launch a **Qdrant** service on a VM that auto‑scales down at 8 pm. You mount the **ephemeral NVMe** at `/var/lib/qdrant`. At 8 pm the orchestrator stops the VM, the local SSD is reclaimed, and the next morning the VM boots on a fresh NVMe volume. Qdrant starts, sees an empty `/var/lib/qdrant`, and happily creates a fresh, empty collection. Your API now returns *no results*—the customers’ embeddings vanished without a trace.

### Concrete Failure Walk‑through  

```bash
# 1️⃣ VM boots, mounts /dev/nvme0n1p1 → /var/lib/qdrant
mount /dev/nvme0n1p1 /var/lib/qdrant

# 2️⃣ Qdrant starts (systemd unit)
systemctl start qdrant

# 3️⃣ At 20:00 the auto‑scale policy triggers
az vm deallocate --resource-group rg-llm --name h100-node-01

# 4️⃣ Ephemeral disk is destroyed. The next day:
az vm start --resource-group rg-llm --name h100-node-01
# New NVMe volume is attached (empty)
mount /dev/nvme0n1p1 /var/lib/qdrant
systemctl start qdrant   # No error, just an empty DB
```

**Impact:**  

| Symptom | Root Cause | Cost |
|---------|------------|------|
| Empty search results | Vector store on ephemeral disk | Lost revenue, support tickets |
| Model loading fails silently | Model directory empty | 5 min of downtime per restart |
| Container image pull storm | Images on ephemeral disk, node restarted | Network throttling, higher egress cost |

The lesson is simple: **stateful services must never rely on a storage tier that disappears on de‑allocate**.

---

## 4. Guard Patterns – Failing Loudly, Not Silently  

### 4.1 `mountpoint -q` – The “Is My Counter Ready?” Check  

A service that needs a mount should verify the mount exists **before** it starts. In a systemd unit you can add an `ExecStartPre` line:

```ini
# /etc/systemd/system/qdrant.service
[Unit]
Description=Qdrant vector store
After=network-online.target local-fs.target
Wants=network-online.target

[Service]
Type=simple
ExecStartPre=/usr/bin/mountpoint -q /var/lib/qdrant || (echo "❌ /var/lib/qdrant not mounted!" && exit 1)
ExecStart=/usr/bin/qdrant --config /etc/qdrant/config.yaml
Restart=on-failure
User=qdrant
Group=qdrant

[Install]
WantedBy=multi-user.target
```

If the mount is missing, the service aborts with a clear log line. You can see it instantly with `journalctl -u qdrant`.

### 4.2 `nofail` in `/etc/fstab` – Let the Kitchen Counter Be Optional  

When you mount an **ephemeral** volume, you usually do **not** want a missing disk to block the whole boot sequence. Add the `nofail,x-systemd.device-timeout=30` options:

```fstab
/dev/nvme0n1p1   /var/lib/qdrant   ext4   defaults,nofail,x-systemd.device-timeout=30   0   2
```

Now the kernel will continue booting even if the NVMe device is not present (e.g., during a cold‑start after a scale‑down). The `mountpoint -q` guard will still stop Qdrant from running, giving you a clean failure instead of a partially functional node.

### 4.3 “Source of Truth” on Durable Tier – The Pantry Holds the Master Recipe  

Even for data that lives on the counter (NVMe) you should keep a **canonical copy** on a durable tier. For model weights:

```bash
# 1️⃣ Store the archive on Premium SSD (once)
mkdir -p /mnt/premium/models
aws s3 cp s3://my-llm-bucket/Meta-Llama-3-70B.tar.gz /mnt/premium/models/

# 2️⃣ At boot, rsync the working copy to NVMe
cat <<'EOF' > /etc/systemd/system/model-sync.service
[Unit]
Description=Sync LLM model from Premium SSD to local NVMe
After=local-fs.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/bin/rsync -a --delete /mnt/premium/models/Meta-Llama-3-70B/ /var/lib/models/Meta-Llama-3-70B/
EOF

systemctl enable model-sync.service
```

If the NVMe volume is ever replaced, the sync restores the exact same files in seconds (30 GB over a 2 GB/s link ≈ 15 s). The **premium SSD** acts as the pantry; the **NVMe** is the counter where the chef works.

---

## 5. Worked Layout for an Azure H100 Host  

Below is a concrete, production‑ready configuration for a **Standard\_HBv3** VM (8 × H100, 2 TB local NVMe, 1 TB Premium SSD). We also attach an Azure Blob container for long‑term backups.

### 5.1 Hardware & Pricing Snapshot  

| Resource | Size | Cost / Month (USD) | Purpose |
|----------|------|-------------------|---------|
| **Local NVMe** | 2 TB (attached) | Included in VM price (~$5,400) | Ephemeral working copies (model, cache, logs) |
| **Premium SSD (P30)** | 1 TB | $0.12 × 1024 ≈ $123 | Durable stateful data (vector store, DB, model archive) |
| **Blob Storage (Hot)** | 5 TB | $0.018 × 5120 ≈ $92 | Backups, nightly snapshots |
| **Blob Storage (Cool)** | 10 TB (archival) | $0.01 × 10240 ≈ $102 | Old model versions, compliance data |
| **Total** | — | **≈ $5,717** | — |

> **Note:** Azure’s “included” NVMe cost is amortized into the VM price; you still pay for the VM’s compute, but the storage itself does not have a separate line item.

### 5.2 Directory Map  

| Mount Point | Tier | Contents | Size (Typical) |
|-------------|------|----------|----------------|
| `/var/lib/models` | **Ephemeral NVMe** | Working copy of model files (extracted `.bin`, `.json`) | 30 GB |
| `/mnt/premium/models` | **Premium SSD** | Compressed model archive (`.tar.gz`) | 35 GB |
| `/var/cache/huggingface` | **Ephemeral NVMe** | HF cache (tokenizers, config) | 8 GB |
| `/var/lib/qdrant` | **Premium SSD** | Qdrant vector data (`.db`) | 200 GB |
| `/var/lib/postgresql` | **Premium SSD** | PostgreSQL data directory | 100 GB |
| `/var/log/llm` | **Ephemeral NVMe** | Rotating logs (keep 7 days) | 5 GB |
| `/mnt/blob/backups` | **Object (Hot)** | Daily snapshots of `/var/lib/qdrant` and PostgreSQL dumps | 2 TB (growing) |
| `/mnt/blob/archive` | **Object (Cool)** | Old model versions, compliance exports | 10 TB |

### 5.3 fstab Entries  

```fstab
# /etc/fstab
# Device                Mountpoint               Fstype   Options                                                       Dump Pass
/dev/nvme0n1p1          /var/lib/models          ext4     defaults,nofail,x-systemd.device-timeout=30                 0    2
/dev/sdb1               /mnt/premium/models      ext4     defaults,nofail                                              0    2
# Azure Blob Fuse (Hot) – requires blobfuse2
blobfuse2:/backups      /mnt/blob/backups        fuse     allow_other,default_permission,config_file=/etc/blobfuse2.cfg 0    0
blobfuse2:/archive      /mnt/blob/archive        fuse     allow_other,default_permission,config_file=/etc/blobfuse2.cfg 0    0
```

> **Why `nofail` on the NVMe mount?** If the VM boots before the NVMe device is attached (rare but possible during a cold start), the system still reaches multi‑user.target. The `mountpoint -q` guard in each service will stop them until the device appears.

### 5.4 Systemd Units – Guarded Starts  

#### 5.4.1 Model Sync Service (runs at boot)

```ini
# /etc/systemd/system/model-sync.service
[Unit]
Description=Sync LLM model archive from Premium SSD to local NVMe
After=local-fs.target network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/bin/rsync -a --delete /mnt/premium/models/Meta-Llama-3-70B/ /var/lib/models/Meta-Llama-3-70B/
ExecStartPost=/usr/bin/chown -R llm:llm /var/lib/models/Meta-Llama-3-70B
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
systemctl enable model-sync.service
```

#### 5.4.2 Qdrant Service (requires persistent mount)

```ini
# /etc/systemd/system/qdrant.service
[Unit]
Description=Qdrant vector database
After=network-online.target local-fs.target
Wants=network-online.target

[Service]
Type=simple
User=qdrant
Group=qdrant
ExecStartPre=/usr/bin/mountpoint -q /var/lib/qdrant || (echo "❌ /var/lib/qdrant not mounted!" && exit 1)
ExecStart=/usr/local/bin/qdrant --config /etc/qdrant/config.yaml
Restart=on-failure
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
```

#### 5.4.3 PostgreSQL (standard package) – add a pre‑check

Edit `/etc/systemd/system/postgresql.service.d/10-mountcheck.conf`:

```ini
[Service]
ExecStartPre=/usr/bin/mountpoint -q /var/lib/postgresql || (echo "❌ /var/lib/postgresql missing!" && exit 1)
```

Reload daemon:

```bash
systemctl daemon-reload
systemctl enable postgresql
```

### 5.5 Backup Pipeline – From Premium SSD to Blob (Hot)

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1️⃣ Dump PostgreSQL
pg_dumpall -U postgres | gzip > /tmp/pg_dump_$(date +%F).sql.gz

# 2️⃣ Snapshot Qdrant (assumes qdrantctl snapshot)
qdrantctl snapshot create --output /tmp/qdrant_snapshot_$(date +%F).tar.gz

# 3️⃣ Copy both to Blob (Hot)
az storage blob upload-batch \
  --destination backups \
  --source /tmp \
  --pattern "pg_dump_*.sql.gz,qdrant_snapshot_*.tar.gz" \
  --account-name mystorageaccount \
  --auth-mode login

# 4️⃣ Cleanup local temp files
rm -f /tmp/pg_dump_*.sql.gz /tmp/qdrant_snapshot_*.tar.gz
```

Schedule with `cron` (run at 02:00 UTC):

```cron
0 2 * * * /usr/local/bin/llm_backup.sh >> /var/log/llm/backup.log 2>&1
```

---

## 6. Advanced Topics – Going Beyond the Basics  

### 6.1 Tiered Blob Policies (Hot → Cool → Archive)  

Azure Blob supports **lifecycle management** rules that automatically transition objects based on age. For LLM stacks, a typical policy looks like:

```json
{
  "rules": [
    {
      "name": "move-old-backups",
      "type": "Lifecycle",
      "definition": {
        "filters": { "blobTypes": ["blockBlob"], "prefixMatch": ["backups/"] },
        "actions": {
          "baseBlob": {
            "tierToCool": { "daysAfterModificationGreaterThan": 30 },
            "tierToArchive": { "daysAfterModificationGreaterThan": 180 }
          }
        }
      },
      "enabled": true
    }
  ]
}
```

Upload via Azure CLI:

```bash
az storage account management-policy create \
  --account-name mystorageaccount \
  --policy @policy.json
```

Now your **daily backups** sit in Hot storage for the first month (fast restores), then slide to Cool, and finally Archive after six months, saving up to **80 %** on storage cost.

### 6.2 Using Azure Blob Fuse 2 vs. NFS  

* **Blob Fuse 2** mounts object storage as a POSIX filesystem, ideal for **read‑heavy** workloads (e.g., loading a model archive directly from Blob without copying).  
* **Azure Files (NFS v4.1)** provides true file‑system semantics (locking, rename) and is better for **write‑heavy** workloads like a shared vector store across nodes.  

If you ever need **horizontal scaling** of Qdrant, you can switch the mount point from a local SSD to an NFS share, but keep the **ephemeral copy** locally for hot reads and sync changes back nightly.

### 6.3 Pre‑warming NVMe with `fio`  

When a VM boots, the NVMe block device may be **cold** (no data in the SSD’s DRAM cache). A quick `fio` warm‑up can shave 30 % off the first‑epoch model load time:

```bash
fio --filename=/dev/nvme0n1p1 --rw=read --bs=1M --size=2G --name=warmup
```

Run this in a `systemd` service that executes **after** the model sync:

```ini
[Service]
ExecStartPost=/usr/local/bin/warmup_nvme.sh
```

The script simply runs the `fio` command above and exits.

---

## 7. Checklist – Your Tier‑Aware LLM Stack in One Glance  

| ✅ Item | How to Verify |
|--------|----------------|
| **Ephemeral NVMe mounted at `/var/lib/models` and `/var/cache/huggingface`** | `mount | grep /var/lib/models` |
| **Premium SSD mounted at `/mnt/premium`** | `df -h /mnt/premium` |
| **Blob Fuse mounts healthy** | `ls /mnt/blob/backups` (should list recent backup files) |
| **`mountpoint -q` guards present** | `systemctl cat qdrant.service` (look for `ExecStartPre`) |
| **`nofail` present in `/etc/fstab` for NVMe** | `grep nvme0n1p1 /etc/fstab` |
| **Model sync runs on every boot** | `systemctl status model-sync.service` (should be `active (exited)`) |
| **Backup script runs and logs** | `grep SUCCESS /var/log/llm/backup.log` |
| **Blob lifecycle policy active** | `az storage account management-policy show --account-name mystorageaccount` |
| **Cost sanity check** | `az consumption usage list --start-date 2024-04-01 --end-date 2024-04-30` (look at `meterCategory: Managed Disks` and `Blob Storage`) |

If every row checks out, you have a **robust, tier‑aware storage architecture** that protects user data, keeps model loading fast, and keeps your Azure bill predictable.

---

## 8. Putting It All Together – A Sample Boot Sequence  

```mermaid
flowchart TD
    A[VM Power‑On] --> B[Kernel mounts /etc/fstab (NVMe with nofail)]
    B --> C{NVMe present?}
    C -- Yes --> D[Mount /var/lib/models]
    C -- No --> E[Boot continues, services wait]
    D --> F[systemd starts model-sync.service]
    F --> G[rsync model archive from Premium SSD → NVMe]
    G --> H[systemd starts qdrant.service]
    H --> I[mountpoint -q passes → Qdrant runs]
    H --> J[mountpoint -q fails → service aborts]
    I --> K[Fast inference ready]
    J --> L[Operator alerted via journal]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px
```

The diagram shows the **guarded flow**: the system never silently starts a service with an empty model directory. If the NVMe is missing, the boot still finishes, but the guard stops the inference service, giving you a clear alert instead of a “model not found” HTTP 500.

---

## 9. Takeaway – Why Tiering Is Not a “Nice‑to‑Have”  

* **Reliability:** Stateful data lives on a tier that survives VM de‑allocation. No more “daily‑deallocate disaster.”  
* **Performance:** Hot model files sit on the NVMe counter, giving sub‑millisecond load times for `torch.load`.  
* **Cost Efficiency:** You pay premium only for the data that truly needs durability; bulk archives sit in cheap Blob storage.  
* **Operational Simplicity:** Guard patterns (`mountpoint -q`, `nofail`) make failures **visible** and **deterministic**, which is far easier to troubleshoot than a silent empty directory.  

When you design your next self‑hosted LLM stack, think of your storage layout as a **kitchen workflow**: the counter for the chef’s immediate work, the pantry for ingredients you’ll reuse, and the warehouse for bulk shipments. Keep the pantry stocked, never leave the chef’s counter empty, and you’ll serve your users reliably, quickly, and at a predictable cost.