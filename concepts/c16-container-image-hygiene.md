# Container Image Hygiene: Tags, SHAs, Layer Cache, and the Reproducibility Trap

## Why you need a clean kitchen before you serve a meal

Imagine you run a restaurant that serves a signature dish every night. If the pantry shelves keep moving around, the same label “Chicken‑Marsala” could point to a different batch of chicken each service. One night the chicken is fresh, the next it’s been sitting a week. Your diners will notice the difference, and your reputation will suffer.

Container images are the pantry of modern cloud‑native applications. A **tag** is the label you stick on the shelf, while the **SHA‑256 digest** is the unique barcode on the actual bag of ingredients. When you start pulling images by tag, you’re trusting that the label never moves. In practice, tags are mutable, caches are clever (sometimes too clever), line endings can betray you, and multi‑arch manifests add a layer of indirection that looks like a magic trick. This article walks you through each of those traps, shows you concrete commands, and hands you a checklist you can paste into your CI/CD repo today.

---

## 1. Tags are pointers, not the image itself  

### 1.1 The hotel‑room analogy  

Think of a Docker image as a hotel room. The **room number** (`myimage:1.1.0.0`) stays the same, but the **guest** inside can change from night to night. The room number is easy to remember, but the guest’s luggage (the exact bits on disk) is what determines whether you get a clean bed or a broken TV.

In Docker terms:

| Concept | Hotel metaphor | Docker |
|---------|----------------|--------|
| Tag     | Room number (`101`) | `myimage:1.1.0.0` |
| Digest  | Guest’s luggage barcode (`AB12‑CD34`) | `sha256:3f2c...` |
| Re‑push | New guest checks in, old guest checks out | New image uploaded under same tag |

The **digest** never moves. It is the SHA‑256 hash of the image’s manifest, which in turn references the digests of every layer. If any byte changes, the digest changes.

### 1.2 Seeing the difference on the command line  

```bash
# Pull the tag once
docker pull myregistry.example.com/app:1.1.0.0
docker inspect --format='{{index .RepoDigests 0}}' myregistry.example.com/app:1.1.0.0
#=> myregistry.example.com/app@sha256:3f2c7e9b...

# Re‑push a new build under the same tag (simulated)
docker build -t myregistry.example.com/app:1.1.0.0 .
docker push myregistry.example.com/app:1.1.0.0

# Pull again, compare digests
docker pull myregistry.example.com/app:1.1.0.0
docker inspect --format='{{index .RepoDigests 0}}' myregistry.example.com/app:1.1.0.0
#=> myregistry.example.com/app@sha256:9a5d4c1e...
```

The tag stayed `1.1.0.0`, but the digest changed from `3f2c…` to `9a5d…`. Any node that pulled the tag **before** the second push still runs the old bits; any node that pulls **after** runs the new bits. This is the *mutability* problem.

### 1.3 Production should never trust a mutable label  

When you write a Kubernetes manifest, you might be tempted to do:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: myregistry.example.com/app:1.1.0.0   # <-- mutable!
```

If a teammate later re‑pushes `1.1.0.0` with a security fix, **all** pods will silently start running a different binary. That’s great for emergency patches, but it also means you lose the ability to reason about *what* is running in any given environment.

The safe pattern is to pin the digest:

```yaml
        image: myregistry.example.com/app@sha256:9a5d4c1e...
```

Now the deployment is guaranteed to run the exact layers you tested, regardless of future tag moves.

---

## 2. The reproducibility trap: “Same tag, different bits”

### 2.1 A day‑to‑day CI example  

Suppose your CI pipeline does the following on every commit to `main`:

```yaml
steps:
  - name: Pull base image
    run: docker pull myregistry.example.com/app:stable
  - name: Run integration tests
    run: docker run --rm myregistry.example.com/app:stable pytest
```

On Monday, `stable` points at digest `sha256:a1b2`. On Friday, a teammate pushes a hot‑fix to the same tag, changing the digest to `sha256:c3d4`. The same CI job now runs **different** code, but the logs still say “using `stable`”. If a flaky test fails only on Friday, you’ll waste hours chasing a ghost.

### 2.2 Quantifying the risk  

| Day | Tag pulled | Digest | Size (MiB) | Test failures |
|-----|------------|--------|------------|----------------|
| Mon | `stable`   | `a1b2` | 124        | 0/120          |
| Tue | `stable`   | `a1b2` | 124        | 0/120          |
| Wed | `stable`   | `a1b2` | 124        | 0/120          |
| Thu | `stable`   | `a1b2` | 124        | 0/120          |
| Fri | `stable`   | `c3d4` | 128 (+4)   | 3/120          |

The extra 4 MiB came from a new dependency that introduced a subtle race condition. Because the tag didn’t change, the failure looked like a flaky test, not a code change.

### 2.3 How to break the loop  

* **Pin by digest in CI** – `docker pull myregistry.example.com/app@sha256:a1b2`.  
* **Version tags semantically** – `1.2.3‑rc1`, `1.2.3‑rc2`, etc., and never re‑push the same tag.  
* **Automate digest extraction** – after a successful build, capture the digest with `docker inspect` and feed it into downstream jobs.

---

## 3. Build‑cache trap: “I thought I changed the code, but Docker gave me the old layer”

### 3.1 How Docker’s layer cache works  

Docker builds images layer by layer. Each `RUN`, `COPY`, or `ADD` instruction creates a new layer whose content is hashed. If the hash of the instruction *and* the hash of its inputs (files, environment, previous layer) match a previously built layer, Docker re‑uses the cached layer.

Think of a **sandwich shop**: the chef prepares the bread, spreads mayo, adds lettuce, then the meat. If the chef sees a note that says “same mayo, same lettuce, same bread”, they skip those steps and reuse the previous sandwich base. If the note says “new lettuce”, they only replace that layer, leaving the rest untouched.

### 3.2 A concrete Dockerfile that falls into the trap  

```dockerfile
# Dockerfile.example
FROM python:3.11-slim AS base
WORKDIR /app

# 1️⃣ Install system deps (cached forever)
RUN apt-get update && apt-get install -y gcc libpq-dev

# 2️⃣ Install Python deps (cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3️⃣ Copy source (cached if nothing changed under .)
COPY . .

# 4️⃣ Run tests (executed at build time)
RUN pytest -q
```

If you modify only `app/main.py` but **don’t change** `requirements.txt`, Docker will:

1. Reuse layer 1 (system deps).  
2. Reuse layer 2 (Python deps).  
3. **Re‑run layer 3** because the source tree changed.  
4. **Re‑run layer 4** because the previous layer changed.

That looks fine, but imagine you have a **large monorepo** where `COPY . .` brings in **megabytes** of unrelated code. Docker will still think the layer is unchanged if the *timestamp* of the copied directory is the same, even though a file deep inside changed. The result: the build finishes quickly, but the binary inside the image is still the old one because the cache hit prevented the copy.

### 3.3 Real‑world numbers  

| Build # | Files changed | Layers rebuilt | Build time |
|---------|---------------|----------------|------------|
| 1       | `requirements.txt` | 1,2,3,4 | 2 min 13 s |
| 2       | `app/main.py` | 3,4 | 45 s |
| 3       | No source change (but `git checkout` updates timestamps) | 0 | 12 s (but binary still old!) |

In Build 3, the cache thought nothing changed, yet the underlying source code was stale because the checkout operation didn’t modify file contents, only timestamps. The resulting image still contained the binary from Build 2.

### 3.4 Strategies to avoid the cache trap  

| Strategy | How it works | Example |
|----------|--------------|---------|
| **`--no-cache` for releases** | Disables all caching, guarantees fresh layers. | `docker build --no-cache -t myapp:1.2.3 .` |
| **Separate source layer** | Put `COPY . .` **last**, after all immutable steps, so any source change forces a rebuild of only the final layer. | See revised Dockerfile below. |
| **Use build arguments to bust cache** | `ARG CACHEBUST=$(date +%s)` and `RUN echo $CACHEBUST` forces a fresh layer. | `ARG CACHEBUST=20240520 && RUN echo $CACHEBUST` |
| **Multi‑stage builds with `--target`** | Build a “builder” stage that always runs fresh, then copy the artifact. | `FROM builder AS final` |

#### Revised Dockerfile that isolates source changes

```dockerfile
# Dockerfile.revised
FROM python:3.11-slim AS builder
WORKDIR /src
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pull source *after* deps so source changes only affect this layer
COPY . .
RUN pip install -e .   # editable install for dev, or `python -m build` for prod

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /src /app
CMD ["python", "-m", "myapp"]
```

Now, if only `app/main.py` changes, Docker only rebuilds the **builder** stage’s `COPY . .` layer and the subsequent `RUN pip install -e .`. All earlier layers (system packages, pip cache) stay cached, but you still get a **fresh binary**.

---

## 4. Line‑ending trap: When a Windows checkout breaks a Linux container  

### 4.1 The war story  

Your team uses a Windows laptop for development, and the repo contains a startup script `scripts/startup.sh`. The script is checked out with **CRLF** line endings (`\r\n`). Docker’s `COPY` instruction faithfully copies the file into the image, preserving those Windows line endings.

When the container starts, the kernel reads the shebang:

```
#!/bin/sh\r
```

Because of the stray `\r`, the kernel looks for an executable named `/bin/sh\r`. That file does not exist, so the error you see is:

```
exec /app/startup.sh: no such file or directory
```

The message is technically correct—the *interpreter* `/bin/sh\r` is missing—but it misleads you into thinking the script itself vanished.

### 4.2 Demonstration with real commands  

```bash
# Simulate a repo with a CRLF script
printf '#!/bin/sh\r\necho "Hello"\r\n' > startup.sh
git init -q && git add startup.sh && git commit -qm "add script"

# Build a tiny image that just runs the script
cat > Dockerfile <<'EOF'
FROM alpine:3.18
WORKDIR /app
COPY startup.sh .
CMD ["./startup.sh"]
EOF

docker build -t crlf-demo .
docker run --rm crlf-demo
#=> exec ./startup.sh: no such file or directory
```

Now convert the script to LF and rebuild:

```bash
# Convert line endings
dos2unix startup.sh   # or: git checkout-index --force --all && git add --renormalize .
docker build -t crlf-demo .
docker run --rm crlf-demo
#=> Hello
```

### 4.3 Fixing the problem at the source  

1. **Add a `.gitattributes` file** that forces LF for shell scripts:

```gitattributes
# .gitattributes
*.sh text eol=lf
*.bash text eol=lf
```

2. **Tell Git not to auto‑convert on checkout** for the repo:

```bash
git config core.autocrlf false
git add --renormalize .
git commit -m "Normalize line endings"
```

3. **Enforce the rule in CI** – run `git check-attr --stdin` or a simple `grep -P '\r'` over the build context before `docker build`.

### 4.4 Checklist for line‑ending hygiene  

| Step | Command | When to run |
|------|---------|-------------|
| Add `.gitattributes` | `echo "*.sh text eol=lf" >> .gitattributes` | At repo init |
| Verify no CRLF in build context | `find . -type f -name "*.sh" -exec grep -P '\r' {} + && echo "CRLF found"` | Pre‑build stage |
| Convert existing files | `git add --renormalize . && git commit -m "Normalize line endings"` | After adding `.gitattributes` |
| CI gate | `if find . -type f -name "*.sh" -exec grep -P '\r' {} +; then exit 1; fi` | In CI pipeline |

---

## 5. Multi‑platform manifest gotcha  

### 5.1 What a manifest list is  

When you push a multi‑arch image with `docker buildx`, Docker creates **separate images** for each architecture (e.g., `linux/amd64`, `linux/arm64`) and then writes a **manifest list** (sometimes called a *fat manifest*) that points to each of those images. The manifest list itself has its own digest.

Think of a **train schedule**: the schedule (manifest list) tells you that a train runs from New York to London, but the actual train you board could be an electric locomotive (amd64) or a diesel locomotive (arm64) depending on the platform you’re on. The schedule’s identifier never changes, but the underlying train composition can.

### 5.2 Seeing two digests in the wild  

```bash
# Build and push a multi‑arch image
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 \
  -t myregistry.example.com/webapp:2.0.0 --push .

# Pull on an amd64 host
docker pull myregistry.example.com/webapp:2.0.0
docker inspect --format='{{index .RepoDigests 0}}' myregistry.example.com/webapp:2.0.0
#=> myregistry.example.com/webapp@sha256:aa11bb22...   (platform‑specific digest)

# Inspect the manifest list from another host
docker manifest inspect myregistry.example.com/webapp:2.0.0
#=> {
#     "schemaVersion": 2,
#     "mediaType": "application/vnd.docker.distribution.manifest.list.v2+json",
#     "manifests": [
#       {"digest":"sha256:aa11bb22...", "platform":{"architecture":"amd64","os":"linux"}},
#       {"digest":"sha256:cc33dd44...", "platform":{"architecture":"arm64","os":"linux"}}
#     ]
#   }
```

If you store the **platform‑specific digest** (`aa11bb22…`) in your deployment manifest, an arm64 node will *still* pull the manifest list, resolve the arm64 image (`cc33dd44…`), and run it. The deployment manifest appears correct, but the digest you see locally (`aa11bb22…`) does **not** match the digest another node will resolve.

### 5.3 How this confuses “is the new build live yet?”  

Suppose you push a new `2.0.1` build:

| Platform | New digest |
|----------|------------|
| amd64    | `ff55ee66...` |
| arm64    | `bb77cc88...` |

Your monitoring script runs on an amd64 host and checks:

```bash
docker pull myregistry.example.com/webapp:2.0.1
docker inspect --format='{{index .RepoDigests 0}}' myregistry.example.com/webapp:2.0.1
#=> myregistry.example.com/webapp@sha256:ff55ee66...
```

It reports “latest digest = `ff55ee66`”. Meanwhile, a fleet of arm64 edge devices still report the old digest `bb77cc88`. If you only look at the amd64 side, you’ll think the rollout is complete when half the world is still on the previous binary.

### 5.4 Best practice: pin the **manifest list** digest, not the platform‑specific one  

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp
spec:
  replicas: 4
  template:
    spec:
      containers:
      - name: webapp
        image: myregistry.example.com/webapp@sha256:11223344...   # manifest list digest
```

Kubernetes will resolve the correct platform image at runtime, but the *list* digest guarantees that every node sees the same set of images. When you push a new version, you compute the **manifest list** digest with:

```bash
docker manifest inspect myregistry.example.com/webapp:2.0.1 \
  | jq -r '.config.digest'   # returns the list digest
```

Store that value in your GitOps repo and you have a single source of truth.

---

## 6. The hygiene checklist – a one‑page cheat sheet you can copy‑paste

| ✅ Item | Why it matters | How to enforce |
|--------|----------------|----------------|
| **Semantic version tags** (`X.Y.Z[-RC]`) | Guarantees immutability if you never re‑push the same tag. | CI rejects pushes that attempt to overwrite an existing tag. |
| **Never use `:latest` in production** | `latest` is a moving target; you lose auditability. | Lint Dockerfiles / Helm charts with `hadolint` or `kubeval`. |
| **Pin images by SHA digest in deployment manifests** | Guarantees the exact layers you tested run in prod. | CI step: `DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' $IMAGE)`; substitute into YAML. |
| **Run smoke tests on the built image before push** | Catches missing files, line‑ending bugs, broken entrypoints. | `docker run --rm $IMAGE /app/startup.sh && curl -f http://localhost:8080/health`. |
| **Include `.gitattributes` for all text files** (`*.sh text eol=lf`, `*.Dockerfile text eol=lf`) | Prevents CRLF‑induced exec errors. | Add to repo root; CI validates with `git check-attr`. |
| **Use `--no-cache` for release builds or order Dockerfile to invalidate source early** | Avoids stale layers that hide code changes. | `docker build --no-cache -t $IMAGE .` or place `COPY . .` after immutable steps. |
| **Prune old images, but keep rollback candidates** | Saves storage, but you must still be able to roll back. | `docker image prune -a --filter "until=720h"` + whitelist digests in a `rollback.txt`. |
| **Verify manifest list digest matches deployment** | Prevents cross‑arch drift. | CI step: `MANIFEST=$(docker manifest inspect $IMAGE:$TAG | jq -r '.config.digest')`; compare to repo. |
| **Lock down build arguments** (`ARG CACHEBUST=$(date +%s)`) only when needed | Prevents accidental cache busting that inflates image size. | Document in Dockerfile comments. |

You can embed this table directly into your `README.md` or your internal wiki. The moment you start treating it as a *definition of done* for every image, you’ll see the “reproducibility trap” disappear.

---

## 7. Putting it all together – a sample CI/CD pipeline (GitHub Actions)

Below is a **complete** workflow that demonstrates the concepts we’ve covered. It builds a multi‑arch image, extracts both the platform‑specific and manifest‑list digests, runs a smoke test, and finally commits the digest to a GitOps manifest.

```yaml
# .github/workflows/publish.yml
name: Publish Container

on:
  push:
    branches: [main]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: write   # to push manifest digest back to repo
      packages: write   # to push to GHCR
    steps:
      # 1️⃣ Checkout with normalized line endings
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          # Enforce LF line endings
          lfs: false
      - name: Enforce LF
        run: |
          git config core.autocrlf false
          git add --renormalize .
          git diff --quiet || (git commit -am "Normalize line endings" && exit 1)

      # 2️⃣ Set up Docker Buildx for multi‑arch
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      # 3️⃣ Log in to registry (example: GHCR)
      - name: Log in to GHCR
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      # 4️⃣ Build and push, capture digests
      - name: Build & push
        id: build
        run: |
          IMAGE=ghcr.io/${{ github.repository_owner }}/webapp
          TAG=$(git describe --tags --abbrev=0 || echo "0.0.0")
          # Build multi‑arch and push
          docker buildx build \
            --platform linux/amd64,linux/arm64 \
            -t $IMAGE:$TAG \
            --push \
            --metadata-file /tmp/meta.json \
            .
          # Extract digests
          MANIFEST=$(jq -r '.imageName' /tmp/meta.json)   # e.g. ghcr.io/.../webapp@sha256:xxxx
          echo "manifest=$MANIFEST" >> $GITHUB_OUTPUT
          # Platform‑specific digests (optional)
          for plat in amd64 arm64; do
            DIG=$(docker manifest inspect $IMAGE:$TAG \
                | jq -r ".manifests[] | select(.platform.architecture==\"$plat\") | .digest")
            echo "$plat=$DIG" >> $GITHUB_OUTPUT
          done

      # 5️⃣ Smoke test the freshly pushed image (using the manifest digest)
      - name: Smoke test
        run: |
          IMAGE=${{ steps.build.outputs.manifest }}
          docker pull $IMAGE
          docker run --rm $IMAGE /app/startup.sh || exit 1
          # Simple health‑check
          docker run -d --name test $IMAGE
          sleep 5
          curl -f http://localhost:8080/health || (docker logs test && exit 1)
          docker rm -f test

      # 6️⃣ Update GitOps manifest with the new manifest‑list digest
      - name: Update deployment manifest
        run: |
          FILE=deploy/webapp.yaml
          DIGEST=${{ steps.build.outputs.manifest }}
          # Replace the line that starts with "image:" (assumes single‑line format)
          sed -i -E "s|image: .*|image: $DIGEST|g" $FILE
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add $FILE
          git commit -m "chore: bump webapp to $DIGEST"
          git push
```

**What this workflow demonstrates:**

| Step | Hygiene principle it enforces |
|------|--------------------------------|
| Checkout + LF enforcement | Line‑ending trap avoidance |
| Buildx multi‑arch | Manifest‑list awareness |
| Capture both manifest and per‑arch digests | Ability to audit cross‑platform consistency |
| Smoke test before push | Early detection of broken entrypoints or missing files |
| Pin manifest digest in deployment manifest | Production never pulls a mutable tag |
| `--no-cache` is omitted because we rely on deterministic Dockerfile ordering; you could add `--no-cache` for a “release” branch. |
| GitOps update | Guarantees that the version you rolled out is the exact one you tested. |

You can adapt the same pattern to GitLab CI, Azure Pipelines, or any other orchestrator. The key is **explicitly handling digests** and **verifying the image before it ever reaches a cluster**.

---

## 8. The payoff – reproducible, auditable deployments  

When you treat tags as *labels* and digests as the *real product*, you gain three concrete benefits:

1. **Deterministic rollbacks** – Because every deployment records the exact manifest‑list digest, you can `kubectl set image …@sha256:…` to go back to a known good state without hunting for a tag that might have been overwritten.
2. **Cross‑team confidence** – QA, security, and ops can all run `docker inspect` on the same digest and see identical layer lists, eliminating “it works on my machine” disputes.
3. **Cost savings** – By pruning stale layers **after** you’ve verified that a digest is no longer referenced, you avoid the “dangling‑image” nightmare where a rollback target disappears because the tag was overwritten.

All of these outcomes stem from a single habit: **never assume a tag points to the same bits you built yesterday**. Pin, test, and record the digest, and you’ll keep your container kitchen clean enough to serve a five‑star experience every time.

--- 

### Quick reference cheat sheet (copy‑paste)

```bash
# 1. Build with deterministic ordering
docker build -t myapp:1.2.3 .

# 2. Capture digest immediately after push
docker push myregistry.example.com/app:1.2.3
DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' myregistry.example.com/app:1.2.3)
echo "Digest: $DIGEST"

# 3. Use digest in k8s manifest
cat > k8s/deploy.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: app
        image: $DIGEST
EOF

# 4. Verify no CRLF in build context
if find . -type f -name "*.sh" -exec grep -P '\r' {} +; then
  echo "CRLF detected! Fix before building."
  exit 1
fi

# 5. Prune old images but keep whitelist
WHITELIST=$(cat rollback.txt)
docker image prune -a --filter "until=720h" --filter "label!=keep"
for img in $(docker images -q); do
  if grep -q "$img" <<<"$WHITELIST"; then
    echo "Skipping $img (whitelisted for rollback)"
    continue
  fi
  docker rmi "$img"
done
```

Keep this script in your repo’s `scripts/` folder, give it a descriptive name like `image_hygiene.sh`, and call it from your CI pipeline before any release. You’ll have turned the abstract “container hygiene” concept into a concrete, repeatable process that protects your production workloads from the subtle bugs that have plagued countless teams. 

--- 

**Your next step:** audit the last three releases of a service you own. List the tag, the digest you actually deployed, and the date you pulled each. If any tag appears more than once with different digests, you’ve already found a reproducibility gap. Fix it by updating the deployment manifest to the recorded digest, add the tag to your versioning policy, and lock the pipeline to use `--no-cache` for any release branch.  

From now on, every time you type `docker run myimage:latest` you’ll know exactly why that’s a red flag, and you’ll have a toolbox of commands ready to replace it with a safe, immutable reference. Happy building!