# Debugging by Hypothesis Pruning: A Method for Strange System Behavior

## Why Debugging Feels Like a Mystery

You’ve probably spent a night staring at a log file, convinced that the culprit is a single line you spotted hours ago. The feeling is familiar: you form a story, then you hunt for evidence that makes the story fit. In cognitive psychology this is called **confirmation bias**, and in engineering it’s the most common way to waste time.

Imagine a detective who, before stepping onto the crime scene, decides the murderer is the butler. Every fingerprint, every alibi, every stray shoe print is interpreted as “proof the butler was there.” The real killer could be hiding in plain sight, but the detective never looks beyond the butler because the hypothesis has already been sealed.

In software, the “butler” is often the first thing that jumps to mind—a recent code change, a particular library version, a network timeout. You then comb the repository, the metrics, the alerts, looking for the missing piece that confirms your suspicion. If the evidence doesn’t line up, you either ignore it or rationalize it away. The result is a loop that can run for hours, days, or even weeks, while the real cause sits idle.

The antidote is not “more detective work” but **hypothesis pruning**: deliberately constructing a set of mutually exclusive possibilities and then designing tests that *eliminate* at least one hypothesis no matter what the outcome. It turns debugging into a binary search through the space of explanations, guaranteeing progress with each experiment.

---

## The Anatomy of Confirmation‑Bias Debugging

| Symptom | Typical Mindset | Result |
|---------|----------------|--------|
| “It must be the new engine” | “I saw a commit that added the engine, so it’s the cause.” | You only look at engine logs, ignore other subsystems. |
| “The network is too slow” | “Latency spikes line up with a firewall rule change.” | You replay the same traffic, never test the storage path. |
| “The model is cached” | “I remember enabling caching last week.” | You assume the cache is warm, never verify disk reads. |

### Analogy 1: The Leaky Pipe

Think of a house with a mysterious water stain on the ceiling. If you assume the leak is coming from the bathroom pipe because you recently replaced the faucet, you’ll start unscrewing the faucet, tightening clamps, and checking for drips there. The real leak might be a cracked roof tile, but you’ll never look upward because your hypothesis has already locked the investigation to the bathroom.

In code, the “pipe” is the data flow, and the “stain” is the symptom (high latency, error code, crash). Confirmation bias narrows the inspection to the most convenient pipe, while the actual leak could be in a completely different system.

### Analogy 2: The Kitchen Recipe

A chef tastes a dish that’s too salty and immediately blames the salt shaker. He reaches for the shaker, scoops out a pinch, and declares the problem solved—without checking whether the dish was actually seasoned earlier in the process. The real issue might be a mis‑measured ingredient added later, but the chef never revisits that step because his hypothesis is already fixed.

In debugging, the “salt shaker” is the most recent change; the “mis‑measured ingredient” is a deeper, perhaps older, configuration or hardware limitation. Confirmation bias keeps you from revisiting the earlier steps.

---

## Introducing Hypothesis Pruning

### What It Means to Prune

Pruning, in the botanical sense, means cutting away branches that are dead or unnecessary so the tree can focus its energy on healthy growth. In debugging, you **prune** hypotheses—each branch represents a possible explanation for the observed behavior. By designing a test that *must* falsify at least one branch, you guarantee that the tree (your mental model) becomes leaner after each experiment.

The key properties of a good pruning test are:

1. **Deterministic elimination** – regardless of pass/fail, at least one hypothesis is ruled out.
2. **Balanced split** – the test should ideally discard roughly half of the remaining hypotheses, mirroring a binary search.
3. **Low cost** – the test should be quick to run and safe to execute in production or staging.

### Step‑by‑Step Process

| Step | Action | Why |
|------|--------|-----|
| 1 | **Write down every plausible explanation** (no matter how far‑fetched). | Externalizing thoughts prevents you from silently discarding ideas. |
| 2 | **Group hypotheses by shared observable** (e.g., “requires network”, “touches SSD”). | Grouping reveals natural test dimensions. |
| 3 | **Design a test that isolates one dimension**. | The test’s outcome will split the hypothesis space. |
| 4 | **Run the test, record the result, prune**. | You now have a smaller, more accurate set of possibilities. |
| 5 | **Iterate until a single hypothesis remains**. | The remaining hypothesis is the most likely root cause. |

---

## Worked Example: A 120B Model Returns in 0.93 s

You are running an LLM inference service locally on a workstation equipped with a 2 TB NVMe SSD. The model size is 120 billion parameters (≈ 240 GB on disk). A cold load from the SSD should take at least 3–5 seconds to read the weights into RAM, yet your service reports **0.93 seconds** for the first request after a fresh reboot. Something is *too* fast.

### 1. Enumerate the Hypotheses

| # | Hypothesis | Observable Signature |
|---|------------|----------------------|
| A | **New lightweight engine** (e.g., a quantized version of the model) is being used instead of the 120B checkpoint. | Model size reported by the engine ≈ 5 GB. |
| B | **Cloud‑served inference** – the request is being proxied to a remote API. | Network traffic to external IP, latency ≈ 10 ms. |
| C | **CPU‑only inference** – the engine fell back to a CPU path that loads a tiny stub model. | CPU usage spikes, memory footprint ≈ 2 GB. |
| D | **Wrapper routing bug** – the request is hitting a cached response or a different endpoint that bypasses the model load. | Logs show “served from cache”, or the request never reaches the model binary. |
| E | **SSD cache / OS prefetch** – the OS happened to have the model pages in memory from a previous run. | `free -h` shows the model’s memory already allocated before the first request. |

You have five plausible explanations. The next step is to design tests that eliminate at least one of them regardless of outcome.

### 2. Test A – Verify the Engine’s Model Size

**Command**

```bash
# Assuming the engine exposes a /info endpoint that returns JSON
curl -s http://localhost:11434/api/info | jq '.model_size_gb'
```

**Interpretation**

| Output | Eliminated Hypotheses |
|--------|-----------------------|
| `240` | A (lightweight engine) is false. |
| `5`   | B, C, D, E remain; A is confirmed (but we still need to verify it’s not a bug). |

Running the command on the problematic system returns `240`. Hypothesis A is pruned.

### 3. Test B – Confirm Whether Traffic Leaves the Host

**Command**

```bash
# Capture outbound connections for 5 seconds
sudo timeout 5 tcpdump -i any -nn -q not port 22 and not port 53
```

**Interpretation**

| Observation | Eliminated Hypotheses |
|-------------|-----------------------|
| No external IPs appear | B is false. |
| Packets to `34.212.45.12:443` (OpenAI) appear | B is true; the request is being proxied. |

The capture shows **no outbound traffic** beyond local ports. Hypothesis B is eliminated.

### 4. Test C – Measure CPU Utilization During Inference

**Command**

```bash
# Run a single inference while monitoring CPU
watch -n 0.5 "ps -p $(pgrep -f ollama) -o %cpu,cmd"
```

**Interpretation**

| CPU pattern | Eliminated Hypotheses |
|-------------|-----------------------|
| CPU spikes to 95 % on a single core | C is likely true (CPU‑only fallback). |
| CPU stays < 5 % and GPU (if present) shows activity | C is false. |

During the 0.93 s request, the process shows **< 2 % CPU** and **no GPU** activity (the machine has no GPU). This does not confirm C; low CPU could also mean the model was already in RAM. However, because the process never reaches high CPU, we cannot yet eliminate C. We keep it for now.

### 5. Test D – Inspect Wrapper Logs for Cache Hits

**Configuration snippet**

```yaml
# wrapper.yaml
log_level: debug
cache:
  enabled: true
  ttl_seconds: 300
```

**Command**

```bash
# Tail the wrapper log while issuing a request
tail -f /var/log/ollama-wrapper.log &
REQ_PID=$!
curl -s http://localhost:11434/api/generate -d '{"model":"120B","prompt":"Hello"}'
kill $REQ_PID
```

**Interpretation**

| Log line | Eliminated Hypotheses |
|----------|-----------------------|
| `Cache hit for request id xyz` | D is true (the request never hit the model). |
| `Forwarding request to engine` | D is false. |

The log shows **`Forwarding request to engine`** with no cache hit. Hypothesis D is eliminated.

### 6. Test E – Check OS Page Cache Before First Request

**Command**

```bash
# Show how many pages of the model file are cached
sudo grep -c "120B" /proc/meminfo   # placeholder; actual check uses /proc/vmstat
# Better: use vmtouch
vmtouch -c /path/to/120B/model.bin
```

**Interpretation**

| Output | Eliminated Hypotheses |
|--------|-----------------------|
| `0` cached pages | E is false (cold load). |
| > 0 cached pages | E is true (OS already cached). |

Running `vmtouch -c` after a fresh reboot returns **`0`** cached pages. The OS cache hypothesis is false.

### 7. What Remains?

After the five tests we have pruned **A, B, D, E**. The only surviving hypothesis is **C – the engine fell back to a CPU‑only path that loads a tiny stub model**. To confirm, we inspect the binary that the wrapper ultimately invoked:

```bash
# Find the binary that handled the request
ps -ef | grep ollama
# Suppose it shows: /usr/local/bin/ollama-cpu
```

Indeed, the process name ends with `-cpu`. The configuration file reveals a fallback rule:

```yaml
# engine.yaml
fallback:
  enabled: true
  target: cpu
```

The fallback was triggered because the GPU driver was missing, causing the engine to load a **2 GB quantized stub** that can answer the prompt in under a second. The original 120 B model was never touched.

#### Summary of Pruning

| Test | Hypotheses Eliminated |
|------|-----------------------|
| Engine size check | A |
| Network capture | B |
| Wrapper log | D |
| OS cache check | E |
| CPU monitor (inconclusive) | – |
| Final binary inspection | C confirmed |

You spent **≈ 12 minutes** on four decisive tests, eliminating 80 % of the hypothesis space each time. The remaining hypothesis was verified with a single, cheap inspection. Contrast this with a confirmation‑bias approach where you might have spent hours chasing the “new engine” story, rewriting code, and still never found the fallback rule.

---

## The Structural Rule: Split the Space Like a Binary Search

When you have *N* hypotheses, a test that halves the space reduces the expected number of remaining hypotheses to **N / 2**. After *k* such balanced tests, the expected size is **N / 2ᵏ**. This is the same mathematics behind binary search in a sorted array.

| Initial hypotheses (N) | Tests needed (balanced) to reach 1 |
|------------------------|-----------------------------------|
| 8 | 3 (8 → 4 → 2 → 1) |
| 16 | 4 |
| 32 | 5 |
| 64 | 6 |

If your tests are unbalanced—say a test eliminates only one hypothesis out of eight—you waste exponential time. The pruning method forces you to ask, *“If this test passes, which hypotheses go away? If it fails, which go away?”* If the answer to either side is a single hypothesis, you need to redesign the test.

### Designing a Balanced Test

1. **Identify a binary attribute** that separates the hypothesis set roughly in half.  
   *Example*: “Does the request travel over the network?” separates remote‑served vs. local‑served hypotheses.

2. **Choose a cheap observable** that directly reflects that attribute.  
   *Example*: Packet capture (`tcpdump`) is cheap and definitive for network traffic.

3. **Validate the split** before running the test. Write a quick table:

| Hypothesis | Network? |
|------------|----------|
| A (local engine) | No |
| B (cloud) | Yes |
| C (CPU stub) | No |
| D (wrapper cache) | No |
| E (OS cache) | No |

Only B is “Yes,” so the split is heavily unbalanced. In this case we refine the attribute: “Is the request *proxied* through an HTTP client library?” That yields a 2‑vs‑3 split, which is acceptable.

---

## Meta‑Lesson: Write It Down Before You Run Anything

A verbal hypothesis is easy to reshape after the fact. By the time you finish a test, you might unconsciously reinterpret the result to fit the story you wanted. The **commit‑like habit** of writing hypotheses in a markdown file, a ticket, or even a sticky note forces you to:

- **Commit to a set**: You can’t later claim you never considered a hypothesis you didn’t write.
- **Track pruning**: You can tick off eliminated items, making progress visible.
- **Prevent retro‑fitting**: When you look back, the list shows exactly what you believed *before* the test.

#### Template for a Debugging Session

```markdown
# Debugging Session: <short description>

## Observed Symptom
- <timestamp> <metric> <value>

## Hypotheses
| ID | Description | Expected Observable |
|----|-------------|----------------------|
| H1 | ... | ... |
| H2 | ... | ... |
| …  | … | … |

## Test Plan
| Test | Target Attribute | Expected Outcome | Eliminates |
|------|------------------|------------------|------------|
| T1   | …                | …                | H1, H3     |
| T2   | …                | …                | H2, H4     |
```

Copy‑paste this template at the start of every incident. You’ll notice a dramatic reduction in “I thought it was X, but the logs said Y” moments.

---

## When the AI Assistant Helps – and When It Hurts

Large language models excel at **hypothesis generation**. Give them a symptom and they can spit out dozens of plausible explanations in seconds. That’s a massive productivity boost for step 1 of the pruning workflow.

### The Strength: Speed and Breadth

```bash
# Prompt to an LLM (pseudo‑code)
curl -X POST https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
        "model": "gpt-4o-mini",
        "messages": [{"role":"user","content":"Inference latency 0.9 s for 120B model on cold SSD. List possible causes."}]
      }' | jq '.choices[0].message.content'
```

The model returns a list identical to our earlier table, plus a few exotic ideas (e.g., “kernel page‑fault throttling”). You now have a richer hypothesis set without spending mental bandwidth.

### The Failure Mode: Over‑confidence in a Single Suggestion

If you treat the AI’s first suggestion as gospel, you fall back into confirmation bias. The model may hallucinate a plausible‑looking cause that doesn’t exist in your environment (e.g., “a mis‑configured SELinux policy”). You then spend time chasing a ghost.

**Mitigation**: Treat the AI output as a *source* for the hypothesis list, not a *decision engine*. Write the list yourself, perhaps after a quick review of the AI’s suggestions, then proceed with the pruning methodology.

### Practical Workflow with AI

1. **Prompt**: “Give me a numbered list of all things that could make a cold‑load inference faster than 1 s on a 2 TB NVMe.”
2. **Copy** the numbered list into your debugging markdown template.
3. **Add** any domain‑specific items you know (e.g., “environment variable OLLAMA_REMOTES”).  
4. **Prioritize** by grouping into binary attributes (network, cache, engine type).
5. **Design** tests as described earlier.

By separating *generation* (AI) from *validation* (your tests), you get the best of both worlds.

---

## Advanced Pruning Techniques for Senior Engineers

### 1. Parallel Pruning

When you have access to multiple isolated test environments (e.g., a Kubernetes dev cluster and a local VM), you can run **different tests in parallel** on different hypotheses. This reduces wall‑clock time from *O(log N)* to *O(log N / P)* where *P* is the number of parallel workers.

**Example**:  
- Worker 1 runs the network capture test (eliminates B).  
- Worker 2 runs the engine size query (eliminates A).  
- Worker 3 runs the OS cache check (eliminates E).  

All finish within 30 seconds, leaving only C and D to be examined sequentially.

### 2. Probabilistic Pruning

Sometimes you cannot guarantee a deterministic split, but you can assign **prior probabilities** based on historical data. Use Bayesian updating after each test:

```
Posterior(H_i) ∝ Likelihood(TestResult | H_i) × Prior(H_i)
```

If a test is more likely under hypothesis H₁ than H₂, the posterior will shift accordingly. This is useful when tests are noisy (e.g., intermittent network latency).

**Python snippet**:

```python
import numpy as np

hypotheses = ['A','B','C','D','E']
priors = np.array([0.2,0.2,0.2,0.2,0.2])

def update(priors, likelihood):
    post = priors * likelihood
    return post / post.sum()

# Likelihood matrix for a test that returns "network traffic observed"
likelihood = np.array([0.1, 0.9, 0.2, 0.1, 0.1])
post = update(priors, likelihood)
print(dict(zip(hypotheses, post)))
```

Running this after observing network traffic would dramatically boost the probability of hypothesis B, prompting you to prioritize confirming or rejecting it next.

### 3. Instrumentation Hooks as Test Oracles

Instead of ad‑hoc commands, embed **instrumentation hooks** in your code that emit a structured event when a particular path is taken. For example, in a Go inference server:

```go
if cfg.FallbackEnabled {
    telemetry.Emit("fallback_used", map[string]string{
        "mode": "cpu",
    })
}
```

You can then query the telemetry store:

```bash
curl -s http://telemetry.local/api/events?type=fallback_used | jq '.count'
```

If the count is non‑zero, hypothesis C is confirmed instantly. Instrumentation turns a *test* into a *passive observation*, eliminating the need for disruptive probes.

---

## A Checklist for Every Strange‑Behavior Incident

| ✅ | Action |
|----|--------|
| **1** | Write down *all* plausible hypotheses in a markdown table. |
| **2** | Group hypotheses by binary attributes (network, cache, engine type, hardware). |
| **3** | For each group, design a test that will eliminate at least one hypothesis regardless of pass/fail. |
| **4** | Verify the test is cheap and safe (no data loss, minimal impact). |
| **5** | Run tests **in parallel** when possible. |
| **6** | Record outcomes, tick off eliminated hypotheses. |
| **7** | If more than one hypothesis remains, repeat from step 2. |
| **8** | When a single hypothesis remains, perform a *confirmatory* deep dive (e.g., code review, config diff). |
| **9** | Document the root cause, the pruning path, and any preventive actions. |
| **10** | Reflect on whether any hypothesis was missed; add to future templates. |

Having this checklist at hand (e.g., pinned in your IDE or in a Confluence page) turns the abstract pruning method into a repeatable habit.

---

## Bringing It All Together

You now have a concrete mental model for turning a vague, “something is wrong” feeling into a systematic, binary‑search‑style investigation. The process forces you to:

- **Resist the allure of the first suspect** (the detective’s bias).  
- **Enumerate the full hypothesis space** (the garden of possible explanations).  
- **Design elimination tests** that are cheap, safe, and balanced (the plumbing test that tells you which pipe is leaking without tearing down the whole wall).  
- **Iterate until a single, well‑supported cause remains** (the final inspection that reveals the hidden pipe behind the wall).  

When you add an LLM into the mix, you get a rapid hypothesis generator, but you still keep the *pruning* step firmly under your control. The AI suggests possibilities; you validate them with disciplined experiments.

The next time you encounter a baffling latency spike, a mysterious crash, or a performance number that defies physics, remember: **the fastest way to the truth is not to chase a single story, but to cut away the branches that don’t belong**. By treating debugging as a series of hypothesis‑pruning steps, you turn every strange system behavior into a tractable, measurable problem—and you keep your engineering time spent on building, not on endless speculation.