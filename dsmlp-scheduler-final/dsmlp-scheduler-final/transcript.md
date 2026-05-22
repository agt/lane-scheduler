# DSMLP Priority Scheduler — Design & Development Chat Transcript

---

## Scheduling Policy Design

**User:** I operate a computer cluster which is used by students at a university. During peak periods, all resources are in use. Classes assigned to the cluster vary in size from a dozen students to several hundred, and at varying levels from introductory undergraduate, to upper division undergraduate, to graduate. Activity is 90% interactive and 10% batch or background processing, but the higher level classes tend to have a greater percentage of batch or background, and typically demand more resources for a job and for a longer time.

I am trying to design a Q priority metric to fairly allocate resources as they become available, one which does not cause the higher population low level classes to drown out smaller upper level classes.

What are some approaches to this problem?

**Claude:** This is a classic cluster scheduling fairness problem. Here are the main approaches:

**1. Weighted Fair Queueing (WFQ) / Fair Share Scheduling** — Assign each class a share weight rather than a raw priority. Resources are distributed proportionally to weights, not queue depth. Weight by class level (grad > upper-div > intro) and inversely by enrollment. A 10-student grad seminar and a 200-student intro course can be given equal share of resources.

**2. Hierarchical Queue with Per-Class Normalization** — Build a two-level queue. Each class competes at the class level, not the job level. This naturally prevents large intro classes from starving others.

**3. Deficit Round Robin (DRR)** — Each class accumulates a deficit counter — how much resource it's owed versus what it's received. Always pick the class with the largest deficit. Simple, self-correcting, works well when job sizes vary widely.

**4. Urgency + Decay Priority Score:**
```
P = W_class × (1 + age_factor × wait_time) / current_class_utilization
```

**5. Dominant Resource Fairness (DRF)** — For heterogeneous resources, fairness across multiple resource dimensions. A user's share is defined by whichever resource they're consuming most of relative to cluster capacity.

**Recommendations:** Hierarchical queues as backbone, tier weights inversely scaled by enrollment (`weight ∝ 1/√enrollment`), age-based boosting, separate interactive vs. batch lanes.

---

**User:** before you sketch out a scoring formula, i should add that resources are indeed heterogeneous — most jobs are CPU-only, while ~35% compete for GPUs. Interactive GPU jobs generally use a pool of small/medium devices, with the larger units for batch.

**Claude:** That's an important constraint — it changes the fairness geometry significantly.

GPU vs CPU are effectively separate sub-problems requiring separate fair-share accounting per resource lane. The interactive/batch GPU split maps naturally to two GPU pools with different scheduling policies. DRF becomes the right theoretical framework. The formula probably needs to be a vector score rather than a scalar — one dimension per resource lane.

**User:** let's go with the middle path and proceed with scoring algorithm

*(Middle path = class gets a share proportional to tier weight, distributed round-robin among waiting students within that class)*

**Claude:** [Presented full scoring framework with three lanes (CPU, GPU_INTERACTIVE, GPU_BATCH), formula `P(j,l) = W(c) × Age(j) / U(c,l)`, deficit round-robin within classes, and dispatch loop.]

---

**User:** yes please *(implement in Python)*

**Claude:** [Implemented `scheduler.py` with `Scheduler`, `PriorityScorer`, `UtilizationTracker`, `DeficitTracker`; `simulate.py` simulation harness; `test_scheduler.py` unit tests. All stdlib, no external dependencies.]

---

## Kubernetes Integration

**User:** Let's handle integration first. The cluster runs Kubernetes, and student jobs are submitted as ordinary Pods.

**Claude:** [Asked three questions about integration pattern, pod identity, and preemption.]

**User:** Integration pattern: nodes will be Tainted to inhibit automatic scheduling. The controller should run watch on `list_pod_for_all_namespaces` and `list_nodes`. When a job is selected, patch with a Toleration matching the inhibitory taint. Scheduling decisions based on Pod's `resources.requests`, Pod namespace (= student), and `dsmlp/course` label. Assume interactive unless labelled `dsmlp/batch=True`. Hold off on preemption.

**Claude:** [Asked about course label mapping to CourseClass.]

**User:** `dsmlp/course` maps to an opaque string like `CSE234_SP26_A00`. A CSV file maps each to level and seat count, imported daily from registrar. If pod lists unknown course, infer from numeric code: <100 lower division, 100-199 upper division, ≥200 graduate; assume 200 seats for lower/upper, 50 for graduate.

**Claude:** [Implemented `course_registry.py`, `pod_translator.py`, `node_capacity.py`, `controller.py`, `manifests.yaml`, `test_integration.py`.]

---

## GPU Class Lanes

**User:** Our GPU inventory is actually broken down into GPU classes; pods and jobs bear labels `gpu-class={xsmall, small, medium, large, xlarge}`. Nodes are tainted by GPU class. An existing mutating admission controller automatically adds nodeSelector+toleration to pods. Please assess feasibility of using these GPU class identifiers as lanes, rather than "batch" and "interactive".

**Claude:** Feasible and preferable. GPU-class lanes give better resource accounting, the admission controller already handles placement, and it simplifies the label contract. Batch/interactive becomes a scoring modifier (`Mode(j) = 0.3` for batch) rather than a structural lane split. Lane count goes from 3 to 6.

**User:** yes please

**Claude:** [Updated `scheduler.py` (new Lane enum: CPU + GPU_XSMALL through GPU_XLARGE), `pod_translator.py` (reads `gpu-class` label), `node_capacity.py` (reads `gpu-class` taint), all tests. Added `batch: bool` to `Job` and `Mode(j)` multiplier to scoring formula.]

---

## Dynamic Lane Discovery

**User:** Can the controller determine the list of gpu classes at startup, based on node `gpu-class` label values?

**Claude:** Yes. Two options: (A) keep static enum, validate at startup; (B) build Lane dynamically from discovered gpu-class values. Option B preferable since node management is handled by a separate team.

Design: `build_lanes(gpu_classes)` constructs `IntEnum` functionally with CPU=0 and GPU classes sorted lexicographically (stable under additions). `initialise_lanes()` sets module-level globals. All other modules reference `Lane` lazily inside function bodies to avoid import-time capture of `None`.

Controller restart required for new GPU classes — documented clearly.

**User:** yes

**Claude:** [Rewrote `scheduler.py` with dynamic Lane enum via `build_lanes()` and `initialise_lanes()`; added `lane_for_gpu_class()` lookup function; updated `pod_translator.py` and `node_capacity.py` to use lazy imports; added `discover_gpu_classes()` to `controller.py`; updated all tests with `setUpModule()` calling `initialise_lanes()`.]

---

## Wait Time Estimation

**User:** Each pod is configured with an `activeDeadlineSeconds` maximum lifetime, and we can assume actual residency time is distributed normally. If the controller is provided cluster-wide mean residency (as percentage of maximum) and standard deviation values for both interactive and batch workloads, can it estimate wait time for a given pod in queue?

**Claude:** Yes, with caveats. For a running pod with `activeDeadlineSeconds = D` and residency `X ~ Normal(μ, σ)`:

```
P(finish within t | survived to age) = [Φ((age+t - μ)/σ) - Φ((age - μ)/σ)] / [1 - Φ((age - μ)/σ)]
```

Summing across running pods gives `E[slots freed within t]`. Bisect to find `t` where expected slots equals queue rank. P20/P80 from variance of slot-count estimate. All stdlib via `math.erf`.

Proposed interface: `ResidencyProfile`, `RunningPod`, `WaitEstimate`, `estimate_wait()`.

**User:** yes

**Claude:** [Implemented `wait_estimator.py` with truncated-normal math, bisection search, `WaitEstimate` dataclass, `format_wait()` formatter; added `queue_rank()` to `Scheduler`; added running pod tracking and `estimate_wait_for_pod()` to controller; added env vars for residency profiles; wrote `test_wait_estimator.py`.]

---

**User:** assuming 600 running pods, 300 cpu-only and the remainder evenly spread across gpu classes, how computationally expensive is the wait time calculation? would it be more efficient to calculate a "snapshot" of wait time for all pods at some cadence?

**Claude:** Benchmarked directly:
- Single call, rank 25-100, 300 running pods: ~57 ms
- Full snapshot (200 queued pods): ~6.8 seconds

The bottleneck is 48 bisection iterations × 3 passes × O(N) pod loop per iteration. A full snapshot at 1-minute cadence would consume ~7 seconds of CPU per cycle — unusable. On-demand at 57ms/call is fine for single requests but would block under concurrent load.

Fix options: adaptive bisection bounds and two-pass instead of three. Or: background thread with cached results.

**User:** Yes, let's run in a background thread 1/min and cache results.

**Claude:** [Implemented `WaitTimeCache` in `wait_estimator.py`: background thread, atomic cache replacement, failure-safe (preserves last good snapshot on error), stale entry eviction; added `_scored_candidates()` to `Scheduler` for O(Q log Q) ranking; updated `controller.py` with `_build_wait_snapshot()`, `get_wait_estimate()`, `DSMLP_WAIT_CACHE_INTERVAL` env var; added 10 cache tests.]

---

## Event Publication

**User:** Using data from the wait snapshot, the controller should periodically publish Kubernetes "Event" objects for each pod with queue position and wait time estimates. Generate an event for a new pod at the next snapshot; then once a minute for the first 5 minutes, then every five minutes thereafter.

**Claude:** [Implemented `event_publisher.py` with `PodEventSchedule` (per-pod emission state), `EventPublisher` (register/deregister/publish_due), `class_rank()` helper, `_format_message()`. Events use unique names per emission to prevent K8s deduplication collapsing them. Added `events: create` to ClusterRole RBAC. Wired into controller: `register()` on enqueue, `deregister()` on dequeue, `publish_due()` called from `_build_wait_snapshot()`. Wrote `test_event_publisher.py` with 25 tests.]

---

**User:** for designated batch jobs, emit one event immediately, then at 25%, 50%, 75% of expected wait.

**Claude:** Key design decision: milestone times anchored to `enqueued_at + fraction × first_median_wait`, computed once at emit 0 and never recomputed — so changing queue estimates don't shift the milestones.

Schedule:
- Emit 0: immediate (next snapshot)
- Emit 1: `enqueued_at + 0.25 × anchor`
- Emit 2: `enqueued_at + 0.50 × anchor`
- Emit 3: `enqueued_at + 0.75 × anchor`
- Exhausted: `next_emit_at = float("inf")`

Guard clause: if milestone already past when computed, fire immediately.

[Updated `PodEventSchedule` with `batch` flag, `_anchor_wait` field, updated `advance(now, median_wait)` method, `exhausted` property; updated `register()` to detect batch flag via `_is_batch(pod)`; updated `publish_due()` to skip exhausted schedules and pass `median_wait`. Added `TestBatchSchedule` and `TestBatchPublishDue` test classes.]

---

## Final File Inventory

| File | Purpose |
|------|---------|
| `scheduler.py` | Core scheduler: dynamic Lane enum, Scheduler, PriorityScorer, UtilizationTracker, DeficitTracker |
| `course_registry.py` | CSV-backed course registry with tier inference fallback |
| `pod_translator.py` | K8s pod dict → Job translation; admission patch |
| `node_capacity.py` | Live node capacity tracking per lane |
| `controller.py` | Main controller: pod/node watches, scheduling cycle, CSV reload, wait cache, event publication |
| `wait_estimator.py` | Truncated-normal wait estimation; WaitTimeCache background thread |
| `event_publisher.py` | Kubernetes Event publication with per-pod emission schedules |
| `simulate.py` | Synthetic workload simulation harness |
| `manifests.yaml` | RBAC, Deployment, ConfigMap, CronJob |
| `test_scheduler.py` | Unit tests for core scheduler (25 tests) |
| `test_integration.py` | Unit tests for K8s integration layer (38 tests) |
| `test_wait_estimator.py` | Unit tests for wait estimator and cache (37 tests) |
| `test_event_publisher.py` | Unit tests for event publisher (37 tests) |

**Total: 137 tests, all passing. Zero external dependencies (besides `kubernetes` client in controller.py).**

---

*Transcript prepared May 2026*
