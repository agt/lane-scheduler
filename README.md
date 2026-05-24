# Lane-based Priority Scheduler

A Kubernetes controller that fairly allocates cluster resources across university courses, preventing high-volume courses from crowding out lower-volume ones with higher scheduling priority, while respecting heterogeneous GPU hardware classes.

---

## Goals

University teaching clusters face a structural fairness problem: a high-enrollment course will naturally generate far more queued jobs than a small one, causing lower-volume courses to starve during peak periods even when they warrant equal or higher scheduling priority.

This scheduler addresses that by:

- Assigning each course an operator-defined weight that can incorporate whatever priority factors are relevant — academic level, enrollment, or any other consideration — so high-priority courses compete on equal or better footing with high-volume ones regardless of raw job count
- Distributing a course's share fairly among its individual students, so no single active student within a course monopolizes its allocation
- Preferring interactive jobs over batch jobs within any resource lane, while ensuring batch jobs eventually drain overnight
- Providing students with real-time queue position and estimated wait time via Kubernetes Events visible in `kubectl describe pod`
- Adapting wait-time estimates over time using course-specific completion data, so estimates improve as the term progresses

The scheduler is implemented as a Kubernetes controller. It does not replace the default Kubernetes scheduler; instead it acts as a gating layer, holding pods in a Pending state via node taints until they reach the top of the priority queue, then patching in a toleration that allows the default scheduler to place them normally.

### Scheduling Model

The system operates as a set of independent finite-capacity multi-server queues (lanes), one per GPU hardware class plus one for CPU workloads. Each lane accepts a multi-class open workload of heterogeneous jobs; jobs are held in a pre-admission gate and released in discrete scheduling cycles. The service discipline within each lane is a dynamic non-preemptive priority rule: at each cycle, queued jobs are scored and the top-*K* are admitted subject to a capacity constraint.

The per-job priority score is P(*j*, *l*) = W(*c*) · M(*j*) · A(*j*) / U(*c*, *l*), where W(*c*) is a static operator-assigned class weight, M(*j*) ∈ {0.3, 1.0} is a batch/interactive mode multiplier, A(*j*) = 1 + α log(1 + *t*_w / *t*_½) is a logarithmic age boost parameterised by wait time *t*_w and configurable half-life *t*_½, and U(*c*, *l*) is the rolling-window resource utilisation of class *c* in lane *l* over a recent time window. The utilisation denominator gives the discipline its fairness character: a class currently consuming a large share of lane capacity scores lower than an equally weighted idle class, producing a weighted max-min allocation across active classes analogous to Generalised Processor Sharing (GPS), but approximated in periodic discrete cycles rather than fluid flow. Logarithmic aging is preferred over linear aging to bound runaway priority elevation while still guaranteeing finite waiting — a job's score grows without bound but at a rate that decreases with wait time, suppressing the synchronised burst discharges that arise when many long-waiting jobs simultaneously reach a linear threshold.

Within each class the scheduler applies a secondary max-min fairness rule: among all students with pending jobs in the lane, it selects the one with the fewest currently running pods, breaking ties by earliest submit time. This min-running / FIFO discipline is a discrete analogue of per-user processor sharing, ensuring that no individual student can exhaust a course's dispatch share while holding active sessions. Admission is gated by a closed-loop capacity check — free capacity is computed as total lane capacity minus running resource units minus admitted-but-unplaced resource units — preventing the phantom-capacity over-commitment that arises in two-level scheduling architectures during the interval between pod admission and physical placement. Wait-time estimates are derived by modelling each running pod's residency as a truncated normal random variable; summing completion-probability integrals across all running pods gives the expected freed capacity as a function of time, and bisection inverts this to find the wait percentile corresponding to a job's queue rank. Per-course residency parameters are maintained online via Welford's algorithm with Bayesian shrinkage toward a cluster-wide prior, so estimates converge from prior toward course-specific behaviour as observations accumulate.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LaneSchedulerController                           │
│                                                             │
│  pod-watch thread ──► _handle_pod_event                     │
│     • enqueue pending pods                                   │
│     • track running pods                                     │
│     • record completions → ResidencyStats                   │
│                                                             │
│  node-watch thread ──► NodeCapacityTracker                  │
│     • maintain lane capacities from node taints             │
│                                                             │
│  cycle thread (10s) ──► Scheduler.cycle()                   │
│     • score queued jobs                                      │
│     • dispatch top-K per lane                               │
│     • patch admitted pods with toleration                   │
│                                                             │
│  wait-cache thread (60s) ──► _build_wait_snapshot()         │
│     • compute WaitEstimate for every queued pod             │
│     • publish Kubernetes Events                             │
│                                                             │
│  csv-reload thread (daily) ──► CourseRegistry               │
│     • reload registrar data                                 │
└─────────────────────────────────────────────────────────────┘
```

### Modules

| Module | Responsibility |
|---|---|
| `lane_scheduler/core/scheduler.py` | Core scheduling logic: priority scoring, within-class fairness, utilization tracking |
| `lane_scheduler/core/course_registry.py` | Course scheduling weights from registrar CSV; fallback for unknown courses |
| `lane_scheduler/core/node_capacity.py` | Tracks allocatable capacity per GPU class lane from node watch events |
| `lane_scheduler/k8s/controller.py` | Orchestrates all threads; Kubernetes API interactions |
| `lane_scheduler/k8s/pod_translator.py` | Translates Kubernetes pod dicts into scheduler domain objects |
| `lane_scheduler/k8s/event_publisher.py` | Kubernetes Event creation with per-pod emission schedules |
| `lane_scheduler/estimation/wait_estimator.py` | Truncated-normal wait-time estimation; background cache |
| `lane_scheduler/estimation/residency_stats.py` | Per-course Bayesian residency distribution tracking |
| `tools/simulate.py` | Synthetic workload simulation for offline testing |
| `deploy/manifests.yaml` | RBAC, Deployment, ConfigMap, and CronJob manifests |

---

## Scheduling Policy

### Resource Lanes

The cluster's resources are divided into independent scheduling lanes, one per GPU class plus one for CPU-only workloads:

```
Lane.CPU          — all non-GPU pods
Lane.GPU_XSMALL   — nodes tainted gpu-class=xsmall
Lane.GPU_SMALL    — nodes tainted gpu-class=small
Lane.GPU_MEDIUM   — nodes tainted gpu-class=medium
Lane.GPU_LARGE    — nodes tainted gpu-class=large
Lane.GPU_XLARGE   — nodes tainted gpu-class=xlarge
```

Lanes are discovered dynamically at controller startup by scanning node taints, so adding a new GPU class requires only a controller restart rather than a code change. Each lane maintains independent utilization accounting and a separate priority queue.

### Priority Score

Every queued job receives a score:

```
P(j, l) = W(c) × Mode(j) × Age(j) / U(c, l)
```

**`W(c)` — Class Weight**

The class weight `W` is a positive float supplied per-course in the registrar CSV. The scheduler uses it as-is; how it is derived is left to the operator. A natural starting point for a tiered academic environment is:

```
W(c) = tier / sqrt(seats)
```

where `tier` encodes academic level (e.g. 1 = lower-division, 2 = upper-division, 3 = graduate) and `seats` is enrollment. The square-root denominator dampens — but does not eliminate — the advantage of large courses: a 200-seat introductory section at tier 1 yields W ≈ 0.07, while a 15-seat graduate seminar at tier 3 yields W ≈ 0.77. Any positive value works; operators can freely encode alternative priority schemes.

**`Mode(j)` — Batch Penalty**

```
Mode(j) = 1.0   (interactive)
Mode(j) = 0.3   (batch, i.e. dsmlp/batch=true)
```

Batch jobs compete in the same lane as interactive jobs but score at 30% of their interactive equivalent. A batch job must accumulate roughly 3× the age boost of an interactive job to overtake it — achievable overnight when interactive load falls away.

**`Age(j)` — Wait-Time Boost**

```
Age(j) = 1 + α × log(1 + wait / t_half)
```

Logarithmic aging ensures no job waits indefinitely. Interactive jobs use `t_half = 10 min`; batch jobs use `t_half = 2 hr`, meaning batch jobs age more slowly (they are expected to wait longer) but will eventually rise if interactive demand subsides.

**`U(c, l)` — Class Utilization**

Rolling-window utilization over the past 5 minutes. Classes already consuming resources in a lane score lower, giving idle classes a natural boost.

### Within-Class Fairness

When multiple students in the same course have jobs waiting, the scheduler selects the student with the fewest currently running pods in that lane. Among students tied on running-pod count, the one with the oldest pending job (earliest submit time) is chosen. This prevents a student who already has a running session from blocking classmates who have none.

### Course Registry

Course metadata is loaded from a CSV file exported daily from the university registrar:

```
course_id,weight
CSE234_SP26_A00,0.775
CSE10_SP26_A00,0.069
CSE150_SP26_A00,0.270
```

`weight` is a positive float used directly as `W` in the priority formula. The operator computes it by whatever means reflect local policy — `tier / sqrt(seats)` is a common starting point for tiered academic environments. If a pod references a course not in the registry, the scheduler defaults to `weight=1.0`, logging a warning.

---

## Kubernetes Integration

### Taint/Toleration Gate

All managed nodes carry an inhibitory taint:

```
dsmlp/scheduling-gate=controller:NoSchedule
```

Pods submitted by students remain Pending until the controller selects them. At that point the controller patches the pod with a matching toleration, and the default Kubernetes scheduler takes over placement. This design keeps the controller simple — it only decides *when* to admit a pod, not *where* to place it.

### GPU Class Routing

An existing mutating admission controller (outside this project's scope) automatically adds `nodeSelector` and `gpu-class` tolerations to GPU pods based on their `gpu-class` label. This controller reads the same label to determine which lane a pod belongs to.

### Pod Identity

| Field | Source |
|---|---|
| Student (scheduling entity) | Pod namespace (one namespace per student) |
| Course | `dsmlp/course` label |
| Lane | `gpu-class` label (absent → CPU lane) |
| Batch mode | `dsmlp/batch=true` label |
| Resource units | `resources.requests` (CPU cores or GPU count) |

---

## Wait-Time Estimation

### Model

For each running pod with `activeDeadlineSeconds = D`, residency is modelled as:

```
X ~ Normal(μ, σ)   where μ = mean_pct × D,  σ = std_pct × D
```

The probability that a pod running for `age` seconds finishes within the next `t` seconds is computed from the left-truncated normal CDF using `math.erf` (no external dependencies):

```
P(finish in t | survived to age) = [Φ((age+t − μ)/σ) − Φ((age − μ)/σ)]
                                    ─────────────────────────────────────
                                              1 − Φ((age − μ)/σ)
```

Summing this probability across all running pods in a lane gives the expected number of resource slots freed within `t` seconds. Bisection over `t` inverts this to find the wait time at which the expected freed slots equals the pod's queue rank.

The result is a `WaitEstimate` with median, P20 (optimistic), and P80 (pessimistic) in seconds.

### Per-Course Adaptation

The cluster-wide residency parameters (`mean_pct`, `std_pct`) serve as a Bayesian prior. As pods complete, the controller records each pod's actual residency fraction and updates per-course, per-lane, per-mode statistics using Welford's online algorithm.

The posterior blends prior and observations via a pseudo-count `prior_weight` (default 10):

```
posterior_mean = (prior_weight × prior_mean + n × sample_mean) / (prior_weight + n)
```

With few observations a course's estimate stays close to the cluster prior. With many (after ~30 completions), it reflects the course's own typical usage patterns — useful for assignments where all students tend to run similar workloads.

Pods killed at their `activeDeadlineSeconds` deadline are recorded as 100% residency.

### Background Cache

Computing wait estimates for all queued pods is CPU-intensive (~7 seconds for 200 queued pods against 600 running). Estimates are recomputed on a background thread every 60 seconds and cached atomically. Callers receive the most recent snapshot with no blocking. Failed snapshots leave the previous cache intact.

---

## Queue Position Events

The controller publishes Kubernetes Events for each queued pod, visible via:

```bash
kubectl describe pod <pod-name> -n <student-namespace>
```

Example event message:
```
Queue position: #3 overall in gpu-medium lane, #1 within course.
Estimated wait: 8.2m (optimistic 4.1m, pessimistic 14.7m).
```

### Emission Schedule

**Interactive pods:**
- Immediately after entering the queue (next snapshot)
- Every 60 seconds for the first 5 minutes
- Every 5 minutes thereafter

**Batch pods:**
- Immediately after entering the queue
- At 25%, 50%, and 75% of the initial median wait estimate
- No further events after the 75% milestone

Batch milestone times are anchored to the first estimated median wait and do not shift as the queue evolves, ensuring predictable notification timing. Each event is created as a distinct Kubernetes object (not an update to an existing one) so that the event stream shows a genuine history of position changes.

---

## Configuration

All tuning parameters are configurable via environment variables or CLI flags.

| Environment Variable | Default | Description |
|---|---|---|
| `LANE_COURSE_CSV` | `/etc/lane-scheduler/courses.csv` | Path to registrar CSV |
| `LANE_CYCLE_INTERVAL` | `10` | Scheduling cycle interval (seconds) |
| `LANE_DISPATCH_K` | `8` | Max jobs dispatched per lane per cycle |
| `LANE_ALPHA` | `1.0` | Wait-time aging scaling factor |
| `LANE_T_HALF_INTERACTIVE` | `600` | Interactive job aging half-life (seconds) |
| `LANE_T_HALF_BATCH` | `7200` | Batch job aging half-life (seconds) |
| `LANE_EPSILON` | `0.01` | Utilization floor (prevents division by zero) |
| `LANE_UTIL_WINDOW` | `300` | Rolling utilization window (seconds) |
| `LANE_RELOAD_INTERVAL` | `86400` | Course CSV reload interval (seconds) |
| `LANE_INTERACTIVE_MEAN_PCT` | `0.4` | Prior: interactive mean residency fraction |
| `LANE_INTERACTIVE_STD_PCT` | `0.2` | Prior: interactive residency std fraction |
| `LANE_BATCH_MEAN_PCT` | `0.7` | Prior: batch mean residency fraction |
| `LANE_BATCH_STD_PCT` | `0.15` | Prior: batch residency std fraction |
| `LANE_PRIOR_WEIGHT` | `10.0` | Bayesian prior pseudo-count |
| `LANE_WAIT_CACHE_INTERVAL` | `60` | Wait estimate refresh interval (seconds) |

---

## Deployment

### Prerequisites

- Kubernetes cluster with nodes tainted `dsmlp/scheduling-gate=controller:NoSchedule`
- GPU nodes additionally tainted `gpu-class=<class>:NoSchedule` and labelled accordingly
- One namespace per student
- Existing mutating admission controller injecting `gpu-class` nodeSelector/tolerations

### Node Setup

```bash
# Taint all managed nodes
kubectl taint nodes <node> dsmlp/scheduling-gate=controller:NoSchedule

# Label GPU nodes by class
kubectl taint nodes <gpu-node> gpu-class=medium:NoSchedule
```

### Apply Manifests

```bash
kubectl create namespace lane-scheduler
kubectl apply -f deploy/manifests.yaml
```

The `manifests.yaml` includes RBAC (ServiceAccount, ClusterRole, ClusterRoleBinding), a ConfigMap for the course CSV, the controller Deployment, and a daily CronJob that refreshes the course ConfigMap from the registrar.

### Required RBAC Permissions

```yaml
- pods:   get, list, watch, patch
- nodes:  get, list, watch
- events: create
```

### Pod Labels

Students must label their pods:

```yaml
metadata:
  labels:
    dsmlp/course: CSE234_SP26_A00      # required
    gpu-class: medium                   # required for GPU pods
    dsmlp/batch: "true"                 # optional, default interactive
```

---

## Design Decisions

**Why not a custom Kubernetes scheduler?** Custom schedulers must handle all scheduling decisions including node affinity, resource fitting, and pod topology. Using taints as a gate lets us focus purely on fairness policy while delegating placement to the well-tested default scheduler.

**Why dynamic Lane enum?** GPU hardware classes are managed by a separate infrastructure team and change independently of the scheduler codebase. Building lanes from node taints at startup means a new GPU class requires only a controller restart rather than a code change and redeployment.

**Why Bayesian shrinkage rather than pure course data?** Course sections may have few completions early in a semester, or may never accumulate enough data if they are small. The prior ensures estimates are always reasonable, while the shrinkage ensures the system learns when data is available. The `prior_weight` parameter controls the trade-off explicitly.

**Why log aging rather than linear?** Linear aging can cause runaway priority elevation for very long-waiting jobs, potentially causing large single-job bursts when they finally dispatch. Logarithmic aging rises steeply at first (preventing short-term starvation) but flattens out, producing smoother dispatch behavior at high queue depths.

**Why fresh Event creates rather than updates?** Kubernetes deduplicates events with identical reason/message by incrementing a `count` field rather than creating a new timeline entry. Creating distinct events (with unique names per emission) ensures that `kubectl describe pod` shows a genuine chronological history of queue position changes, which is more useful to students than a single entry with an incrementing counter.

---

## Testing

```bash
pip install -e .
python -m pytest tests/
```

200+ tests, all passing. No external dependencies beyond the `kubernetes` Python client (used only in `lane_scheduler/k8s/controller.py`). All scheduling logic, wait estimation, residency statistics, and event scheduling are tested with stdlib only.

The `tools/simulate.py` harness runs a synthetic workload against the scheduler for offline validation of fairness properties without a live cluster:

```bash
python tools/simulate.py
```
