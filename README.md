# DSMLP Priority Scheduler

A Kubernetes controller that fairly allocates cluster resources across university courses, preventing large introductory classes from crowding out smaller graduate seminars while respecting heterogeneous GPU hardware classes.

---

## Goals

University teaching clusters face a structural fairness problem: a 200-student introductory course will naturally generate far more queued jobs than a 12-student graduate seminar, causing smaller classes to starve during peak periods even when those smaller classes have higher pedagogical priority.

This scheduler addresses that by:

- Weighting courses by academic tier (graduate > upper-division > introductory) and inversely by enrollment, so a small graduate seminar competes on equal or better footing with a large introductory section
- Distributing a course's share fairly among its individual students, so no single active student within a course monopolizes its allocation
- Preferring interactive jobs over batch jobs within any resource lane, while ensuring batch jobs eventually drain overnight
- Providing students with real-time queue position and estimated wait time via Kubernetes Events visible in `kubectl describe pod`
- Adapting wait-time estimates over time using course-specific completion data, so estimates improve as the semester progresses

The scheduler is implemented as a Kubernetes controller. It does not replace the default Kubernetes scheduler; instead it acts as a gating layer, holding pods in a Pending state via node taints until they reach the top of the priority queue, then patching in a toleration that allows the default scheduler to place them normally.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DSMLPController                           │
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
| `scheduler.py` | Core scheduling logic: priority scoring, deficit round-robin, utilization tracking |
| `course_registry.py` | Course metadata (tier, enrollment) from registrar CSV with inference fallback |
| `pod_translator.py` | Translates Kubernetes pod dicts into scheduler domain objects |
| `node_capacity.py` | Tracks allocatable capacity per GPU class lane from node watch events |
| `controller.py` | Orchestrates all threads; Kubernetes API interactions |
| `wait_estimator.py` | Truncated-normal wait-time estimation; background cache |
| `residency_stats.py` | Per-course Bayesian residency distribution tracking |
| `event_publisher.py` | Kubernetes Event creation with per-pod emission schedules |
| `simulate.py` | Synthetic workload simulation for offline testing |
| `manifests.yaml` | RBAC, Deployment, ConfigMap, and CronJob manifests |

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

```
W(c) = tier_weight(c) / √enrollment(c)
```

Tier weights are 1 (introductory), 2 (upper-division), and 3 (graduate). The square-root enrollment denominator softens but does not eliminate the size difference — a 200-student intro class gets less weight than a 12-student grad seminar, but is not ignored entirely.

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

When multiple students in the same course have jobs waiting, the scheduler uses deficit round-robin to select which student's job to promote to the global queue. Each student accrues a deficit (proportional to the class weight and elapsed time) while waiting; the student with the highest deficit is promoted first. This prevents a single active student from monopolizing a class's share of resources.

### Course Registry

Course metadata is loaded from a CSV file exported daily from the university registrar:

```
course_id,level,seats
CSE234_SP26_A00,graduate,18
CSE101_SP26_A00,lower,210
CSE150_SP26_A00,upper,55
```

If a pod references a course not in the registry, the tier is inferred from the numeric portion of the course code: codes below 100 are lower-division, 100–199 are upper-division, and 200 and above are graduate. Missing enrollment data defaults to 200 for lower/upper courses and 50 for graduate courses.

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
| `DSMLP_COURSE_CSV` | `/etc/dsmlp/courses.csv` | Path to registrar CSV |
| `DSMLP_CYCLE_INTERVAL` | `10` | Scheduling cycle interval (seconds) |
| `DSMLP_DISPATCH_K` | `8` | Max jobs dispatched per lane per cycle |
| `DSMLP_ALPHA` | `1.0` | Wait-time aging scaling factor |
| `DSMLP_T_HALF_INTERACTIVE` | `600` | Interactive job aging half-life (seconds) |
| `DSMLP_T_HALF_BATCH` | `7200` | Batch job aging half-life (seconds) |
| `DSMLP_EPSILON` | `0.01` | Utilization floor (prevents division by zero) |
| `DSMLP_UTIL_WINDOW` | `300` | Rolling utilization window (seconds) |
| `DSMLP_RELOAD_INTERVAL` | `86400` | Course CSV reload interval (seconds) |
| `DSMLP_INTERACTIVE_MEAN_PCT` | `0.4` | Prior: interactive mean residency fraction |
| `DSMLP_INTERACTIVE_STD_PCT` | `0.2` | Prior: interactive residency std fraction |
| `DSMLP_BATCH_MEAN_PCT` | `0.7` | Prior: batch mean residency fraction |
| `DSMLP_BATCH_STD_PCT` | `0.15` | Prior: batch residency std fraction |
| `DSMLP_PRIOR_WEIGHT` | `10.0` | Bayesian prior pseudo-count |
| `DSMLP_WAIT_CACHE_INTERVAL` | `60` | Wait estimate refresh interval (seconds) |

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
kubectl create namespace dsmlp-system
kubectl apply -f manifests.yaml
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
python -m unittest test_scheduler test_integration test_wait_estimator \
                   test_event_publisher test_residency_stats
```

158 tests, all passing. No external dependencies beyond the `kubernetes` Python client (used only in `controller.py`). All scheduling logic, wait estimation, residency statistics, and event scheduling are tested with stdlib only.

The `simulate.py` harness runs a synthetic workload against the scheduler for offline validation of fairness properties without a live cluster.
