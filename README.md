# Lane-based Priority Scheduler

A Kubernetes controller that fairly allocates cluster resources across university scheduling groups (courses), preventing high-volume groups from crowding out lower-volume ones, while respecting heterogeneous GPU hardware classes.

---

## Goals

University teaching clusters face a structural fairness problem: a high-enrollment course will naturally generate far more queued jobs than a small one, causing lower-volume courses to starve during peak periods even when they warrant equal scheduling priority.

This scheduler addresses that by:

- Giving every scheduling group equal weight (W = 1.0) so the priority formula treats all groups symmetrically; per-group weights are a planned future capability
- Distributing a group's share fairly among its individual users, so no single active user within a scheduling group monopolizes its allocation
- Preferring interactive jobs over batch jobs within any resource lane, while ensuring batch jobs eventually drain overnight
- Providing students with real-time queue position and estimated wait time via Kubernetes Events visible in `kubectl describe pod`
- Adapting wait-time estimates over time using per-group completion data, so estimates improve as the term progresses

The scheduler is implemented as a Kubernetes controller. It does not replace the default Kubernetes scheduler; instead it acts as a gating layer. Pods of interest arrive in a SchedulingGated state — held by a `lane-scheduler` scheduling gate injected by an external mutating admission controller. When a pod reaches the top of the priority queue, the controller patches in the matching GPU-class `nodeSelector` and `NoSchedule` toleration and removes the scheduling gate, allowing the default scheduler to place the pod normally.

### Scheduling Model

The system operates as a set of independent finite-capacity multi-server queues (lanes), one per GPU hardware class. Only pods that arrive bearing the `lane-scheduler` scheduling gate are managed; pods without a recognised `gpu-class` label are rejected with an error Event. Each lane accepts a multi-group open workload of heterogeneous jobs; jobs are held in the SchedulingGated state and released in discrete scheduling cycles. The service discipline within each lane is a dynamic non-preemptive priority rule: at each cycle, queued jobs are scored and the top-*K* are admitted subject to a capacity constraint.

The per-job priority score is P(*j*, *l*) = W(*g*) · M(*j*) · A(*j*) / U(*g*, *l*), where W(*g*) = 1.0 for all scheduling groups (equal weight; per-group configuration is deferred), M(*j*) ∈ {0.3, 1.0} is a batch/interactive mode multiplier, A(*j*) = 1 + α log(1 + *t*_w / *t*_½) is a logarithmic age boost parameterised by wait time *t*_w and configurable half-life *t*_½, and U(*g*, *l*) is the rolling-window resource utilisation of scheduling group *g* in lane *l* over a recent time window. The utilisation denominator gives the discipline its fairness character: a group currently consuming a large share of lane capacity scores lower than an equally weighted idle group, producing a weighted max-min allocation across active groups analogous to Generalised Processor Sharing (GPS), but approximated in periodic discrete cycles rather than fluid flow. Logarithmic aging is preferred over linear aging to bound runaway priority elevation while still guaranteeing finite waiting.

Within each scheduling group the scheduler applies a secondary max-min fairness rule: among all users with pending jobs in the lane, it selects the one with the fewest currently running pods, breaking ties by earliest submit time. This min-running / FIFO discipline is a discrete analogue of per-user processor sharing, ensuring that no individual user can exhaust a group's dispatch share while holding active sessions.

Admission is gated by a closed-loop capacity check — free capacity is computed as:

```
free = lane_cap − running_units − kubernetes_pending_units − admitted_units
```

This three-term deduction prevents over-commitment during the interval between pod admission and physical placement. Pods transition through four states: SchedulingGated → admitted (reserved in `_admitted_resources`) → k8s-Pending (tracked in `_kubernetes_pending`, resource reservation maintained) → Running (tracked in `_kubernetes_running`).

Wait-time estimates are derived by modelling each running pod's residency as a truncated normal random variable; summing completion-probability integrals across all running pods gives the expected freed capacity as a function of time, and bisection inverts this to find the wait percentile corresponding to a job's queue rank. Per-group residency parameters are maintained online via Welford's algorithm with Bayesian shrinkage toward a cluster-wide prior.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LaneSchedulerController                  │
│                                                             │
│  pod-watch thread ──► _handle_pod_event                     │
│     • enqueue SchedulingGated pods                          │
│     • track: admitted → k8s-Pending → Running transitions   │
│     • record completions → ResidencyStats                   │
│                                                             │
│  node-watch thread ──► NodeCapacityTracker                  │
│     • maintain lane capacities from node labels             │
│                                                             │
│  cycle thread (10s) ──► Scheduler.cycle()                   │
│     • score queued jobs                                      │
│     • dispatch top-K per lane                               │
│     • patch admitted pods (nodeSelector + toleration +      │
│       scheduling gate removal)                              │
│                                                             │
│  wait-cache thread (60s) ──► _build_wait_snapshot()         │
│     • compute WaitEstimate for every queued pod             │
│     • publish Kubernetes Events                             │
└─────────────────────────────────────────────────────────────┘
```

### Modules

| Module | Responsibility |
|---|---|
| `lane_scheduler/core/scheduler.py` | Core scheduling logic: priority scoring, within-group fairness, utilization tracking |
| `lane_scheduler/core/sched_group_registry.py` | Stub registry returning W=1.0 for all scheduling groups; per-group weights deferred |
| `lane_scheduler/core/node_capacity.py` | Tracks allocatable capacity per GPU class lane from node watch events |
| `lane_scheduler/k8s/controller.py` | Orchestrates all threads; Kubernetes API interactions |
| `lane_scheduler/k8s/pod_translator.py` | Translates Kubernetes pod dicts into scheduler domain objects |
| `lane_scheduler/k8s/event_publisher.py` | Kubernetes Event creation with per-pod emission schedules |
| `lane_scheduler/estimation/wait_estimator.py` | Truncated-normal wait-time estimation; background cache |
| `lane_scheduler/estimation/residency_stats.py` | Per-group Bayesian residency distribution tracking |
| `tools/simulate.py` | Synthetic workload simulation for offline testing |
| `deploy/manifests.yaml` | RBAC and Deployment manifests |

---

## Scheduling Policy

### Resource Lanes

The cluster's resources are divided into independent scheduling lanes, one per GPU hardware class:

```
Lane.GPU_XSMALL   — nodes labelled/tainted gpu-class=xsmall
Lane.GPU_SMALL    — nodes labelled/tainted gpu-class=small
Lane.GPU_MEDIUM   — nodes labelled/tainted gpu-class=medium
Lane.GPU_LARGE    — nodes labelled/tainted gpu-class=large
Lane.GPU_XLARGE   — nodes labelled/tainted gpu-class=xlarge
```

Only pods bearing the `lane-scheduler` scheduling gate are managed. A gated pod without a `gpu-class` label, or with an unrecognised class, is rejected immediately with a Warning Event and ignored. Lanes are discovered dynamically at controller startup by scanning node labels, so adding a new GPU class requires only a controller restart rather than a code change.

### Priority Score

Every queued job receives a score:

```
P(j, l) = W(g) × Mode(j) × Age(j) / U(g, l)
```

**`W(g)` — Scheduling Group Weight**

All scheduling groups default to W = 1.0 (equal weight). The `SchedGroupRegistry` is currently a stub; per-group weight configuration is planned for a future release.

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

**`U(g, l)` — Group Utilization**

Rolling-window utilization over the past 5 minutes. Groups already consuming resources in a lane score lower, giving idle groups a natural boost.

### Within-Group Fairness

When multiple users in the same scheduling group have jobs waiting, the scheduler selects the user with the fewest currently running pods in that lane. Among users tied on running-pod count, the one with the oldest pending job (earliest submit time) is chosen. This prevents a user who already has a running session from blocking others who have none.

---

## Kubernetes Integration

### Scheduling Gate Pattern

An external mutating admission controller (outside this project's scope) injects two things onto every GPU pod before it reaches the lane scheduler:

1. A scheduling gate: `spec.schedulingGates: [{name: "lane-scheduler"}]` — this places the pod in the SchedulingGated / Pending state so the default Kubernetes scheduler ignores it.
2. The `gpu-class` label (e.g. `gpu-class: medium`) — used to route the pod to the correct lane.

The lane scheduler watches for pods with this gate. When a pod wins a scheduling cycle, the controller issues a single PATCH that:

- Adds `spec.nodeSelector: {gpu-class: <class>}` to target the correct GPU nodes.
- Adds a `gpu-class=<class>:NoSchedule` toleration to satisfy the node taint.
- Removes the `lane-scheduler` scheduling gate, releasing the pod to the default Kubernetes scheduler.

### Pod Identity

| Field | Source |
|---|---|
| User (fairness entity) | `dsmlp/user` label (falls back to namespace if absent) |
| Scheduling group | `dsmlp/course` label |
| Lane | `gpu-class` label (absent or unknown → Warning Event, pod ignored) |
| Batch mode | `dsmlp/batch=true` label |
| Resource units | `resources.requests` (GPU count, floor 1.0) |

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

### Per-Group Adaptation

The cluster-wide residency parameters (`mean_pct`, `std_pct`) serve as a Bayesian prior. As pods complete, the controller records each pod's actual residency fraction and updates per-group, per-lane, per-mode statistics using Welford's online algorithm.

The posterior blends prior and observations via a pseudo-count `prior_weight` (default 10):

```
posterior_mean = (prior_weight × prior_mean + n × sample_mean) / (prior_weight + n)
```

With few observations a group's estimate stays close to the cluster prior. With many (after ~30 completions), it reflects the group's own typical usage patterns.

### Background Cache

Computing wait estimates for all queued pods is CPU-intensive. Estimates are recomputed on a background thread every 60 seconds and cached atomically. Callers receive the most recent snapshot with no blocking. Failed snapshots leave the previous cache intact.

---

## Queue Position Events

The controller publishes Kubernetes Events for each queued pod, visible via:

```bash
kubectl describe pod <pod-name> -n <student-namespace>
```

Example event message:
```
Queue position: #3 overall in gpu-medium lane, #1 within scheduling group.
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

---

## Configuration

All tuning parameters are configurable via environment variables or CLI flags.

| Environment Variable | Default | Description |
|---|---|---|
| `LANE_CYCLE_INTERVAL` | `10` | Scheduling cycle interval (seconds) |
| `LANE_DISPATCH_K` | `8` | Max jobs dispatched per lane per cycle |
| `LANE_ALPHA` | `1.0` | Wait-time aging scaling factor |
| `LANE_T_HALF_INTERACTIVE` | `600` | Interactive job aging half-life (seconds) |
| `LANE_T_HALF_BATCH` | `7200` | Batch job aging half-life (seconds) |
| `LANE_EPSILON` | `0.01` | Utilization floor (prevents division by zero) |
| `LANE_UTIL_WINDOW` | `300` | Rolling utilization window (seconds) |
| `LANE_INTERACTIVE_MEAN_PCT` | `0.4` | Prior: interactive mean residency fraction |
| `LANE_INTERACTIVE_STD_PCT` | `0.2` | Prior: interactive residency std fraction |
| `LANE_BATCH_MEAN_PCT` | `0.7` | Prior: batch mean residency fraction |
| `LANE_BATCH_STD_PCT` | `0.15` | Prior: batch residency std fraction |
| `LANE_PRIOR_WEIGHT` | `10.0` | Bayesian prior pseudo-count |
| `LANE_WAIT_CACHE_INTERVAL` | `60` | Wait estimate refresh interval (seconds) |
| `LANE_USER_LABEL` | `dsmlp/user` | Pod label key for user identity (falls back to namespace) |
| `LANE_COURSE_LABEL` | `dsmlp/course` | Pod label key for scheduling group |
| `LANE_BATCH_LABEL` | `dsmlp/batch` | Pod label key for batch mode |
| `LANE_GPU_CLASS_LABEL` | `gpu-class` | Label key on pods and nodes for GPU class |
| `LANE_SCHEDULING_GATE_NAME` | `lane-scheduler` | Scheduling gate name injected by webhook |

---

## Deployment

### Prerequisites

- Kubernetes cluster with GPU nodes labelled and tainted by `gpu-class`
- Mutating admission controller that injects `schedulingGates: [{name: "lane-scheduler"}]`, the `gpu-class` label, and the `dsmlp/user` label onto GPU pods

### Node Setup

```bash
kubectl label nodes <gpu-node> gpu-class=medium
kubectl taint nodes <gpu-node> gpu-class=medium:NoSchedule
```

### Apply Manifests

```bash
kubectl create namespace lane-scheduler
kubectl apply -f deploy/manifests.yaml
```

### Pod Labels

The mutating admission controller injects the scheduling gate, `gpu-class`, and `dsmlp/user` labels automatically. Student workload manifests should supply:

```yaml
metadata:
  labels:
    dsmlp/course: CSE234_SP26_A00      # scheduling group
    dsmlp/batch: "true"                 # optional; absent = interactive
```

---

## Testing

```bash
pip install -e .
python -m pytest tests/
```

250+ tests, all passing. No external dependencies beyond the `kubernetes` Python client (used only in `lane_scheduler/k8s/controller.py`). All scheduling logic, wait estimation, residency statistics, and event scheduling are tested with stdlib only.

The `tools/simulate.py` harness runs a synthetic workload against the scheduler for offline validation of fairness properties without a live cluster:

```bash
python tools/simulate.py
```
