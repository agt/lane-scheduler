# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install and run tests:**
```bash
pip install -e .
python -m pytest tests/
```

**Run a single test file or test:**
```bash
python -m pytest tests/test_scheduler.py
python -m pytest tests/test_scheduler.py::test_priority_score_basic
```

**Offline workload simulation (no cluster required):**
```bash
python tools/simulate.py
```

**Build Docker image:**
```bash
docker build -t lane-scheduler .
```

**Regenerate docs diagrams** (run whenever `docs/scheduler-flow.dot` changes):
```bash
dot -Tsvg docs/scheduler-flow.dot -o docs/scheduler-flow.svg
dot -Tpng docs/scheduler-flow.dot -o docs/scheduler-flow.png
```

Requires Graphviz (`apt-get install -y graphviz` or `brew install graphviz`). Commit both output files alongside any `.dot` change.

There is no separate lint or format step configured.

## Architecture

This is a **multi-threaded Kubernetes controller** implementing weighted fair-share scheduling for a university GPU teaching cluster. The core scheduler has **zero Kubernetes dependencies** — only `lane_scheduler/k8s/controller.py` imports the `kubernetes` client.

### Thread model

`LaneSchedulerController` (k8s/controller.py) runs three independent threads:

| Thread | Role |
|--------|------|
| pod-watch | Enqueues SchedulingGated pods; tracks Pending/Running/Completed transitions |
| node-watch | Feeds `NodeCapacityTracker` with allocatable capacity per lane |
| cycle (10 s) | `Scheduler.cycle()` scores jobs and dispatches top-K per lane by patching pod tolerations |

A background goroutine-style loop (60 s) refreshes `WaitTimeCache` and publishes Kubernetes Events for queued pods.

### Admission gate pattern

Pods arrive in `SchedulingGated` state with a `schedulingGates: [{name: "lane-scheduler"}]` entry injected by an external mutating admission controller. GPU nodes carry `gpu-class=<class>:NoSchedule` taints. When admitting a pod the controller patches it to: (a) add a `nodeSelector` entry (`gpu-class=<class>`), (b) add the matching `gpu-class=<class>:NoSchedule` toleration, and (c) remove the scheduling gate. Only then does the default Kubernetes scheduler place the pod.

### Lane model

`Lane` is a dynamic enum built at controller startup by reading node **labels**. It always includes a CPU lane plus one entry per GPU class discovered (xsmall, small, medium, large, xlarge). No code changes are needed when new GPU classes are added — only a controller restart.

### Priority scoring formula

```
P(job, lane) = W(g) × Mode(job) × Age(job) / U(g, lane)
```

- **W** = 1.0 for all scheduling groups (`SchedGroupRegistry` is a stub; per-group weights are a future capability)
- **Mode** = 1.0 (interactive) or 0.3 (batch)
- **Age** = `1 + α × log(1 + wait / t_half)`  (logarithmic, prevents starvation)
- **U** = 5-minute rolling utilization (idle scheduling groups score higher)

Within-group fairness is enforced by selecting the user with the **fewest running pods** in that lane; ties broken by oldest pending job submit time (FIFO).

### Pod state machine

Pods transition through four states tracked by the controller:

1. `SchedulingGated` → added to `_pending`
2. Admitted (after patch) → entry in `_admitted_resources` + `_admitted_with_timestamp`
3. k8s-Pending (gate removed, not yet Running) → removed from `_admitted_resources`; added to `_kubernetes_pending`
4. Running → removed from `_kubernetes_pending`; added to `_kubernetes_running`

Capacity gate: `free = lane_cap − running_units − kubernetes_pending_units − admitted_units`

### Wait-time estimation

`WaitEstimator` (estimation/wait_estimator.py) models job residency as a truncated-normal distribution. It uses a bisection solver to compute median/P20/P80 wait times without any scipy dependency — all math uses `math.erf` from stdlib. `ResidencyStats` applies a Welford online algorithm with Bayesian shrinkage toward a cluster-wide prior (controlled by `LANE_PRIOR_WEIGHT`).

### Key environment variables

All defaults live in `controller.py` lines ~98–127. The most operationally relevant:

| Variable | Default | Notes |
|----------|---------|-------|
| `LANE_CYCLE_INTERVAL` | 10 s | Scheduling cadence |
| `LANE_DISPATCH_K` | 8 | Max jobs admitted per lane per cycle |
| `LANE_ALPHA` | 1.0 | Aging boost scale |
| `LANE_T_HALF_INTERACTIVE` | 600 s | Interactive aging half-life |
| `LANE_T_HALF_BATCH` | 7200 s | Batch aging half-life |
| `LANE_PRIOR_WEIGHT` | 10.0 | Bayesian pseudo-count for residency convergence |
| `LANE_EWMA_ALPHA` | 0.1 | EWMA smoothing factor for per-group residency; higher = faster adaptation to recent data |
| `LANE_GPU_CLASS_LABEL` | `gpu-class` | Label key on both pods and nodes identifying the GPU class/lane |
| `LANE_SCHEDULING_GATE_NAME` | `lane-scheduler` | Name of the scheduling gate injected by the mutating webhook |
| `LANE_COURSE_LABEL` | `dsmlp/course` | Pod label key used to identify the scheduling group |
| `LANE_USER_LABEL` | `dsmlp/user` | Pod label key used to identify the user (falls back to namespace) |
| `LANE_BATCH_LABEL` | `dsmlp/batch` | Pod label key used to identify batch-mode jobs |

### Package layout

```
lane_scheduler/
  core/
    scheduler.py            # Priority scoring, Lane strings, Scheduler
    sched_group_registry.py # Stub registry; all scheduling groups default to weight=1.0
    node_capacity.py        # Watches node labels; aggregates per-lane capacity
  k8s/
    controller.py           # Thread orchestration; Kubernetes API calls
    pod_translator.py       # K8s pod dicts ↔ Job domain objects
    event_publisher.py      # Kubernetes Events for queued pods (emission schedule)
  estimation/
    wait_estimator.py       # Truncated-normal model; bisection; WaitTimeCache
    residency_stats.py      # EWMA online stats; Bayesian shrinkage toward cluster prior
  web/
    snapshot.py             # Build JSON snapshot of controller state for dashboard
    server.py               # Stdlib HTTP server; / and /api/snapshot endpoints
tools/
  simulate.py           # Offline fairness simulator
tests/                  # 250+ tests; mirrors core/ + estimation/ + k8s/ structure
deploy/
  manifests.yaml        # RBAC, Deployment
```
