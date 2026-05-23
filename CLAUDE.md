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

There is no separate lint or format step configured.

## Architecture

This is a **multi-threaded Kubernetes controller** implementing weighted fair-share scheduling for a university GPU teaching cluster. The core scheduler has **zero Kubernetes dependencies** — only `lane_scheduler/k8s/controller.py` imports the `kubernetes` client.

### Thread model

`LaneSchedulerController` (k8s/controller.py) runs four independent threads:

| Thread | Role |
|--------|------|
| pod-watch | Enqueues Pending pods; tracks Running/Completed transitions |
| node-watch | Feeds `NodeCapacityTracker` with allocatable capacity per lane |
| cycle (10 s) | `Scheduler.cycle()` scores jobs and dispatches top-K per lane by patching pod tolerations |
| csv-reload (daily) | Reloads registrar CSV into `CourseRegistry` |

A background goroutine-style loop (60 s) refreshes `WaitTimeCache` and publishes Kubernetes Events for queued pods.

### Admission gate pattern

Every managed node carries an inhibitory taint (`dsmlp/scheduling-gate=controller:NoSchedule`). The controller **patches** admitted pods with the matching toleration; only then does the default Kubernetes scheduler place them. GPU nodes additionally carry `gpu-class=<class>:NoSchedule` taints so lane routing is enforced at the node level.

### Lane model

`Lane` is a dynamic enum built at controller startup by reading node **labels**. It always includes a CPU lane plus one entry per GPU class discovered (xsmall, small, medium, large, xlarge). No code changes are needed when new GPU classes are added — only a controller restart.

### Priority scoring formula

```
P(job, lane) = W(course) × Mode(job) × Age(job) / U(course, lane)
```

- **W** = `tier_weight / √enrollment`  (tier weights: intro=1, upper=2, grad=3)
- **Mode** = 1.0 (interactive) or 0.3 (batch)
- **Age** = `1 + α × log(1 + wait / t_half)`  (logarithmic, prevents starvation)
- **U** = 5-minute rolling utilization (idle courses score higher)

Within-class fairness is enforced by a **deficit round-robin** `DeficitTracker` so no single student monopolizes a course's allocation.

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
| `LANE_COURSE_CSV` | `/etc/lane-scheduler/courses.csv` | Registrar CSV path |
| `LANE_NODE_GPU_CLASS_LABEL` | `gpu-class` | Node label key used to identify GPU class/lane |
| `LANE_POD_GPU_CLASS_LABEL` | `gpu-class` | Pod label key used to identify requested GPU class/lane |
| `LANE_COURSE_LABEL` | `dsmlp/course` | Pod label key used to identify the course |
| `LANE_INHIBIT_TAINT_KEY` | `dsmlp/scheduling-gate` | Taint key on managed nodes (inhibitory gate) |
| `LANE_INHIBIT_TAINT_VALUE` | `controller` | Taint value paired with the inhibitory gate key |

### Package layout

```
lane_scheduler/
  core/
    scheduler.py        # Priority scoring, Lane enum, Scheduler, DeficitTracker
    course_registry.py  # Loads/parses registrar CSV; tier inference
    node_capacity.py    # Watches node labels; aggregates per-lane capacity
  k8s/
    controller.py       # Thread orchestration; Kubernetes API calls
    pod_translator.py   # K8s pod dicts ↔ Job domain objects
    event_publisher.py  # Kubernetes Events for queued pods (emission schedule)
  estimation/
    wait_estimator.py   # Truncated-normal model; bisection; WaitTimeCache
    residency_stats.py  # Welford online stats; Bayesian shrinkage
tools/
  simulate.py           # Offline fairness simulator
tests/                  # 200+ tests; mirrors core/ + estimation/ + k8s/ structure
deploy/
  manifests.yaml        # RBAC, Deployment, ConfigMap, CronJob
```
