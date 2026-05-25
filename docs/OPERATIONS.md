# Lane Scheduler Operations Guide

This guide is for cluster administrators responsible for deploying and operating the lane-scheduler on a university GPU teaching cluster.

---

## Table of Contents

1. [Quick-Start Checklist](#1-quick-start-checklist)
2. [Cluster-Level Labels and Taints](#2-cluster-level-labels-and-taints)
3. [Pod Labels](#3-pod-labels)
4. [Course Registry CSV](#4-course-registry-csv)
5. [Scheduling Algorithm Knobs](#5-scheduling-algorithm-knobs)
6. [Operational Flags](#6-operational-flags)
7. [Kubernetes Deployment and RBAC](#7-kubernetes-deployment-and-rbac)
8. [Observability](#8-observability)
9. [Scenario: Dry-Run Mode](#9-scenario-dry-run-mode)
10. [Scenario: Invalid GPU Class Events](#10-scenario-invalid-gpu-class-events)
11. [Tuning Reference](#11-tuning-reference)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Quick-Start Checklist

```
[ ] Label each GPU node with its gpu-class
[ ] Apply per-class gpu-class=<class>:NoSchedule taints on GPU nodes
[ ] Deploy the mutating admission controller that injects schedulingGates and gpu-class label
[ ] Mount the course registry CSV
[ ] Create the ServiceAccount, ClusterRole, and ClusterRoleBinding (deploy/manifests.yaml)
[ ] Deploy the lane-scheduler controller (single replica)
[ ] Verify startup logs show discovered GPU classes and loaded course count
[ ] Confirm /api/snapshot returns data within two cycle intervals
```

---

## 2. Cluster-Level Labels and Taints

### 2.1 GPU Class Label (Required on Every GPU Node)

The controller discovers lanes dynamically at startup by reading node labels. Every GPU node must be labelled with its hardware class:

| Field | Default value | Env var override |
|-------|---------------|-----------------|
| Label key | `gpu-class` | `LANE_GPU_CLASS_LABEL` |
| Label values | `xsmall` `small` `medium` `large` `xlarge` | — |

```bash
kubectl label nodes <gpu-node> gpu-class=medium
```

**How it works:** The controller's `NodeCapacityTracker` identifies a node as part of the managed pool by the presence of this label. Nodes without a `gpu-class` label are silently excluded. Lanes are assembled once at startup; adding a new GPU class to the cluster requires a **controller restart** to pick up the new label.

**Removing a node from management:**

```bash
kubectl label nodes <node-name> gpu-class-
```

### 2.2 Per-Class GPU Taint (Required for Lane Isolation)

Each GPU node must carry a matching `NoSchedule` taint. This prevents the default Kubernetes scheduler from placing a pod on a GPU node until the lane-scheduler has patched the matching toleration onto it:

```bash
kubectl taint nodes <gpu-node> gpu-class=medium:NoSchedule
```

The lane-scheduler's admission patch adds `tolerations: [{key: gpu-class, value: medium, effect: NoSchedule}]` to an admitted pod, allowing it to land on the correctly tainted node.

### 2.3 Node Eligibility Rules

A node is included in capacity calculations only when **all** of the following are true:

- Has a recognised `gpu-class` label (see §2.1)
- `status.conditions[Ready] == True`
- `spec.unschedulable != true`

Nodes failing any condition are tracked but contribute zero capacity until they recover. Nodes labelled with an unrecognised `gpu-class` value (one not present at controller startup) are excluded entirely from the pool.

---

## 3. Pod Labels

These labels are read from the pod at enqueue time (`pod_translator.py`). The `gpu-class` label and the scheduling gate are injected automatically by the mutating admission webhook. The course and batch labels should be set by the student's workload manifest.

| Label key | Default key | Env var override | Source | Values |
|-----------|-------------|-----------------|--------|--------|
| Course ID | `dsmlp/course` | `LANE_COURSE_LABEL` | Student manifest | e.g. `CSE234_SP26_A00` |
| GPU class | `gpu-class` | `LANE_GPU_CLASS_LABEL` | Injected by webhook | `xsmall` `small` `medium` `large` `xlarge` |
| Batch mode | `dsmlp/batch` | `LANE_BATCH_LABEL` | Student manifest (optional) | `"true"` |

**Course label:** Pods without it are bucketed under `__unlabelled__` and scored using a fallback weight of 1.0. They are still scheduled but receive no course-aware fairness treatment.

**GPU class label:** Injected by the mutating admission webhook alongside the scheduling gate. If the label is absent on a gated pod, or its value does not correspond to a lane discovered at startup, the pod is rejected: a Warning Event is emitted and the pod is never admitted. See [Section 10](#10-scenario-invalid-gpu-class-events).

**Batch mode label:** Any value equal to `"true"` (case-insensitive) applies the batch mode penalty to the priority score (default 0.3×). Batch jobs are treated as lower-urgency background work. See [Section 5.3](#53-age-boost).

---

## 4. Course Registry CSV

### 4.1 Format

The registrar CSV is the authoritative source of course scheduling weights.

```
LANE_COURSE_CSV=/etc/lane-scheduler/courses.csv   (default path)
```

```csv
course_id,weight
CSE101_SP26_A00,0.069
CSE150_SP26_A00,0.270
CSE234_SP26_A00,0.775
```

Headers are required; column order is flexible. Blank lines and leading/trailing whitespace are tolerated.

**`weight`** must be a positive float. It is used directly as `W` in the priority formula `P = W × Mode × Age / U`. The operator is responsible for computing meaningful values — a common starting point is `tier / sqrt(enrollment)`, which gives higher-tier and smaller courses stronger weight. Rows with a non-positive, non-numeric, or missing weight are skipped with a warning logged.

### 4.2 Unknown Course Fallback

If a pod's course label is absent from the CSV, the scheduler defaults to `weight=1.0`, logs a warning, and caches the synthetic entry so the warning appears only once per unknown course per registry load.

### 4.3 Reloading

The controller watches the CSV file for changes by comparing its modification time every `LANE_RELOAD_INTERVAL` seconds (default 30 s). A reload is triggered only when the mtime changes, so replacing the file — including a Kubernetes ConfigMap update — is picked up within one check interval. Reloads are atomic; the controller never reads a partially-updated registry.

---

## 5. Scheduling Algorithm Knobs

All knobs have both an environment variable form and a `--flag` form for the controller binary. They are read once at startup.

### 5.1 Scheduling Cycle

| Env var | CLI flag | Default | Effect |
|---------|----------|---------|--------|
| `LANE_CYCLE_INTERVAL` | `--cycle-interval` | `10.0` s | How often the scoring and dispatch loop runs |
| `LANE_DISPATCH_K` | `--dispatch-k` | `8` | Maximum pods admitted per lane per cycle |

Lowering `LANE_CYCLE_INTERVAL` improves latency for newly queued pods but increases Kubernetes API call rate. `LANE_DISPATCH_K` acts as a burst limiter; raise it if the queue drains too slowly when many nodes are idle.

### 5.2 Priority Score Formula

```
P(job, lane) = W(course) × Mode(job) × Age(job) / U(course, lane)
```

**W — Course weight** (`scheduler.py:224-230`)

The class weight `W` is a positive float read directly from the course CSV. The scheduler treats it as opaque; computation is the operator's responsibility. For a tiered academic environment a useful formula is `tier / sqrt(seats)` — this encodes both academic level and enrollment, rewarding smaller high-priority courses while still allowing large courses to compete.

**Mode** (`scheduler.py:168, 208-209`)

| Mode | Multiplier |
|------|-----------|
| Interactive | 1.0 |
| Batch | 0.3 (default, configurable as `batch_mode_penalty` in SchedulerConfig) |

**U — Utilization** (`scheduler.py:255-285`)

Rolling GPU/CPU units used by the course in the past `LANE_UTIL_WINDOW` seconds, floored at `LANE_EPSILON` to prevent division by zero. Courses that have not used the cluster recently score higher.

| Env var | CLI flag | Default | Effect |
|---------|----------|---------|--------|
| `LANE_UTIL_WINDOW` | `--util-window` | `300.0` s | Rolling window for per-course utilization tracking |
| `LANE_EPSILON` | `--epsilon` | `0.01` | Utilization floor; prevents divide-by-zero and rank inversions among idle courses |

### 5.3 Age Boost

```
Age(job) = 1 + α × log(1 + wait / t_half)
```

The logarithmic form prevents starvation while keeping priority growth bounded. A job at `t_half` has age ≈ `1 + 0.69 × α`; at `10 × t_half` it has age ≈ `1 + 2.40 × α`.

| Env var | CLI flag | Default | Effect |
|---------|----------|---------|--------|
| `LANE_ALPHA` | `--alpha` | `1.0` | Overall aging scale; raise to accelerate anti-starvation |
| `LANE_T_HALF_INTERACTIVE` | `--t-half-interactive` | `600` s (10 min) | Half-life for interactive jobs |
| `LANE_T_HALF_BATCH` | `--t-half-batch` | `7200` s (2 hr) | Half-life for batch jobs |

Setting `LANE_ALPHA=0` disables aging entirely (pure weighted fair-share with no starvation protection).

### 5.4 Within-Course Student Ordering

Within each course, the student selected as the course's representative candidate for each cycle is determined by two ordered rules:

1. **Fewest running pods in the lane** — a student who already has a running session is deferred in favour of classmates who have none.
2. **Oldest pending job** — among students tied on running-pod count, the one whose job has been waiting longest is selected (FIFO tiebreaker).

There are no tuning knobs for this ordering; it is fully determined by the current running-pod snapshot and job submit times.

### 5.5 Wait-Time Estimation

The `WaitEstimator` models pod residency as a truncated-normal distribution per lane. The Bayesian prior prevents wild estimates early in the semester when few real observations exist.

| Env var | CLI flag | Default | Effect |
|---------|----------|---------|--------|
| `LANE_PRIOR_WEIGHT` | `--prior-weight` | `10.0` | Pseudo-count for cluster-wide prior; lower = trust course data sooner |
| `LANE_EWMA_ALPHA` | `--ewma-alpha` | `0.1` | EWMA smoothing for per-class residency; higher = adapt faster to recent data |
| `LANE_INTERACTIVE_MEAN_PCT` | `--interactive-mean-pct` | `0.4` | Prior mean residency as fraction of `activeDeadlineSeconds` |
| `LANE_INTERACTIVE_STD_PCT` | `--interactive-std-pct` | `0.2` | Prior std deviation for interactive pods |
| `LANE_BATCH_MEAN_PCT` | `--batch-mean-pct` | `0.7` | Prior mean residency for batch pods |
| `LANE_BATCH_STD_PCT` | `--batch-std-pct` | `0.15` | Prior std deviation for batch pods |
| `LANE_WAIT_CACHE_INTERVAL` | `--wait-cache-interval` | `60.0` s | How often wait estimates are recomputed and cached |

---

## 6. Operational Flags

### 6.1 Log Level

```
LANE_LOG_LEVEL=INFO   (default; options: DEBUG INFO WARNING ERROR)
```

Set to `DEBUG` to log per-pod scoring details, per-cycle capacity summaries, and API call traces. Not recommended in production due to volume.

### 6.2 Kubeconfig

```
KUBECONFIG=<path>   (omit for in-cluster service-account credentials)
```

### 6.3 Web Dashboard Port

```
LANE_WEB_PORT=8080   (default; set to 0 to disable)
```

Serves a live HTML dashboard at `/` and a JSON API at `/api/snapshot`. See [Section 8.3](#83-web-dashboard-and-json-api).

### 6.4 CSV Check Interval

```
LANE_RELOAD_INTERVAL=30   (default: 30 seconds)
```

How often the controller stats the course CSV to detect changes. The file is only re-parsed when its mtime has changed, so a short interval is inexpensive. Increase it on high-iops-sensitive storage.

---

## 7. Kubernetes Deployment and RBAC

### 7.1 Required Permissions

The controller's ServiceAccount needs a ClusterRole with these rules (see `deploy/manifests.yaml`):

| Resource | Verbs | Purpose |
|----------|-------|---------|
| `pods` | `get list watch` | Bootstrap pending queue; stream lifecycle events |
| `pods` | `patch` | Add nodeSelector + gpu-class toleration; remove scheduling gate |
| `nodes` | `get list watch` | Discover lanes and track allocatable capacity |
| `events` | `create` | Publish queue position and wait estimates to students |

### 7.2 Single-Replica Requirement

The controller maintains in-memory state (utilization windows, residency statistics, running-pod snapshots). **Do not run more than one replica.** Horizontal scaling is not supported; use a PodDisruptionBudget with `minAvailable: 1` to avoid eviction during node maintenance.

### 7.3 Namespace Scope

The controller watches pods and nodes cluster-wide. It should be deployed in its own namespace (e.g. `lane-scheduler`) with the service account bound via a ClusterRoleBinding.

---

## 8. Observability

### 8.1 Structured Log Messages

All logs go to stdout. Key operational messages:

**Startup**

```
Discovered GPU classes from node labels: {'medium', 'large'}
Loaded 23 courses from /etc/lane-scheduler/courses.csv
Lane enum initialised: [cpu, gpu-medium, gpu-large]
```

**Per-Cycle**

```
Cycle dispatching 5 jobs
Dispatched job <uid> [lane=gpu-medium mode=interactive score=2.4312 course=CSE234 student=jdoe wait=47.2s]
```

**Pod Lifecycle**

```
Enqueued pod default/jupyter-abc123 [course=CSE234_SP26_A00 lane=gpu-medium]
Admitted pod default/jupyter-abc123 — gpu-class toleration added, scheduling gate removed [course=CSE234_SP26_A00 lane=gpu-medium wait=52.1s]
Completion recorded [course=CSE234_SP26_A00 lane=gpu-medium batch=False residency=0.823 n=41]
```

**Wait Cache**

```
WaitTimeCache: 47 estimates refreshed in 0.15s
Published 12 queue position event(s)
```

**CSV Reload**

```
CSV reloaded (25 courses); mtime changed
CSV reload failed: [Errno 2] No such file or directory: '/etc/lane-scheduler/courses.csv'
```

### 8.2 Kubernetes Events

Events appear in `kubectl describe pod <pod>` and `kubectl get events`.

**Queue position event (Normal)**

Emitted by the wait-time cache loop (default: every 60 s). Reason: `SchedulingQueued`.

```
Normal  SchedulingQueued  lane-scheduler
  Queue position: #3 overall in gpu-medium lane, #1 within course CSE234_SP26_A00.
  Estimated wait: 2m 10s (optimistic 45s, pessimistic 5m 20s).
```

Emission cadence:
- **Interactive pods:** every 60 s for the first 5 minutes, then every 5 minutes.
- **Batch pods:** at enqueue, then at 25 % / 50 % / 75 % of the first median wait estimate; no further emissions after that.

**Invalid GPU class warning (Warning)**

Reason: `UnknownGpuClass`. Emitted once per pod when its `gpu-class` label is absent or does not correspond to any lane the controller discovered at startup. Suppressible via `--no-unknown-gpu-class-events` (see [Section 10](#10-scenario-invalid-gpu-class-events)).

### 8.3 Web Dashboard and JSON API

Port configurable via `LANE_WEB_PORT` (default 8080).

```bash
kubectl port-forward -n lane-scheduler svc/lane-scheduler 8080:8080
```

**`GET /api/snapshot`** returns a JSON document with three top-level keys:

```json
{
  "generated_at": "...",
  "system": {
    "cycle_interval_s": 10,
    "wait_cache_age_s": 5.2,
    "wait_cache_duration_s": 0.15,
    "course_count": 23,
    "pending_count": 47
  },
  "lanes": [
    {
      "name": "gpu-medium",
      "node_count": 4,
      "capacity_units": 16.0,
      "running_count": 9,
      "running_units": 9.0,
      "queued_count": 12,
      "drain_p80_s": 380.2
    }
  ],
  "courses": [...]
}
```

**Health indicators in `system`:**

| Field | Healthy | Warning | Stale |
|-------|---------|---------|-------|
| `wait_cache_age_s` | < 90 s | 90 – 180 s | > 180 s |
| `wait_cache_duration_s` | < `cycle_interval_s` | — | > `cycle_interval_s` |
| `wait_cache_age_s` = `null` | — | — | Cache never populated |

The dashboard renders a colour-coded health strip based on these thresholds.

---

## 9. Scenario: Dry-Run Mode

### When to use it

- **Before the first production deployment** — verify the controller discovers the correct lanes, parses the CSV, and scores jobs as expected without touching any running workloads.
- **After tuning a knob** (e.g. changing `LANE_ALPHA` or `LANE_DISPATCH_K`) — confirm that the admission order matches intuition before committing to the change.
- **Testing the mutating webhook integration** — confirm that gated pods are detected, lanes are correctly assigned, and the three-part admission patch would be applied in the right order before enabling real patching.

### How to enable

Set the environment variable or pass the CLI flag:

```bash
LANE_DRY_RUN=true
# or
./lane-scheduler --dry-run
```

### What changes

| Action | Normal mode | Dry-run mode |
|--------|-------------|--------------|
| Pod toleration patch | Applied via Kubernetes API | Logged only |
| Kubernetes Events | Created | Logged only |
| Scoring and queueing | Runs normally | Runs normally |
| Utilization tracking | Updated | Updated |
| Web dashboard | Live data | Live data |

Log lines emitted in dry-run mode:

```
DRY RUN: would patch pod default/jupyter-abc123 to add gpu-class toleration and remove scheduling gate [course=CSE234_SP26_A00 lane=gpu-medium wait=47.1s]
DRY RUN: would create event for pod default/jupyter-abc123 (reason=SchedulingQueued)
```

Startup prints a prominent notice:

```
*** DRY RUN MODE — no pods will be patched, no events will be created ***
```

### Caveats

Because pods are never actually admitted, the queue will grow without bound during a dry run. Scoring, utilization, and running-pod counts will diverge from what would happen in production over time. Dry-run sessions are best kept short (a few minutes, covering a handful of cycles) unless the intent is queue analysis rather than admission verification.

---

## 10. Scenario: Invalid GPU Class Events

### Background

The controller only manages pods that arrive with the `lane-scheduler` scheduling gate **and** a recognised `gpu-class` label. When either condition fails, the pod is rejected immediately with a `Warning` Kubernetes event (reason `UnknownGpuClass`) and never admitted. Two distinct cases trigger this:

**Case A — missing `gpu-class` label.** A gated pod has no `gpu-class` label, which should not happen if the mutating admission webhook is functioning correctly.

```
Warning  UnknownGpuClass  lane-scheduler
  Pod has scheduling gate but no gpu-class label — cannot route to any lane.
```

**Case B — unrecognised `gpu-class` value.** The pod's `gpu-class` label value does not correspond to any GPU class the controller discovered at startup.

```
Warning  UnknownGpuClass  lane-scheduler
  gpu-class 'xlarge' is not managed by this lane scheduler and will be ignored.
  If this class should be gated, restart the controller after adding the node label.
```

In both cases the pod is ignored by this controller and will remain SchedulingGated indefinitely (it will never run) unless corrected.

### When the warning becomes noisy

1. **Multiple schedulers coexist.** A second scheduler handles a different set of GPU classes; pods destined for it will appear to have an unrecognised `gpu-class` from this controller's perspective.

2. **Gradual GPU class rollout.** A new GPU class is being added but the nodes have not yet been labelled or the controller has not been restarted. Pods arriving during this window generate warnings until the controller is restarted.

3. **Intentionally ungated classes.** Some classes are deliberately left outside the scheduling gate (e.g. high-priority research nodes) and will never be managed by this controller.

### How to suppress

```bash
LANE_NO_UNKNOWN_GPU_CLASS_EVENTS=true
# or
./lane-scheduler --no-unknown-gpu-class-events
```

### What changes

| Behaviour | Default | With flag |
|-----------|---------|-----------|
| Pod is ignored (not admitted) | Yes | Yes (unchanged) |
| Warning event emitted | Yes | No |
| Error log message emitted | Yes | Yes (unchanged) |

The flag suppresses only the Kubernetes Event. The controller still logs the situation at ERROR level:

```
ERROR Ignoring pod default/jupyter-xyz — has scheduling gate but no 'gpu-class' label
ERROR Ignoring pod default/jupyter-xyz — gpu-class 'xlarge' is not managed by this controller
```

### Recommended practice

Keep the flag **off** (default). Case A (missing label) always indicates a webhook misconfiguration and should surface as a visible event. Case B warnings are appropriate for single-scheduler clusters; suppress them only when GPU classes are intentionally partitioned across multiple schedulers.

---

## 11. Tuning Reference

Full table of all environment variables:

| Env var | CLI flag | Default | Unit | Notes |
|---------|----------|---------|------|-------|
| `LANE_WEB_PORT` | `--web-port` | `8080` | port | Set to `0` to disable web server |
| `LANE_CYCLE_INTERVAL` | `--cycle-interval` | `10.0` | s | Scheduling cadence |
| `LANE_DISPATCH_K` | `--dispatch-k` | `8` | jobs | Max admitted per lane per cycle |
| `LANE_ALPHA` | `--alpha` | `1.0` | — | Aging urgency scale |
| `LANE_T_HALF_INTERACTIVE` | `--t-half-interactive` | `600` | s | Interactive aging half-life |
| `LANE_T_HALF_BATCH` | `--t-half-batch` | `7200` | s | Batch aging half-life |
| `LANE_EPSILON` | `--epsilon` | `0.01` | — | Utilization floor |
| `LANE_UTIL_WINDOW` | `--util-window` | `300.0` | s | Rolling utilization window |
| `LANE_COURSE_CSV` | `--course-csv` | `/etc/lane-scheduler/courses.csv` | path | Registrar CSV |
| `LANE_RELOAD_INTERVAL` | `--reload-interval` | `30` | s | How often to check the course CSV for changes |
| `LANE_INTERACTIVE_MEAN_PCT` | `--interactive-mean-pct` | `0.4` | fraction | Prior mean residency (interactive) |
| `LANE_INTERACTIVE_STD_PCT` | `--interactive-std-pct` | `0.2` | fraction | Prior std (interactive) |
| `LANE_BATCH_MEAN_PCT` | `--batch-mean-pct` | `0.7` | fraction | Prior mean residency (batch) |
| `LANE_BATCH_STD_PCT` | `--batch-std-pct` | `0.15` | fraction | Prior std (batch) |
| `LANE_WAIT_CACHE_INTERVAL` | `--wait-cache-interval` | `60.0` | s | Wait estimate refresh cadence |
| `LANE_PRIOR_WEIGHT` | `--prior-weight` | `10.0` | pseudo-count | Bayesian shrinkage toward cluster prior |
| `LANE_EWMA_ALPHA` | `--ewma-alpha` | `0.1` | (0,1) | EWMA smoothing for residency; higher = faster |
| `LANE_COURSE_LABEL` | `--course-label` | `dsmlp/course` | label key | Pod label carrying course ID |
| `LANE_BATCH_LABEL` | `--batch-label` | `dsmlp/batch` | label key | Pod label for batch mode flag |
| `LANE_GPU_CLASS_LABEL` | `--gpu-class-label` | `gpu-class` | label key | Label key on pods and nodes identifying GPU class |
| `LANE_SCHEDULING_GATE_NAME` | `--scheduling-gate-name` | `lane-scheduler` | gate name | Scheduling gate injected by the mutating webhook |
| `KUBECONFIG` | `--kubeconfig` | — | path | Omit for in-cluster credentials |
| `LANE_LOG_LEVEL` | `--log-level` | `INFO` | level | DEBUG INFO WARNING ERROR |
| `LANE_DRY_RUN` | `--dry-run` | `""` (off) | bool | Log-only mode; no patches or events |
| `LANE_NO_UNKNOWN_GPU_CLASS_EVENTS` | `--no-unknown-gpu-class-events` | `""` (off) | bool | Suppress UnknownGpuClass Warning events |

---

## 12. Troubleshooting

**Pods remain SchedulingGated indefinitely**

1. Confirm the pod has the `lane-scheduler` scheduling gate (`kubectl get pod <pod> -o jsonpath='{.spec.schedulingGates}'`). If absent, the mutating webhook did not fire — check webhook logs and configuration.
2. Check the pod has the `gpu-class` label and its value matches a GPU class discovered at startup (`kubectl describe pod <pod> | grep gpu-class`).
3. Confirm the controller is running and the cycle thread is active (`kubectl logs -n lane-scheduler <pod>`).
4. Check `/api/snapshot` — if `pending_count > 0` and `running_units < capacity_units`, a scoring or dispatch issue is likely.

**GPU class Warning events flooding the event stream**

Enable `LANE_NO_UNKNOWN_GPU_CLASS_EVENTS` if the class is intentionally unmanaged. If the class *should* be managed, label and taint the nodes (`kubectl label nodes <node> gpu-class=<class> && kubectl taint nodes <node> gpu-class=<class>:NoSchedule`) and restart the controller.

**Wait estimates are stale or null**

`wait_cache_age_s` in `/api/snapshot` will exceed 90 s (or be null). Check logs for `WaitTimeCache` refresh failures and verify `LANE_WAIT_CACHE_INTERVAL` is not set extremely high.

**Uneven fairness across courses**

Verify that all active courses appear in the CSV with meaningful weights. Pods from courses absent from the CSV fall back to `weight=1.0`, which may over- or under-weight them relative to the rest of the registry and distort fairness. Set `LANE_LOG_LEVEL=DEBUG` and watch cycle log output to see per-job scores and which student is selected as each course's candidate.

**High Kubernetes API error rate**

The pod-watch and node-watch threads reconnect automatically on API errors with a 5-second back-off. Temporary errors are normal during node maintenance. Sustained errors indicate RBAC misconfiguration or API server availability issues. Check `kubectl auth can-i patch pods --as system:serviceaccount:lane-scheduler:lane-scheduler`.

**Controller OOMKilled**

In-memory state grows with queue depth and number of courses. Increase the Deployment's memory limit. The residency stats (`ResidencyStats`) and utilization windows are the primary consumers and are bounded by the number of (course, lane) pairs.
