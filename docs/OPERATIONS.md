# Lane Scheduler Operations Guide

This guide is for cluster administrators responsible for deploying and operating the lane-scheduler on a university GPU teaching cluster.

---

## Table of Contents

1. [Quick-Start Checklist](#1-quick-start-checklist)
2. [Cluster-Level Labels and Taints](#2-cluster-level-labels-and-taints)
3. [Pod Labels](#3-pod-labels)
4. [Scheduling Algorithm Knobs](#4-scheduling-algorithm-knobs)
5. [Operational Flags](#5-operational-flags)
6. [Kubernetes Deployment and RBAC](#6-kubernetes-deployment-and-rbac)
7. [Observability](#7-observability)
8. [Scenario: Dry-Run Mode](#8-scenario-dry-run-mode)
9. [Scenario: Invalid GPU Class Events](#9-scenario-invalid-gpu-class-events)
10. [Tuning Reference](#10-tuning-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Quick-Start Checklist

```
[ ] Label each GPU node with its gpu-class
[ ] Apply per-class gpu-class=<class>:NoSchedule taints on GPU nodes
[ ] Deploy the mutating admission controller that injects schedulingGates, gpu-class, and dsmlp/user labels
[ ] Create the ServiceAccount, ClusterRole, and ClusterRoleBinding (deploy/manifests.yaml)
[ ] Deploy the lane-scheduler controller (single replica)
[ ] Verify startup logs show discovered GPU classes
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

Nodes failing any condition are tracked but contribute zero capacity until they recover.

---

## 3. Pod Labels

These labels are read from the pod at enqueue time (`pod_translator.py`). The `gpu-class` label, the scheduling gate, and the `dsmlp/user` label are injected automatically by the mutating admission webhook. The scheduling group and batch labels should be set by the student's workload manifest.

| Label key | Default key | Env var override | Source | Values |
|-----------|-------------|-----------------|--------|--------|
| Scheduling group | `dsmlp/course` | `LANE_COURSE_LABEL` | Student manifest | e.g. `CSE234_SP26_A00` |
| User (fairness entity) | `dsmlp/user` | `LANE_USER_LABEL` | Injected by webhook | e.g. `jrodriguez` |
| GPU class | `gpu-class` | `LANE_GPU_CLASS_LABEL` | Injected by webhook | `xsmall` `small` `medium` `large` `xlarge` |
| Batch mode | `dsmlp/batch` | `LANE_BATCH_LABEL` | Student manifest (optional) | `"true"` |

**Scheduling group label:** Pods without it are bucketed under `__unlabelled__` and scored using weight 1.0 (the default for all groups). They are still scheduled but receive no group-aware fairness treatment.

**User label:** The `dsmlp/user` label identifies the individual user for per-user fairness within a scheduling group. If absent, the pod's namespace is used as the fallback username.

**GPU class label:** Injected by the mutating admission webhook alongside the scheduling gate. If the label is absent on a gated pod, or its value does not correspond to a lane discovered at startup, the pod is rejected: a Warning Event is emitted and the pod is never admitted. See [Section 9](#9-scenario-invalid-gpu-class-events).

**Batch mode label:** Any value equal to `"true"` (case-insensitive) applies the batch mode penalty to the priority score (default 0.3×). Batch jobs are treated as lower-urgency background work. See [Section 4.3](#43-age-boost).

---

## 4. Scheduling Algorithm Knobs

All knobs have both an environment variable form and a `--flag` form for the controller binary. They are read once at startup.

### 4.1 Scheduling Cycle

| Env var | CLI flag | Default | Effect |
|---------|----------|---------|--------|
| `LANE_CYCLE_INTERVAL` | `--cycle-interval` | `10.0` s | How often the scoring and dispatch loop runs |
| `LANE_DISPATCH_K` | `--dispatch-k` | `8` | Maximum pods admitted per lane per cycle |

Lowering `LANE_CYCLE_INTERVAL` improves latency for newly queued pods but increases Kubernetes API call rate. `LANE_DISPATCH_K` acts as a burst limiter; raise it if the queue drains too slowly when many nodes are idle.

### 4.2 Priority Score Formula

```
P(job, lane) = W(g) × Mode(job) × Age(job) / U(g, lane)
```

**W — Scheduling Group Weight**

All scheduling groups default to W = 1.0. The `SchedGroupRegistry` is a stub; per-group weight configuration is planned for a future release.

**Mode**

| Mode | Multiplier |
|------|-----------|
| Interactive | 1.0 |
| Batch | 0.3 (default, configurable as `batch_mode_penalty` in SchedulerConfig) |

**U — Utilization**

Rolling GPU/CPU units used by the scheduling group in the past `LANE_UTIL_WINDOW` seconds, floored at `LANE_EPSILON` to prevent division by zero.

| Env var | CLI flag | Default | Effect |
|---------|----------|---------|--------|
| `LANE_UTIL_WINDOW` | `--util-window` | `300.0` s | Rolling window for per-group utilization tracking |
| `LANE_EPSILON` | `--epsilon` | `0.01` | Utilization floor; prevents divide-by-zero among idle groups |

### 4.3 Age Boost

```
Age(job) = 1 + α × log(1 + wait / t_half)
```

| Env var | CLI flag | Default | Effect |
|---------|----------|---------|--------|
| `LANE_ALPHA` | `--alpha` | `1.0` | Overall aging scale; raise to accelerate anti-starvation |
| `LANE_T_HALF_INTERACTIVE` | `--t-half-interactive` | `600` s (10 min) | Half-life for interactive jobs |
| `LANE_T_HALF_BATCH` | `--t-half-batch` | `7200` s (2 hr) | Half-life for batch jobs |

Setting `LANE_ALPHA=0` disables aging entirely (pure weighted fair-share with no starvation protection).

### 4.4 Within-Group User Ordering

Within each scheduling group, the user selected as the group's representative candidate for each cycle is determined by two ordered rules:

1. **Fewest running pods in the lane** — a user who already has a running session is deferred in favour of others who have none.
2. **Oldest pending job** — among users tied on running-pod count, the one whose job has been waiting longest is selected (FIFO tiebreaker).

### 4.5 Wait-Time Estimation

| Env var | CLI flag | Default | Effect |
|---------|----------|---------|--------|
| `LANE_PRIOR_WEIGHT` | `--prior-weight` | `10.0` | Pseudo-count for cluster-wide prior; lower = trust group data sooner |
| `LANE_EWMA_ALPHA` | `--ewma-alpha` | `0.1` | EWMA smoothing for per-group residency; higher = adapt faster to recent data |
| `LANE_INTERACTIVE_MEAN_PCT` | `--interactive-mean-pct` | `0.4` | Prior mean residency as fraction of `activeDeadlineSeconds` |
| `LANE_INTERACTIVE_STD_PCT` | `--interactive-std-pct` | `0.2` | Prior std deviation for interactive pods |
| `LANE_BATCH_MEAN_PCT` | `--batch-mean-pct` | `0.7` | Prior mean residency for batch pods |
| `LANE_BATCH_STD_PCT` | `--batch-std-pct` | `0.15` | Prior std deviation for batch pods |
| `LANE_WAIT_CACHE_INTERVAL` | `--wait-cache-interval` | `60.0` s | How often wait estimates are recomputed and cached |

---

## 5. Operational Flags

### 5.1 Log Level

```
LANE_LOG_LEVEL=INFO   (default; options: DEBUG INFO WARNING ERROR)
```

### 5.2 Kubeconfig

```
KUBECONFIG=<path>   (omit for in-cluster service-account credentials)
```

### 5.3 Web Dashboard Port

```
LANE_WEB_PORT=8080   (default; set to 0 to disable)
```

Serves a live HTML dashboard at `/` and a JSON API at `/api/snapshot`. See [Section 7.3](#73-web-dashboard-and-json-api).

---

## 6. Kubernetes Deployment and RBAC

### 6.1 Required Permissions

The controller's ServiceAccount needs a ClusterRole with these rules (see `deploy/manifests.yaml`):

| Resource | Verbs | Purpose |
|----------|-------|---------|
| `pods` | `get list watch` | Bootstrap pending queue; stream lifecycle events |
| `pods` | `patch` | Add nodeSelector + gpu-class toleration; remove scheduling gate |
| `nodes` | `get list watch` | Discover lanes and track allocatable capacity |
| `events` | `create` | Publish queue position and wait estimates to students |

### 6.2 Single-Replica Requirement

The controller maintains in-memory state (utilization windows, residency statistics, running-pod snapshots). **Do not run more than one replica.** Horizontal scaling is not supported.

### 6.3 Namespace Scope

The controller watches pods and nodes cluster-wide. It should be deployed in its own namespace (e.g. `lane-scheduler`) with the service account bound via a ClusterRoleBinding.

---

## 7. Observability

### 7.1 Structured Log Messages

All logs go to stdout. Key operational messages:

**Startup**

```
Discovered GPU classes from node labels: {'medium', 'large'}
Lane enum initialised: [cpu, gpu-medium, gpu-large]
```

**Per-Cycle**

```
Cycle dispatching 5 jobs
Dispatched job <uid> [lane=gpu-medium mode=interactive score=2.4312 sched_group=CSE234 user=jdoe wait=47.2s]
```

**Pod Lifecycle**

```
Enqueued pod default/jupyter-abc123 [sched_group=CSE234_SP26_A00 lane=gpu-medium user=jdoe]
Admitted pod default/jupyter-abc123 — gpu-class toleration added, scheduling gate removed [sched_group=CSE234_SP26_A00 lane=gpu-medium wait=52.1s]
Completion recorded [sched_group=CSE234_SP26_A00 lane=gpu-medium batch=False residency=0.823 n=41]
```

**Wait Cache**

```
WaitTimeCache: 47 estimates refreshed in 0.15s
Published 12 queue position event(s)
```

### 7.2 Kubernetes Events

Events appear in `kubectl describe pod <pod>` and `kubectl get events`.

**Queue position event (Normal)**

Emitted by the wait-time cache loop (default: every 60 s). Reason: `SchedulingQueued`.

```
Normal  SchedulingQueued  lane-scheduler
  Queue position: #3 overall in gpu-medium lane, #1 within scheduling group CSE234_SP26_A00.
  Estimated wait: 2m 10s (optimistic 45s, pessimistic 5m 20s).
```

Emission cadence:
- **Interactive pods:** every 60 s for the first 5 minutes, then every 5 minutes.
- **Batch pods:** at enqueue, then at 25 % / 50 % / 75 % of the first median wait estimate.

**Invalid GPU class warning (Warning)**

Reason: `UnknownGpuClass`. Emitted once per pod when its `gpu-class` label is absent or does not correspond to any lane the controller discovered at startup.

### 7.3 Web Dashboard and JSON API

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
    "sched_group_count": 23,
    "pending_count": 47
  },
  "lanes": [...],
  "sched_groups": [...]
}
```

---

## 8. Scenario: Dry-Run Mode

Set the environment variable or pass the CLI flag:

```bash
LANE_DRY_RUN=true
# or
./lane-scheduler --dry-run
```

In dry-run mode, pod admission patches and Kubernetes Events are logged but not applied. Scoring, queueing, utilization tracking, and the web dashboard all function normally.

---

## 9. Scenario: Invalid GPU Class Events

The controller only manages pods that arrive with the `lane-scheduler` scheduling gate **and** a recognised `gpu-class` label. When either condition fails, the pod is rejected immediately with a `Warning` Kubernetes event (reason `UnknownGpuClass`) and never admitted.

To suppress the Warning events (e.g. when multiple schedulers coexist):

```bash
LANE_NO_UNKNOWN_GPU_CLASS_EVENTS=true
```

The pod is still ignored and the situation is still logged at ERROR level; only the Kubernetes Event is suppressed.

---

## 10. Tuning Reference

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
| `LANE_INTERACTIVE_MEAN_PCT` | `--interactive-mean-pct` | `0.4` | fraction | Prior mean residency (interactive) |
| `LANE_INTERACTIVE_STD_PCT` | `--interactive-std-pct` | `0.2` | fraction | Prior std (interactive) |
| `LANE_BATCH_MEAN_PCT` | `--batch-mean-pct` | `0.7` | fraction | Prior mean residency (batch) |
| `LANE_BATCH_STD_PCT` | `--batch-std-pct` | `0.15` | fraction | Prior std (batch) |
| `LANE_WAIT_CACHE_INTERVAL` | `--wait-cache-interval` | `60.0` | s | Wait estimate refresh cadence |
| `LANE_PRIOR_WEIGHT` | `--prior-weight` | `10.0` | pseudo-count | Bayesian shrinkage toward cluster prior |
| `LANE_EWMA_ALPHA` | `--ewma-alpha` | `0.1` | (0,1) | EWMA smoothing for residency; higher = faster |
| `LANE_COURSE_LABEL` | `--course-label` | `dsmlp/course` | label key | Pod label carrying scheduling group ID |
| `LANE_USER_LABEL` | `--user-label` | `dsmlp/user` | label key | Pod label carrying username (falls back to namespace) |
| `LANE_BATCH_LABEL` | `--batch-label` | `dsmlp/batch` | label key | Pod label for batch mode flag |
| `LANE_GPU_CLASS_LABEL` | `--gpu-class-label` | `gpu-class` | label key | Label key on pods and nodes identifying GPU class |
| `LANE_SCHEDULING_GATE_NAME` | `--scheduling-gate-name` | `lane-scheduler` | gate name | Scheduling gate injected by the mutating webhook |
| `KUBECONFIG` | `--kubeconfig` | — | path | Omit for in-cluster credentials |
| `LANE_LOG_LEVEL` | `--log-level` | `INFO` | level | DEBUG INFO WARNING ERROR |
| `LANE_DRY_RUN` | `--dry-run` | `""` (off) | bool | Log-only mode; no patches or events |
| `LANE_NO_UNKNOWN_GPU_CLASS_EVENTS` | `--no-unknown-gpu-class-events` | `""` (off) | bool | Suppress UnknownGpuClass Warning events |

---

## 11. Troubleshooting

**Pods remain SchedulingGated indefinitely**

1. Confirm the pod has the `lane-scheduler` scheduling gate (`kubectl get pod <pod> -o jsonpath='{.spec.schedulingGates}'`). If absent, the mutating webhook did not fire.
2. Check the pod has the `gpu-class` label and its value matches a GPU class discovered at startup.
3. Confirm the controller is running and the cycle thread is active (`kubectl logs -n lane-scheduler <pod>`).
4. Check `/api/snapshot` — if `pending_count > 0` and `running_units < capacity_units`, a scoring or dispatch issue is likely.

**GPU class Warning events flooding the event stream**

Enable `LANE_NO_UNKNOWN_GPU_CLASS_EVENTS` if the class is intentionally unmanaged. If the class *should* be managed, label and taint the nodes and restart the controller.

**Wait estimates are stale or null**

`wait_cache_age_s` in `/api/snapshot` will exceed 90 s (or be null). Check logs for `WaitTimeCache` refresh failures.

**Uneven fairness across scheduling groups**

All groups currently share W = 1.0. Within-group fairness is determined by the running-pod count per user. Set `LANE_LOG_LEVEL=DEBUG` and watch cycle log output to see per-job scores and which user is selected as each group's candidate.

**High Kubernetes API error rate**

The pod-watch and node-watch threads reconnect automatically on API errors with a 5-second back-off. Sustained errors indicate RBAC misconfiguration or API server availability issues. Check `kubectl auth can-i patch pods --as system:serviceaccount:lane-scheduler:lane-scheduler`.

**Controller OOMKilled**

In-memory state grows with queue depth and number of scheduling groups. Increase the Deployment's memory limit. The residency stats and utilization windows are the primary consumers.
