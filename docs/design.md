

## 

Lane Scheduler is a Kubernetes controller that implements weighted fair-share scheduling for a university GPU teaching cluster. 

It intercepts pods in a SchedulingGated holding state, and once resources become available, Lane Scheduler admits pending pods in weighted fair-share priority order.  Once this controller removes a pod's schedulingGate, the ordinary Kubernetes scheduler takes over and assigns the pod to a node.

Core concepts:

* Not all cluster nodes are in-scope for Lane Scheduler; only those marked with a defined label key (default "gpu-class") designating the node's lane.  
* Lanes are defined by these node label values, e.g. "gpu-class=medium" or "gpu-class=large" which reflect different hardware capabilities (e.g. larger or smaller GPU models).  
* SchedulingGated pods will arrive labelled with their target lane (e.g. "gpu-class=medium"); a scheduling group, representing the course context of a student's work (default label "dsmlp/course=CSE123\_FA26\_A00");  and the workload within that group, reflecting the student's username (default label "dsmlp/user=jrodriguez").  
* The scheduler will track utilization of "nvidia.com/gpu" resources on target nodes & the corresponding requests from incoming pods.  
* At startup and during operation, the controller may encounter existing pods in a Running state already placed on target nodes and consuming target "nvidia.com/gpu" resources.   Controller should consider these GPU resources occupied within the node's lane, regardless of whether pods bear lane markings ("gpu-class=medium")  
* Similarly, the controller may encounter Pending pods bearing lane markings which are active within the Kubernetes scheduler, but not yet placed onto a specific node.   This may occur if pods are injected directly into Kubernetes bypassing our controller; or when a node becomes unavailable prior to a released job reaching Running state.  The resources requested by these Pending pods should be counted against the total capacity of that lane.  
* Some pods may bear labels indicating batch jobs ("dsmlp/batch=true"); these should be weighted lower (see below).

---

### 1\. Core Scheduler (`core/scheduler.py`)

Function: Pure scheduling logic — no Kubernetes dependencies. Maintains per-lane priority queues and selects which jobs to dispatch each cycle.  
Implementation approach:  
The scheduling domain is built around three layers:

* Lane model — dynamically built as a frozenset of strings ("gpu-small", "gpu-medium", etc.) by calling initialise\_lanes() at startup. No code changes are needed when new GPU classes appear; only a controller restart.  
* Priority formula — P(j,l) \= W(scheduling group) × Mode(j) × Age(j) / U(scheduling group, lane) where age is logarithmic (1 \+ α × log(1 \+ wait/t\_half)), preventing starvation without letting old jobs dominate, and batch jobs carry a 0.3 mode penalty so interactive work is preferred during the day but batch drains overnight.  
* Scheduler.cycle() — for each lane, picks *a single*  top pod  candidate per scheduling group (username with fewest running GPUs in that lane, possibly zero; then oldest submit time as tiebreaker), scores all candidates, and dispatches up to min(K, free) candidates, where free is the capacity remaining after running, kubernetes-pending, and admitted reservations are subtracted.   
* Queue structure is {lane: {sched\_group: {username: \[Job, ...\]}}}, so within-group per-user fairness is intrinsic to the data layout.  
* Scheduler will store past snapshot data in a rolling-window log of (lane, scheduling\_group) utilization to ensure fair behavior over LANE\_UTIL\_WINDOW (default 300 seconds).  U(c,l) is computed as the mean GPU-units consumed by scheduling group c in lane l over all cycle snapshots within the past LANE\_UTIL\_WINDOW seconds.

Key interactions: The controller calls update\_running\_counts() before each cycle to supply the per-user running-pod snapshot needed for the fairness rule. set\_lane\_capacity() is pushed from NodeCapacityTracker after every node event.  
---

### 2\. Scheduling Group Registry (`core/sched_group_registry.py`)

Deferred \- eventually will track scheduling groups' weights, initial residency estimation parameters, etc.  

For now, assume a default weight of 1 \= fair share between all scheduling groups, and system-default estimation parameters.

---

### 3\. Node Capacity Tracker (`core/node_capacity.py`)

Function: Maintains a live map of {lane: total\_GPU\_count} by tracking node ADDED/MODIFIED/DELETED events. 

Implementation approach:

Each node is represented as a NodeInfo dataclass. Nodes without a gpu-class label are silently ignored (they're not in the managed pool). Nodes with an unknown gpu-class are excluded and logged as warnings. Only ready=True and schedulable=True nodes contribute to lane\_capacity(). State is protected by an RLock; the entire \_nodes dict is updated on each event, making queries consistent at any point.  
Key interactions: The node-watch thread calls upsert() / remove(), then \_sync\_lane\_capacity() pushes the result into Scheduler.set\_lane\_capacity(). The snapshot builder also reads node state for the dashboard.  
---

### 4\. Controller (`k8s/controller.py`)

Function: Thread orchestration hub. Bridges Kubernetes API events to the pure scheduler core, admits pods by patching them, and coordinates all background threads.  
Implementation approach:

Three daemon threads:

| Thread | Role |
| :---- | :---- |
| pod-watch | Streams all pod events; calls \_enqueue / \_dequeue / \_upsert\_kubernetes\_pending / \_upsert\_kubernetes\_running / \_record\_completion |
| node-watch | Streams node events; updates NodeCapacityTracker; syncs lane total GPU capacity into Scheduler |
| cycle (10 s) | Calls scheduler.cycle(), applies a capacity gate, then \_admit\_pod() for each dispatched job |
|  |  |
|  |  |

The pod-watch and cycle thread share multiple dicts under separate threading.Lock objects:

* \_pending — pods waiting in Lane Scheduler queue  
* \_admitted — pods already released by Lane Scheduler but not yet `Pending`  
* \_kubernetes\_pending / \_kubernetes\_pending\_user — `Pending` pods per lane, and with user attribution  
* \_kubernetes\_running / \_kubernetes\_running\_user — `Running` pods per lane, and with user attribution  
* \_running\_ctx — completion context (sched\_group, lane, batch, deadline) needed to record residency on pod finish

When encountering a Pod which passes needs\_scheduling() \[has correct schedulingGates\[\] value\] but an unknown or absent gpu-class= lane marker: emit an error and then ignore the pod; such pods will remain SchedulingGated for human intervention.   Ignored pods associated with a subsequently deployed gpu-class will be handled normally when the controller is restarted to recognize the new gpu-class.

Admission flow: cycle() returns dispatched Job objects → capacity gate checks free \= capacity \- running \- pending \- admitted  → speculative reservation in \_admitted\_resources → patch\_namespaced\_pod() to add nodeSelector, GPU-class toleration, and remove the scheduling gate → move UID to \_admitted. Failed patches use exponential backoff up to 5 attempts before giving up \- controller should emit an error and then ignore the pods; these unpatched pods will remain SchedulingGated for human intervention.

Once a pod UID is seen as Pending within kubernetes, remove it from \_admitted and add to \_kubernetes\_pending.  Once a pod UID is seen as Running within kubernetes, remove it from \_kubernetes\_pending and add to \_kubernetes\_running.  

Watch resilience: Both watch loops track resourceVersion and resume from it on reconnect. A HTTP 410 ("Gone") forces a full relist. Pod bootstrap waits up to 10 s for node bootstrap to complete so that running pods can be attributed to lanes correctly.

Periodically, the cycle thread should validate \_admitted entries older than 10 minutes against pods currently in the cluster \- if absent, remove the \_admitted entry to recover reserved capacity.

---

### 5\. Pod Translator (`k8s/pod_translator.py`)

Function: Stateless converter between Kubernetes pod dicts and scheduler domain objects. Also constructs the admission patch payload.  
Implementation approach:  
Three pure functions do the core work:

* pod\_to\_job() — reads dsmlp/course, gpu-class, dsmlp/batch labels and nvidia.com/gpu resource requests from the pod dict, returns a Job with us \= dsmlp/user   
* needs\_scheduling() — returns True only if the pod is SchedulingGated, and still carries the lane-scheduler scheduling gate.  
* admission\_patch() — builds a strategic-merge patch body that adds nodeSelector, a gpu-class:NoSchedule toleration, and removes the scheduling gate using $patch: delete (preserving any other gates). Idempotent: returns {} if the gate is already absent.

Label keys (LANE\_COURSE\_LABEL, LANE\_GPU\_CLASS\_LABEL, etc.) are module-level constants from env vars, overridden at startup by main() if CLI flags are provided.  
---

### 6\. Wait Estimator (`estimation/wait_estimator.py`)

Function: Estimates how long a queued pod (SchedulingGate still in place) will wait before being dispatched — with P20/P50/P80 uncertainty bounds — using a probabilistic model of running pod residencies.

Implementation approach:  
The core insight: each running pod is a Bernoulli trial with probability p(t) of finishing within the next t seconds, modeled as a left-truncated normal. The sum across all running pods gives E\[slots freed by t\]. estimate\_wait() inverts this function using bisection (48 iterations, converging to sub-millisecond resolution over a 24-hour ceiling) to find the t at which enough slots are expected to free up for the queued pod's rank.  
Uncertainty bounds are computed by solving the same bisection with z\_adjust \= ±0.8416 (normal quantiles for P20/P80), treating the slot count as approximately normal via the variance sum of independent Bernoullis.  
The free\_units parameter accounts for currently-free capacity so rank-1 jobs at the front of a non-full lane return t=0 immediately.  
Key interactions: WaitTimeCache calls controller.\_build\_wait\_snapshot() on its background thread every 60 s, which calls estimate\_wait() for each pending pod and then calls event\_publisher.publish\_due() to fire Kubernetes events. The cache is a read-only dict swap under an RLock, so cache.get() is never-blocking.  
---

### 7\. Residency Stats (`estimation/residency_stats.py`)

Function: Maintains per-(sched\_group, lane, batch) residency distributions that feed the wait estimator with progressively more accurate sched\_group-specific priors.  
Implementation approach:  
Each stratum maintains three accumulators: a raw observation count n, an EWMA of residency fraction (decay factor λ, distinct from the age-scaling α), and an EWMA of squared residency fraction, from which variance is derived as E\[X²\] − E\[X\]². The posterior blends prior and observations via pseudo-count prior\_weight (default 10), using n as the blending weight: `posterior_mean = (prior_weight × prior_mean + n × ewma_mean) / (prior_weight + n)`, with posterior variance computed analogously. A sched\_group with mean residency far from the prior therefore receives wider uncertainty bounds in proportion to the EWMA variance. With few observations the estimate stays close to the cluster prior; after approximately 30 completions the sched\_group-specific EWMA dominates.  
record() is called from \_record\_completion() in the controller when a pod transitions to Succeeded/Failed, computing residency as (finishAt − startTime) / activeDeadlineSeconds.  
profile\_for() is called from \_build\_wait\_snapshot() per pending pod to get the sched\_group-specific posterior ResidencyProfile, enabling the wait estimator to be more accurate for sched\_group with many observations.

---

### 9\. Event Publisher (`k8s/event_publisher.py`)

Function: Publishes Kubernetes Event objects on queued pods  (SchedulingGate still in place)  describing queue position and wait estimate, so students can see status via kubectl describe pod.  
Implementation approach:  
Each queued pod gets a PodEventSchedule tracking emit\_count and next\_emit\_at. Two distinct emission schedules prevent event spam:

* Interactive: immediate first event, then every 60 s for 5 minutes, then every 5 minutes.  
* Batch: immediate first event, then at 25%/50%/75% of the initial median wait estimate (milestones anchored once on first observation; re-anchored upward if the wait grows 2×, never shrunk); then every 6 hours until scheduled.

Each event is a fresh create (not patch) with a unique name ({pod-name}.queue.{emit\_count:04d}), preventing Kubernetes event deduplication from collapsing distinct entries. Publication is best-effort — errors are logged and skipped, never blocking the scheduling cycle.  
publish\_due() is called from WaitTimeCache.\_run\_snapshot(), meaning event publication runs on the wait-cache background thread (every 60 s), not the cycle thread.  
---

### 10\. Web Dashboard (`web/server.py`, `web/snapshot.py`)

Function: Optional HTTP server (default port 8080\) serving a self-contained HTML dashboard and a JSON API for real-time queue state.  
Implementation approach:  
The server runs in a web-dashboard daemon thread using Python's stdlib HTTPServer. Two endpoints:

* GET / — serves a 300-line self-contained HTML/CSS/JS dashboard (no external CDN) that auto-polls /api/snapshot every 10 s.  
* GET /api/snapshot — calls build\_snapshot(controller) which reads controller state under the existing locks (read-only) and returns lane utilization, per-sched\_group queue depths with wait estimates, running pod detail, and system health metadata.

build\_snapshot() intentionally omits username identifiers. The snapshot re-calls estimate\_wait() directly for tail-queue estimates (the last job in each sched\_group's sub-queue), while the top-candidate wait uses the already-computed WaitTimeCache values.  
---

### Subsystem interaction map

Kubernetes API  
   │  pod/node events (watch.stream)  
   ▼  
Controller ──────────────────────────────────────────┐  
 │ enqueue/dequeue                                   │ patch\_namespaced\_pod (admit)  
 ▼                                                   │ create\_namespaced\_event  
Scheduler.cycle()                                     │  
 │ uses:                                             │  
 ├─ SchedGroupRegistry.get()                          │  
 ├─ NodeCapacityTracker    ← node-watch thread        │  
 └─ PriorityScorer (W×Mode×Age/U)                    │  
                                                     │  
 Controller.\_build\_wait\_snapshot() ──► EventPublisher ┘  
   │ called by:  
   ▼  
 WaitTimeCache (background, 60 s)  
   │ uses:  
   ├─ WaitEstimator.estimate\_wait()  
       └─ ResidencyStats.profile\_for()  
            └─ Bayesian blend of prior \+ EWMA  
     
                                                      
 Web dashboard (port 8080\)  
   └─ build\_snapshot() reads controller state read-only  
