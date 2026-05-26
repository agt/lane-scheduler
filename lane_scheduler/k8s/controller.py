"""
Lane-based Priority Scheduler Controller
-----------------------------------------
Watches all pod and node events cluster-wide.  Pending pods that carry the
dsmlp/course label are held by the scheduling gate until our scheduler
selects them, at which point we patch in the matching toleration and let the
default Kubernetes scheduler handle actual placement.

Entry point:
    lane-scheduler   (via pyproject.toml entry_point)
    python -m lane_scheduler.k8s.controller

Kubernetes RBAC required (see deploy/manifests.yaml):
    Pods  : get, list, watch, patch
    Nodes : get, list, watch
    Events: create
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

try:
    from kubernetes import client, config as k8s_config, watch
except ImportError:
    sys.exit("kubernetes package not found.  Run: pip install kubernetes")

from lane_scheduler.core.scheduler import (
    Job, Lane, SchedulerConfig, Scheduler,
    initialise_lanes, lane_for_gpu_class, is_known_gpu_class,
)
from lane_scheduler.core.sched_group_registry import SchedGroupRegistry
from lane_scheduler.core.node_capacity import (
    NodeCapacityTracker,
    INHIBIT_TAINT_KEY, INHIBIT_TAINT_VALUE, GPU_CLASS_LABEL_KEY,
)
from lane_scheduler.k8s.pod_translator import (
    NO_SCHED_GROUP_LABEL,
    LABEL_SCHED_GROUP,
    LABEL_USER,
    LABEL_BATCH,
    LABEL_GPU_CLASS,
    SCHEDULING_GATE_NAME,
    admission_patch,
    needs_scheduling,
    pod_to_job,
    _gpu_request_count,
)
from lane_scheduler.estimation.wait_estimator import (
    ResidencyProfile, RunningPod, WaitEstimate,
    WaitTimeCache, estimate_wait, format_wait,
)
from lane_scheduler.k8s.event_publisher import EventPublisher
from lane_scheduler.estimation.residency_stats import ResidencyStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU class discovery
# ---------------------------------------------------------------------------

def discover_gpu_classes(core_v1: "client.CoreV1Api") -> list[str]:
    """
    List all nodes and collect distinct gpu-class label values.
    Called once at startup, before initialise_lanes().
    """
    found: set[str] = set()
    try:
        nodes = core_v1.list_node().items or []
    except Exception as exc:
        logger.error("Failed to list nodes during GPU class discovery: %s", exc)
        return []

    for node in nodes:
        labels = node.metadata.labels or {}
        gpu_class = labels.get(GPU_CLASS_LABEL_KEY, "")
        if gpu_class:
            found.add(gpu_class.strip().lower())

    classes = sorted(found)
    if classes:
        logger.info("Discovered GPU classes from node labels: %s", classes)
    else:
        logger.warning(
            "No %r labels found on any node — "
            "no GPU lanes will be available until a controller restart.",
            GPU_CLASS_LABEL_KEY,
        )
    return classes


# ---------------------------------------------------------------------------
# Configuration (environment variables with sensible defaults)
# ---------------------------------------------------------------------------

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return default

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


WEB_PORT            = _env_int(  "LANE_WEB_PORT",           8080)
CYCLE_INTERVAL      = _env_float("LANE_CYCLE_INTERVAL",     10.0)
DISPATCH_K          = _env_int(  "LANE_DISPATCH_K",         8)
ALPHA               = _env_float("LANE_ALPHA",              1.0)
T_HALF_INTERACTIVE  = _env_float("LANE_T_HALF_INTERACTIVE", 600.0)
T_HALF_BATCH        = _env_float("LANE_T_HALF_BATCH",       7200.0)
UTIL_WINDOW         = _env_float("LANE_UTIL_WINDOW",        300.0)
WAIT_CACHE_INTERVAL = _env_float("LANE_WAIT_CACHE_INTERVAL",60.0)

INTERACTIVE_MEAN_PCT = _env_float("LANE_INTERACTIVE_MEAN_PCT", 0.4)
INTERACTIVE_STD_PCT  = _env_float("LANE_INTERACTIVE_STD_PCT",  0.2)
BATCH_MEAN_PCT       = _env_float("LANE_BATCH_MEAN_PCT",       0.7)
BATCH_STD_PCT        = _env_float("LANE_BATCH_STD_PCT",        0.15)
PRIOR_WEIGHT         = _env_float("LANE_PRIOR_WEIGHT",          10.0)
EWMA_ALPHA           = _env_float("LANE_EWMA_ALPHA",             0.1)
DEFAULT_ACTIVE_DEADLINE = _env_int("LANE_DEFAULT_ACTIVE_DEADLINE_SECONDS", 86400)

KUBECONFIG = os.environ.get("KUBECONFIG")
LOG_LEVEL  = os.environ.get("LANE_LOG_LEVEL", "INFO").upper()
DRY_RUN    = os.environ.get("LANE_DRY_RUN", "").lower() in ("1", "true", "yes")
NO_UNKNOWN_GPU_CLASS_EVENTS = os.environ.get(
    "LANE_NO_UNKNOWN_GPU_CLASS_EVENTS", ""
).lower() in ("1", "true", "yes")

# Admission retry / circuit breaker
_MAX_ADMIT_ATTEMPTS = 5
_MAX_ADMIT_BACKOFF  = 300.0

# TTL for completion-context entries when a pod's completion event was missed.
_RUNNING_CTX_TTL = 86400.0

# How long an admitted entry may linger before we validate it against the cluster.
_ADMITTED_STALE_TTL = 600.0  # 10 minutes


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class LaneSchedulerController:
    """
    Orchestrates the pod/node watch loops and the scheduling cycle.

    Threading model:
        - pod-watch thread  : streams pod events; maintains pending/running state
        - node-watch thread : streams node events; updates NodeCapacityTracker
        - cycle thread      : runs scheduler.cycle() every CYCLE_INTERVAL seconds
        - wait-cache thread : background WaitTimeCache refresh (60 s cadence)
    """

    def __init__(
        self,
        core_v1:             client.CoreV1Api,
        registry:            SchedGroupRegistry,
        sched_config:        SchedulerConfig,
        residency_profiles:  dict[str, ResidencyProfile],
        prior_weight:        float = PRIOR_WEIGHT,
        ewma_alpha:          float = EWMA_ALPHA,
        cycle_interval:      float = CYCLE_INTERVAL,
        wait_cache_interval: float = WAIT_CACHE_INTERVAL,
        web_port:            int   = 0,
        dry_run:             bool  = False,
        no_unknown_gpu_class_events: bool = False,
        default_active_deadline:     int  = DEFAULT_ACTIVE_DEADLINE,
    ):
        self.core_v1        = core_v1
        self.registry       = registry
        self.sched_config   = sched_config
        self.cycle_interval = cycle_interval
        self.web_port       = web_port
        self.dry_run        = dry_run
        self.no_unknown_gpu_class_events = no_unknown_gpu_class_events
        self._default_active_deadline    = default_active_deadline

        if dry_run:
            logger.warning("DRY RUN mode enabled — pods will not be patched and events will not be created")

        self.node_tracker = NodeCapacityTracker()

        from lane_scheduler.core.scheduler import Lane as _Lane
        self.scheduler = Scheduler(
            lane_capacity={lane: 0.0 for lane in (_Lane or [])},
            config=sched_config,
        )

        # Pods waiting in Lane Scheduler queue: uid → pod dict
        self._pending: dict[str, dict] = {}
        self._pending_lock = threading.Lock()

        # Pods with an unrecognised gpu-class that we are ignoring
        self._ignored_gpu_class: set[str] = set()

        # Pods we have already patched (gate removed), not yet k8s-Pending
        self._admitted: set[str] = set()
        self._admitted_lock = threading.Lock()

        # Admitted pods not yet k8s-Pending: uid → (lane, resource_units).
        # Cleared when the pod transitions to Pending phase.
        self._admitted_resources: dict[str, tuple[object, float]] = {}
        self._admitted_resources_lock = threading.Lock()

        # Timestamp of admission for staleness detection:
        # uid → (namespace, name, lane, resource_units, admitted_at_monotonic)
        self._admitted_with_timestamp: dict[str, tuple[str, str, object, float, float]] = {}
        self._admitted_timestamp_lock = threading.Lock()

        # k8s-Pending pods (gate removed, waiting for default scheduler to place):
        # {lane: {uid: RunningPod}}  — RunningPod.start_time is approximate here
        self._kubernetes_pending: dict[object, dict[str, RunningPod]] = {}
        self._kubernetes_pending_lock = threading.Lock()
        # User attribution for k8s-pending pods
        self._kubernetes_pending_user: dict[str, str] = {}

        # Running pods per lane: {lane: {uid: RunningPod}}
        self._kubernetes_running: dict[object, dict[str, RunningPod]] = {}
        # username attribution for running pods
        self._kubernetes_running_user: dict[str, str] = {}
        self._running_lock = threading.Lock()

        # Completion context: uid → (sched_group_id, lane_name, batch, deadline, ctx_created_monotonic)
        self._running_ctx: dict[str, tuple[str, str, bool, float, float]] = {}
        self._running_ctx_lock = threading.Lock()

        # Per-pod admission retry state: uid → (attempt_count, next_retry_monotonic)
        self._admit_attempts: dict[str, tuple[int, float]] = {}
        self._admit_attempts_lock = threading.Lock()

        # Last-seen resourceVersion for each watch
        self._pod_resource_version:  Optional[str] = None
        self._node_resource_version: Optional[str] = None

        # Watch handles for clean shutdown
        self._pod_watch:  Optional["watch.Watch"] = None
        self._node_watch: Optional["watch.Watch"] = None

        self._threads: list[threading.Thread] = []

        self.residency_stats = ResidencyStats(
            interactive_prior = residency_profiles["interactive"],
            batch_prior       = residency_profiles["batch"],
            prior_weight      = prior_weight,
            ewma_alpha        = ewma_alpha,
        )

        self.wait_cache = WaitTimeCache(
            snapshot_fn = self._build_wait_snapshot,
            interval    = wait_cache_interval,
        )

        self.event_publisher = EventPublisher(
            core_v1,
            dry_run=dry_run,
            no_unknown_gpu_class_events=no_unknown_gpu_class_events,
        )

        self._stop = threading.Event()
        self._nodes_bootstrapped = threading.Event()
        self._bootstrap_pod_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("Lane Scheduler starting")

        self.wait_cache.start()

        self._threads = [
            threading.Thread(target=self._pod_watch_loop,  name="pod-watch",  daemon=True),
            threading.Thread(target=self._node_watch_loop, name="node-watch", daemon=True),
            threading.Thread(target=self._cycle_loop,      name="cycle",      daemon=True),
        ]
        for t in self._threads:
            t.start()

        if self.web_port > 0:
            from lane_scheduler.web.server import start_server
            start_server(self, port=self.web_port)

        try:
            while not self._stop.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            self.wait_cache.stop()
            for t in self._threads:
                t.join(timeout=10.0)
                if t.is_alive():
                    logger.warning("Thread %s did not exit within 10s", t.name)
            logger.info("Lane Scheduler stopped")

    def stop(self) -> None:
        self._stop.set()
        for w in (self._pod_watch, self._node_watch):
            if w is not None:
                try:
                    w.stop()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Pod watch
    # ------------------------------------------------------------------

    def _pod_watch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self._pod_resource_version is None:
                    self._bootstrap_pods()

                w = watch.Watch()
                self._pod_watch = w
                logger.info(
                    "Starting pod watch (resource_version=%s)",
                    self._pod_resource_version,
                )
                stream_kwargs = {"timeout_seconds": 60, "allow_watch_bookmarks": True}
                if self._pod_resource_version is not None:
                    stream_kwargs["resource_version"] = self._pod_resource_version

                for event in w.stream(
                    self.core_v1.list_pod_for_all_namespaces,
                    **stream_kwargs,
                ):
                    if self._stop.is_set():
                        break
                    self._update_resource_version(event, is_pod=True)
                    if event.get("type") == "BOOKMARK":
                        continue
                    self._handle_pod_event(event)
            except client.exceptions.ApiException as exc:
                if exc.status == 410:
                    logger.warning(
                        "Pod watch resourceVersion %s expired (410) — relisting",
                        self._pod_resource_version,
                    )
                    self._pod_resource_version = None
                else:
                    logger.warning("Pod watch API error: %s — reconnecting in 5s", exc)
                    if not self._stop.wait(5.0):
                        pass
            except Exception as exc:
                logger.warning("Pod watch error: %s — reconnecting in 5s", exc)
                if not self._stop.wait(5.0):
                    pass
            finally:
                self._pod_watch = None

    def _bootstrap_pods(self) -> None:
        """List all pods and seed our state + resourceVersion."""
        self._nodes_bootstrapped.wait(timeout=10.0)
        resp  = self.core_v1.list_pod_for_all_namespaces()
        items = getattr(resp, "items", None) or []
        rv    = None
        meta  = getattr(resp, "metadata", None)
        if meta is not None:
            rv = getattr(meta, "resource_version", None)
        _ser  = self.core_v1.api_client.sanitize_for_serialization
        cache: dict[str, dict] = {}
        for item in items:
            pod = _ser(item) if hasattr(item, "to_dict") else item
            uid = (pod.get("metadata") or {}).get("uid")
            if uid:
                cache[uid] = pod
            self._handle_pod_event({"type": "ADDED", "object": item})
        self._bootstrap_pod_cache = cache
        self._pod_resource_version = rv
        if self._nodes_bootstrapped.is_set():
            self._rescan_unlaned_running_pods()

    def _update_resource_version(self, event: dict, *, is_pod: bool) -> None:
        obj = event.get("object")
        rv: Optional[str] = None
        if hasattr(obj, "metadata") and getattr(obj.metadata, "resource_version", None):
            rv = obj.metadata.resource_version
        elif isinstance(obj, dict):
            md = obj.get("metadata") or {}
            rv = md.get("resourceVersion") or md.get("resource_version")
        if not rv:
            return
        if is_pod:
            self._pod_resource_version = rv
        else:
            self._node_resource_version = rv

    def _handle_pod_event(self, event: dict) -> None:
        etype = event.get("type", "")
        pod   = event.get("object", {})

        if hasattr(pod, "to_dict"):
            pod = self.core_v1.api_client.sanitize_for_serialization(pod)

        uid = (pod.get("metadata") or {}).get("uid")
        if not uid:
            return

        if etype in ("ADDED", "MODIFIED"):
            if needs_scheduling(pod):
                self._enqueue(uid, pod)
            else:
                self._dequeue(uid)
                phase = (pod.get("status") or {}).get("phase", "")
                if phase == "Running":
                    self._transition_to_running(uid, pod)
                elif phase in ("Succeeded", "Failed"):
                    self._record_completion(uid, pod)
                    self._remove_running(uid)
                    self._remove_kubernetes_pending(uid)
                    with self._admitted_resources_lock:
                        self._admitted_resources.pop(uid, None)
                    with self._admitted_timestamp_lock:
                        self._admitted_with_timestamp.pop(uid, None)
                else:
                    # Admitted and now k8s-Pending (gate removed, no nodeName yet)
                    self._transition_to_kubernetes_pending(uid, pod)

        elif etype == "DELETED":
            self._dequeue(uid)
            self._remove_running(uid)
            self._remove_kubernetes_pending(uid)
            with self._admitted_resources_lock:
                self._admitted_resources.pop(uid, None)
            with self._admitted_timestamp_lock:
                self._admitted_with_timestamp.pop(uid, None)

    def _transition_to_kubernetes_pending(self, uid: str, pod: dict) -> None:
        """
        Pod has been admitted by us (gate removed) and is now Pending in k8s.
        Move it from _admitted_resources to _kubernetes_pending.
        """
        with self._admitted_lock:
            if uid not in self._admitted:
                # Pod was Pending before we knew about it (bypassed our gate)
                # — still count it against capacity.
                pass

        # Build a RunningPod-like entry for capacity accounting.
        # Use a sentinel start_time; resource_units is what matters here.
        rp = self._running_pod_from_pod(uid, pod)
        if rp is None:
            # Can't parse — use admitted_resources as fallback
            with self._admitted_resources_lock:
                entry = self._admitted_resources.get(uid)
            if entry is not None:
                lane, units = entry
                rp = RunningPod(
                    pod_uid                 = uid,
                    start_time              = time.monotonic(),
                    active_deadline_seconds = float(self._default_active_deadline),
                    batch                   = False,
                    resource_units          = units,
                )
            if rp is None:
                return

        gpu_lane = self._lane_for_pod(pod)
        if gpu_lane is None:
            return

        username = self._username_from_pod(pod)

        with self._kubernetes_pending_lock:
            if gpu_lane not in self._kubernetes_pending:
                self._kubernetes_pending[gpu_lane] = {}
            self._kubernetes_pending[gpu_lane][uid] = rp
            self._kubernetes_pending_user[uid] = username

        # Release the speculative reservation from _admitted_resources
        with self._admitted_resources_lock:
            self._admitted_resources.pop(uid, None)
        with self._admitted_timestamp_lock:
            self._admitted_with_timestamp.pop(uid, None)

    def _transition_to_running(self, uid: str, pod: dict) -> None:
        """Pod has transitioned to Running. Remove from k8s-pending, add to running."""
        with self._admitted_resources_lock:
            self._admitted_resources.pop(uid, None)
        with self._admitted_timestamp_lock:
            self._admitted_with_timestamp.pop(uid, None)
        self._remove_kubernetes_pending(uid)
        self._upsert_running(uid, pod)

    def _running_pod_from_pod(self, uid: str, pod: dict) -> Optional[RunningPod]:
        """Build a RunningPod from a Kubernetes pod dict."""
        from lane_scheduler.k8s.pod_translator import _is_batch, _resource_units, _gpu_lane
        spec   = pod.get("spec")   or {}
        status = pod.get("status") or {}

        deadline = spec.get("activeDeadlineSeconds") or self._default_active_deadline

        start_str = status.get("startTime")
        if start_str:
            try:
                import datetime
                dt      = datetime.datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                wall_age = (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds()
                start_time = time.monotonic() - max(0.0, wall_age)
            except Exception:
                start_time = time.monotonic()
        else:
            start_time = time.monotonic()

        gpu_lane = _gpu_lane(pod)
        if gpu_lane is None:
            _nname = spec.get("nodeName", "") or ""
            gpu_lane = self.node_tracker.lane_for_node(_nname) if _nname else None
        batch = _is_batch(pod)

        from lane_scheduler.k8s.pod_translator import _resource_units as _ru
        units = _ru(pod, gpu_lane)

        return RunningPod(
            pod_uid                 = uid,
            start_time              = start_time,
            active_deadline_seconds = float(deadline),
            batch                   = batch,
            resource_units          = units,
        )

    def _lane_for_pod(self, pod: dict) -> Optional[str]:
        """Return the GPU lane for a pod, trying label then node lookup."""
        from lane_scheduler.k8s.pod_translator import _gpu_lane
        gpu_lane = _gpu_lane(pod)
        if gpu_lane is None:
            nname = ((pod.get("spec") or {}).get("nodeName", "") or "")
            if nname:
                gpu_lane = self.node_tracker.lane_for_node(nname)
        return gpu_lane

    def _username_from_pod(self, pod: dict) -> str:
        """Extract the username from dsmlp/user label, fall back to namespace."""
        meta      = pod.get("metadata") or {}
        labels    = meta.get("labels") or {}
        namespace = meta.get("namespace", "")
        return labels.get(LABEL_USER, "").strip() or namespace

    def _upsert_running(self, uid: str, pod: dict) -> None:
        rp = self._running_pod_from_pod(uid, pod)
        if rp is None:
            return
        from lane_scheduler.k8s.pod_translator import _gpu_lane, _is_batch
        gpu_lane = self._lane_for_pod(pod)
        if gpu_lane is None:
            if _gpu_request_count(pod) == 0.0:
                return
            logger.info(
                "Running pod %s: no gpu-class label and node not in managed pool; "
                "skipping utilization tracking", uid,
            )
            return
        lane          = gpu_lane
        lane_name     = lane
        sched_group_id = (
            (pod.get("metadata") or {}).get("labels") or {}
        ).get(LABEL_SCHED_GROUP, NO_SCHED_GROUP_LABEL) or NO_SCHED_GROUP_LABEL
        username      = self._username_from_pod(pod)
        batch         = _is_batch(pod)
        deadline      = float(
            (pod.get("spec") or {}).get("activeDeadlineSeconds") or self._default_active_deadline
        )

        with self._running_lock:
            if lane not in self._kubernetes_running:
                self._kubernetes_running[lane] = {}
            self._kubernetes_running[lane][uid] = rp
            self._kubernetes_running_user[uid] = username

        with self._running_ctx_lock:
            self._running_ctx[uid] = (
                sched_group_id, lane_name, batch, deadline, time.monotonic(),
            )

    def _remove_running(self, uid: str) -> None:
        with self._running_lock:
            for lane_dict in self._kubernetes_running.values():
                lane_dict.pop(uid, None)
            self._kubernetes_running_user.pop(uid, None)
        with self._running_ctx_lock:
            self._running_ctx.pop(uid, None)

    def _remove_kubernetes_pending(self, uid: str) -> None:
        with self._kubernetes_pending_lock:
            for lane_dict in self._kubernetes_pending.values():
                lane_dict.pop(uid, None)
            self._kubernetes_pending_user.pop(uid, None)

    def _record_completion(self, uid: str, pod: dict) -> None:
        with self._running_ctx_lock:
            ctx = self._running_ctx.get(uid)
        if ctx is None:
            return

        sched_group_id, lane_name, batch, deadline, _ctx_created = ctx
        if deadline <= 0:
            return

        residency_pct = 1.0
        try:
            import datetime as _dt
            status     = pod.get("status") or {}
            start_str  = status.get("startTime")
            finish_str = None
            for cs in (status.get("containerStatuses") or []):
                term = (cs.get("state") or {}).get("terminated") or {}
                fat  = term.get("finishedAt")
                if fat:
                    if finish_str is None or fat > finish_str:
                        finish_str = fat

            if start_str and finish_str:
                def _parse(s: str) -> _dt.datetime:
                    return _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
                elapsed = (_parse(finish_str) - _parse(start_str)).total_seconds()
                if elapsed > 0:
                    residency_pct = min(1.0, elapsed / deadline)
        except Exception as exc:
            logger.debug("Could not compute residency for pod %s: %s", uid, exc)

        self.residency_stats.record(
            sched_group_id = sched_group_id,
            lane_name      = lane_name,
            batch          = batch,
            residency_pct  = residency_pct,
        )
        logger.info(
            "Completion recorded [group=%s lane=%s batch=%s residency=%.3f n=%d]",
            sched_group_id, lane_name, batch, residency_pct,
            self.residency_stats.observation_count(sched_group_id, lane_name, batch),
        )

    def running_pods_for_lane(self, lane: object) -> list[RunningPod]:
        """Return a snapshot of RunningPod objects for a given lane."""
        with self._running_lock:
            return list((self._kubernetes_running.get(lane) or {}).values())

    def _enqueue(self, uid: str, pod: dict) -> None:
        gpu_class = ((pod.get("metadata") or {}).get("labels") or {}).get(
            LABEL_GPU_CLASS, ""
        ).strip()
        if not gpu_class:
            if _gpu_request_count(pod) == 0.0:
                return
            if uid not in self._ignored_gpu_class:
                self._ignored_gpu_class.add(uid)
                logger.error(
                    "Ignoring pod %s/%s — has scheduling gate but no %r label",
                    (pod.get("metadata") or {}).get("namespace", "?"),
                    (pod.get("metadata") or {}).get("name", "?"),
                    LABEL_GPU_CLASS,
                )
                self.event_publisher.warn_unknown_gpu_class(uid, pod)
            return
        if not is_known_gpu_class(gpu_class):
            if uid not in self._ignored_gpu_class:
                self._ignored_gpu_class.add(uid)
                logger.error(
                    "Ignoring pod %s/%s — gpu-class '%s' is not managed by this controller",
                    (pod.get("metadata") or {}).get("namespace", "?"),
                    (pod.get("metadata") or {}).get("name", "?"),
                    gpu_class,
                )
                self.event_publisher.warn_unknown_gpu_class(uid, pod)
            return

        with self._admitted_lock:
            if uid in self._admitted:
                return

        with self._pending_lock:
            if uid in self._pending:
                return

            sched_group_id = (
                (pod.get("metadata", {}).get("labels") or {}).get(
                    LABEL_SCHED_GROUP, NO_SCHED_GROUP_LABEL
                ).strip() or NO_SCHED_GROUP_LABEL
            )
            group = self.registry.get(sched_group_id)

            if not self.scheduler.has_group(sched_group_id):
                self.scheduler.register_group(group)

            submit_time = time.monotonic()
            job = pod_to_job(pod, submit_time=submit_time)
            self.scheduler.submit(job)
            self._pending[uid] = pod
            self.event_publisher.register(uid, pod)

            logger.info(
                "Enqueued pod %s/%s [group=%s lane=%s user=%s]",
                (pod.get("metadata") or {}).get("namespace", "?"),
                (pod.get("metadata") or {}).get("name", "?"),
                sched_group_id, job.lane, job.username,
            )

    def _dequeue(self, uid: str) -> None:
        with self._pending_lock:
            was_pending = self._pending.pop(uid, None) is not None
        with self._admitted_lock:
            was_admitted = uid in self._admitted
            self._admitted.discard(uid)
        if not was_admitted:
            with self._admitted_resources_lock:
                self._admitted_resources.pop(uid, None)
            with self._admitted_timestamp_lock:
                self._admitted_with_timestamp.pop(uid, None)
        self._ignored_gpu_class.discard(uid)
        with self._admit_attempts_lock:
            self._admit_attempts.pop(uid, None)
        if was_pending:
            self.scheduler.remove_job(uid)
        self.event_publisher.deregister(uid)
        self.event_publisher.clear_unknown_gpu_warning(uid)

    # ------------------------------------------------------------------
    # Node watch
    # ------------------------------------------------------------------

    def _node_watch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self._node_resource_version is None:
                    self._bootstrap_nodes()

                w = watch.Watch()
                self._node_watch = w
                logger.info(
                    "Starting node watch (resource_version=%s)",
                    self._node_resource_version,
                )
                stream_kwargs = {"timeout_seconds": 60, "allow_watch_bookmarks": True}
                if self._node_resource_version is not None:
                    stream_kwargs["resource_version"] = self._node_resource_version

                for event in w.stream(
                    self.core_v1.list_node,
                    **stream_kwargs,
                ):
                    if self._stop.is_set():
                        break
                    self._update_resource_version(event, is_pod=False)
                    if event.get("type") == "BOOKMARK":
                        continue
                    self._handle_node_event(event)
                    self._sync_lane_capacity()
            except client.exceptions.ApiException as exc:
                if exc.status == 410:
                    logger.warning(
                        "Node watch resourceVersion %s expired (410) — relisting",
                        self._node_resource_version,
                    )
                    self._node_resource_version = None
                else:
                    logger.warning("Node watch API error: %s — reconnecting in 5s", exc)
                    if not self._stop.wait(5.0):
                        pass
            except Exception as exc:
                logger.warning("Node watch error: %s — reconnecting in 5s", exc)
                if not self._stop.wait(5.0):
                    pass
            finally:
                self._node_watch = None

    def _bootstrap_nodes(self) -> None:
        resp  = self.core_v1.list_node()
        items = getattr(resp, "items", None) or []
        rv    = None
        meta  = getattr(resp, "metadata", None)
        if meta is not None:
            rv = getattr(meta, "resource_version", None)
        for item in items:
            self._handle_node_event({"type": "ADDED", "object": item})
        self._sync_lane_capacity()
        self._node_resource_version = rv
        self._nodes_bootstrapped.set()
        self._rescan_unlaned_running_pods()

    def _rescan_unlaned_running_pods(self) -> None:
        with self._pending_lock:
            pending_uids = set(self._pending)
        with self._running_lock:
            already_tracked: set[str] = set()
            for lane_dict in self._kubernetes_running.values():
                already_tracked.update(lane_dict.keys())

        cache = self._bootstrap_pod_cache
        for uid, pod in cache.items():
            if uid in pending_uids or uid in already_tracked:
                continue
            phase = (pod.get("status") or {}).get("phase", "")
            if phase == "Running":
                self._upsert_running(uid, pod)

    def _handle_node_event(self, event: dict) -> None:
        etype = event.get("type", "")
        node  = event.get("object", {})
        if hasattr(node, "to_dict"):
            node = self.core_v1.api_client.sanitize_for_serialization(node)
        name = (node.get("metadata", {}) or {}).get("name", "")
        if etype in ("ADDED", "MODIFIED"):
            self.node_tracker.upsert(node)
        elif etype == "DELETED":
            self.node_tracker.remove(name)

    def _sync_lane_capacity(self) -> None:
        caps = self.node_tracker.lane_capacity()
        self.scheduler.set_lane_capacity(caps)

    # ------------------------------------------------------------------
    # Scheduling cycle
    # ------------------------------------------------------------------

    def _cycle_loop(self) -> None:
        self._stop.wait(max(self.cycle_interval, 5.0))

        while not self._stop.is_set():
            try:
                self._sweep_running_ctx()
                self._sweep_stale_admitted()
                self._run_cycle()
            except Exception as exc:
                logger.error("Cycle error: %s", exc, exc_info=True)
            self._stop.wait(self.cycle_interval)

    def _sweep_running_ctx(self) -> None:
        """Evict completion-context entries older than TTL that are no longer running."""
        now = time.monotonic()
        with self._running_lock:
            running_uids: set[str] = set()
            for lane_dict in self._kubernetes_running.values():
                running_uids.update(lane_dict.keys())
        stale: list[str] = []
        with self._running_ctx_lock:
            for uid, ctx in list(self._running_ctx.items()):
                ctx_created = ctx[4]
                if uid not in running_uids and (now - ctx_created) > _RUNNING_CTX_TTL:
                    stale.append(uid)
            for uid in stale:
                self._running_ctx.pop(uid, None)
        if stale:
            logger.info("Evicted %d stale completion-context entries (TTL exceeded)", len(stale))

    def _sweep_stale_admitted(self) -> None:
        """
        Validate admitted entries older than _ADMITTED_STALE_TTL against the cluster.
        If the pod no longer exists or is no longer in a gated/pending state,
        remove the reserved capacity to prevent permanent over-commitment.
        """
        now = time.monotonic()
        with self._admitted_timestamp_lock:
            stale = {
                uid: entry
                for uid, entry in self._admitted_with_timestamp.items()
                if now - entry[4] > _ADMITTED_STALE_TTL
            }

        for uid, (namespace, name, lane, units, _ts) in stale.items():
            pod_exists = True
            still_admitted = True
            try:
                pod = self.core_v1.read_namespaced_pod(name=name, namespace=namespace)
                pod_dict = self.core_v1.api_client.sanitize_for_serialization(pod)
                phase = (pod_dict.get("status") or {}).get("phase", "")
                if phase not in ("Pending", ""):
                    still_admitted = False
            except client.exceptions.ApiException as exc:
                if exc.status == 404:
                    pod_exists = False
                    still_admitted = False

            if not still_admitted:
                logger.warning(
                    "Stale admitted entry for pod %s/%s (age>%.0fs, exists=%s) — "
                    "releasing %.1f units from lane %s",
                    namespace, name, _ADMITTED_STALE_TTL, pod_exists, units, lane,
                )
                with self._admitted_lock:
                    self._admitted.discard(uid)
                with self._admitted_resources_lock:
                    self._admitted_resources.pop(uid, None)
                with self._admitted_timestamp_lock:
                    self._admitted_with_timestamp.pop(uid, None)

    def _run_cycle(self) -> None:
        caps = self.node_tracker.lane_capacity()
        if caps and all(v == 0.0 for v in caps.values()):
            logger.debug("No node capacity known yet — skipping cycle")
            return

        now = time.monotonic()

        with self._running_lock:
            running_units: dict = {
                lane: sum(rp.resource_units for rp in pods.values())
                for lane, pods in self._kubernetes_running.items()
            }
            running_counts: dict = {}
            for lane, pods in self._kubernetes_running.items():
                counts: dict = {}
                for uid in pods:
                    uname = self._kubernetes_running_user.get(uid, "")
                    counts[uname] = counts.get(uname, 0) + 1
                running_counts[lane] = counts

        self.scheduler.update_running_counts(running_counts)

        with self._kubernetes_pending_lock:
            pending_units: dict = {
                lane: sum(rp.resource_units for rp in pods.values())
                for lane, pods in self._kubernetes_pending.items()
            }

        with self._admitted_resources_lock:
            admitted_units: dict = {}
            for _uid, (lane, units) in self._admitted_resources.items():
                admitted_units[lane] = admitted_units.get(lane, 0.0) + units

        dispatched = self.scheduler.cycle(now=now)

        if not dispatched:
            logger.debug("Cycle complete — no jobs dispatched")
            return

        logger.info("Cycle dispatching %d jobs", len(dispatched))

        for job in dispatched:
            with self._admit_attempts_lock:
                attempts_entry = self._admit_attempts.get(job.job_id)
            if attempts_entry is not None and attempts_entry[1] > now:
                with self._pending_lock:
                    pod = self._pending.get(job.job_id)
                if pod is not None:
                    try:
                        self.scheduler.submit(job)
                    except ValueError:
                        pass
                continue

            # Capacity gate: free = capacity - running - k8s_pending - admitted
            lane_cap  = caps.get(job.lane, 0.0)
            running   = running_units.get(job.lane, 0.0)
            k8s_pend  = pending_units.get(job.lane, 0.0)
            admitted  = admitted_units.get(job.lane, 0.0)
            free      = lane_cap - running - k8s_pend - admitted
            if free < job.resource_units:
                with self._pending_lock:
                    pod = self._pending.get(job.job_id)
                if pod is not None:
                    try:
                        self.scheduler.submit(job)
                    except ValueError:
                        pass
                logger.debug(
                    "Capacity gate: deferred %s (need=%.1f free=%.1f lane=%s)",
                    job.job_id, job.resource_units, free, job.lane,
                )
                continue

            # Speculatively reserve capacity before the patch
            meta = self._get_pod_meta(job.job_id)
            with self._admitted_resources_lock:
                self._admitted_resources[job.job_id] = (job.lane, job.resource_units)
            with self._admitted_timestamp_lock:
                ns   = meta[0] if meta else ""
                name = meta[1] if meta else ""
                self._admitted_with_timestamp[job.job_id] = (
                    ns, name, job.lane, job.resource_units, now,
                )
            admitted_units[job.lane] = admitted + job.resource_units

            pod = self._pop_pending(job.job_id)
            if pod is None:
                logger.warning("Dispatched job %s has no matching pending pod", job.job_id)
                with self._admitted_resources_lock:
                    self._admitted_resources.pop(job.job_id, None)
                with self._admitted_timestamp_lock:
                    self._admitted_with_timestamp.pop(job.job_id, None)
                admitted_units[job.lane] -= job.resource_units
                continue
            self._admit_pod(pod, job)

        logger.info("Capacity: %s", self.node_tracker.summary())

    def _get_pod_meta(self, uid: str) -> Optional[tuple[str, str]]:
        with self._pending_lock:
            pod = self._pending.get(uid)
        if pod is None:
            return None
        meta = pod.get("metadata") or {}
        return meta.get("namespace", ""), meta.get("name", "")

    def _pop_pending(self, uid: str) -> Optional[dict]:
        with self._pending_lock:
            return self._pending.pop(uid, None)

    def _admit_pod(self, pod: dict, job: Job) -> None:
        meta      = pod.get("metadata", {}) or {}
        namespace = meta.get("namespace", "")
        name      = meta.get("name", "")
        uid       = meta.get("uid", "")

        patch = admission_patch(pod)
        if not patch:
            logger.debug("Pod %s/%s already admitted — skipping patch", namespace, name)
            with self._admitted_lock:
                self._admitted.add(uid)
            with self._admit_attempts_lock:
                self._admit_attempts.pop(uid, None)
            return

        if self.dry_run:
            logger.info(
                "DRY RUN: would patch pod %s/%s [group=%s lane=%s wait=%.1fs]",
                namespace, name, job.sched_group_id, job.lane,
                job.wait_seconds(now=time.monotonic()),
            )
            with self._admitted_lock:
                self._admitted.add(uid)
            with self._admit_attempts_lock:
                self._admit_attempts.pop(uid, None)
            return

        try:
            self.core_v1.patch_namespaced_pod(
                name      = name,
                namespace = namespace,
                body      = patch,
            )
            with self._admitted_lock:
                self._admitted.add(uid)
            with self._admit_attempts_lock:
                self._admit_attempts.pop(uid, None)
            logger.info(
                "Admitted pod %s/%s — gate removed [group=%s lane=%s wait=%.1fs]",
                namespace, name, job.sched_group_id, job.lane,
                job.wait_seconds(now=time.monotonic()),
            )
        except client.exceptions.ApiException as exc:
            with self._admitted_resources_lock:
                self._admitted_resources.pop(uid, None)
            with self._admitted_timestamp_lock:
                self._admitted_with_timestamp.pop(uid, None)
            if exc.status == 404:
                logger.info("Pod %s/%s vanished before admission — skipping", namespace, name)
                with self._admit_attempts_lock:
                    self._admit_attempts.pop(uid, None)
                return
            self._handle_admit_failure(pod, job, exc)

    def _handle_admit_failure(self, pod: dict, job: Job,
                              exc: "client.exceptions.ApiException") -> None:
        meta      = pod.get("metadata", {}) or {}
        namespace = meta.get("namespace", "")
        name      = meta.get("name", "")
        uid       = meta.get("uid", "")

        with self._admit_attempts_lock:
            prev_attempts, _ = self._admit_attempts.get(uid, (0, 0.0))
            attempts = prev_attempts + 1
            if attempts >= _MAX_ADMIT_ATTEMPTS:
                self._admit_attempts.pop(uid, None)
                give_up = True
            else:
                backoff = min(2.0 ** attempts, _MAX_ADMIT_BACKOFF)
                self._admit_attempts[uid] = (attempts, time.monotonic() + backoff)
                give_up = False

        if give_up:
            logger.error(
                "Giving up on pod %s/%s after %d admit failures: %s",
                namespace, name, _MAX_ADMIT_ATTEMPTS, exc,
            )
            while self.scheduler.remove_job(uid) is not None:
                pass
            with self._pending_lock:
                self._pending.pop(uid, None)
            try:
                self.event_publisher.warn_unknown_gpu_class(uid, pod)
            except Exception:
                pass
            return

        logger.warning(
            "Failed to patch pod %s/%s (attempt %d/%d): %s — backing off",
            namespace, name, attempts, _MAX_ADMIT_ATTEMPTS, exc,
        )
        with self._pending_lock:
            self._pending[uid] = pod
        try:
            self.scheduler.submit(job)
        except ValueError:
            pass

    def get_wait_estimate(self, pod_uid: str) -> Optional[WaitEstimate]:
        return self.wait_cache.get(pod_uid)

    def _build_wait_snapshot(self) -> dict[str, WaitEstimate]:
        """
        Compute WaitEstimates for every currently queued pod, then publish
        Kubernetes Events for pods whose emission schedule is due.
        """
        from lane_scheduler.core.scheduler import Lane as _Lane

        now  = time.monotonic()
        caps = self.node_tracker.lane_capacity()

        with self._pending_lock:
            pending_snapshot = dict(self._pending)
        with self._running_lock:
            running_snapshot = {
                lane: dict(pods)
                for lane, pods in self._kubernetes_running.items()
            }
        with self._kubernetes_pending_lock:
            kpending_snapshot = {
                lane: dict(pods)
                for lane, pods in self._kubernetes_pending.items()
            }
        with self._admitted_resources_lock:
            admitted_by_lane: dict = {}
            for _uid, (lane, units) in self._admitted_resources.items():
                admitted_by_lane[lane] = admitted_by_lane.get(lane, 0.0) + units

        estimates:       dict[str, WaitEstimate] = {}
        lane_candidates: dict[str, list]         = {}

        for lane in (_Lane or []):
            lane_name    = lane
            running      = list((running_snapshot.get(lane) or {}).values())
            running_u    = sum(rp.resource_units for rp in running)
            k8s_pend_u   = sum(
                rp.resource_units for rp in (kpending_snapshot.get(lane) or {}).values()
            )
            admitted_u   = admitted_by_lane.get(lane, 0.0)
            free_units   = max(0.0, caps.get(lane, 0.0) - running_u - k8s_pend_u - admitted_u)
            candidates   = self.scheduler._scored_candidates(lane, now)
            lane_candidates[lane] = candidates

            for rank, (_, job) in enumerate(candidates, start=1):
                profiles = {
                    "interactive": self.residency_stats.profile_for(
                        job.sched_group_id, lane_name, batch=False),
                    "batch":       self.residency_stats.profile_for(
                        job.sched_group_id, lane_name, batch=True),
                }
                est = estimate_wait(
                    queue_rank     = rank,
                    lane_name      = lane_name,
                    running        = running,
                    profiles       = profiles,
                    required_units = job.resource_units,
                    free_units     = free_units,
                    now            = now,
                )
                estimates[job.job_id] = est

        try:
            n = self.event_publisher.publish_due(
                estimates        = estimates,
                lane_candidates  = lane_candidates,
                pending_snapshot = pending_snapshot,
                now              = now,
            )
            if n:
                logger.info("Published %d queue position event(s)", n)
        except Exception as exc:
            logger.error("Event publication error: %s", exc, exc_info=True)

        return estimates


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lane-based Priority Scheduler")

    p.add_argument("--kubeconfig", default=KUBECONFIG,
                   help="Path to kubeconfig (default: $KUBECONFIG; omit for in-cluster config)")
    p.add_argument("--cycle-interval", type=float, default=CYCLE_INTERVAL,
                   help="Scheduling cycle interval in seconds (default: %(default)s)")
    p.add_argument("--dispatch-k", type=int, default=DISPATCH_K,
                   help="Max jobs admitted per lane per cycle (default: %(default)s)")
    p.add_argument("--alpha", type=float, default=ALPHA,
                   help="Aging boost scale α in P=W×Mode×Age/U (default: %(default)s)")
    p.add_argument("--t-half-interactive", type=float, default=T_HALF_INTERACTIVE,
                   help="Interactive aging half-life in seconds (default: %(default)s)")
    p.add_argument("--t-half-batch", type=float, default=T_HALF_BATCH,
                   help="Batch aging half-life in seconds (default: %(default)s)")
    p.add_argument("--util-window", type=float, default=UTIL_WINDOW,
                   help="Utilization rolling window in seconds (default: %(default)s)")
    p.add_argument("--interactive-mean-pct", type=float, default=INTERACTIVE_MEAN_PCT,
                   help="Interactive residency mean as fraction of deadline (default: %(default)s)")
    p.add_argument("--interactive-std-pct", type=float, default=INTERACTIVE_STD_PCT,
                   help="Interactive residency std as fraction of deadline (default: %(default)s)")
    p.add_argument("--batch-mean-pct", type=float, default=BATCH_MEAN_PCT,
                   help="Batch residency mean as fraction of deadline (default: %(default)s)")
    p.add_argument("--batch-std-pct", type=float, default=BATCH_STD_PCT,
                   help="Batch residency std as fraction of deadline (default: %(default)s)")
    p.add_argument("--prior-weight", type=float, default=PRIOR_WEIGHT,
                   help="Bayesian prior pseudo-count for per-group residency (default: %(default)s)")
    p.add_argument("--ewma-alpha", type=float, default=EWMA_ALPHA,
                   help="EWMA smoothing factor for per-group residency (default: %(default)s)")
    p.add_argument("--wait-cache-interval", type=float, default=WAIT_CACHE_INTERVAL,
                   help="Wait-time cache refresh interval in seconds (default: %(default)s)")
    p.add_argument("--default-active-deadline-seconds", type=int, default=DEFAULT_ACTIVE_DEADLINE,
                   help="Default activeDeadlineSeconds for pods that have none set (default: %(default)s)")
    p.add_argument("--course-label", default=LABEL_SCHED_GROUP,
                   help="Pod label key for scheduling group identifier (default: %(default)s)")
    p.add_argument("--user-label", default=LABEL_USER,
                   help="Pod label key for username / per-user fairness (default: %(default)s)")
    p.add_argument("--batch-label", default=LABEL_BATCH,
                   help="Pod label key for batch mode flag (default: %(default)s)")
    p.add_argument("--gpu-class-label", default=LABEL_GPU_CLASS,
                   help="Label key on both pods and nodes identifying the GPU class (default: %(default)s)")
    p.add_argument("--scheduling-gate-name", default=SCHEDULING_GATE_NAME,
                   help="Name of the scheduling gate injected by the mutating webhook (default: %(default)s)")
    p.add_argument("--web-port", type=int, default=WEB_PORT,
                   help="Port for the queue-snapshot dashboard (0 = disabled, default: %(default)s)")
    p.add_argument("--dry-run", action="store_true", default=DRY_RUN,
                   help="Log what would be done without patching pods or creating events")
    p.add_argument("--no-unknown-gpu-class-events", action="store_true",
                   default=NO_UNKNOWN_GPU_CLASS_EVENTS,
                   help="Suppress Warning events for pods with unmanaged gpu-class labels")
    p.add_argument("--log-level", default=LOG_LEVEL,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default: %(default)s)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
    )

    # Propagate CLI overrides to modules that capture env vars at import time
    import lane_scheduler.core.node_capacity as _nc
    import lane_scheduler.k8s.pod_translator as _pt
    _nc.INHIBIT_TAINT_KEY   = args.scheduling_gate_name  # kept for node tracker compat
    _nc.GPU_CLASS_LABEL_KEY = args.gpu_class_label
    _pt.LABEL_SCHED_GROUP   = args.course_label
    _pt.LABEL_USER          = args.user_label
    _pt.LABEL_BATCH         = args.batch_label
    _pt.LABEL_GPU_CLASS     = args.gpu_class_label
    _pt._GPU_TAINT_KEY      = args.gpu_class_label
    _pt.SCHEDULING_GATE_NAME = args.scheduling_gate_name
    _g = globals()
    _g['LABEL_SCHED_GROUP']    = args.course_label
    _g['LABEL_GPU_CLASS']      = args.gpu_class_label
    _g['SCHEDULING_GATE_NAME'] = args.scheduling_gate_name

    if args.kubeconfig:
        k8s_config.load_kube_config(config_file=args.kubeconfig)
    else:
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            logger.info("Not in-cluster; falling back to default kubeconfig")
            k8s_config.load_kube_config()

    core_v1 = client.CoreV1Api()

    gpu_classes = discover_gpu_classes(core_v1)
    initialise_lanes(gpu_classes)

    registry = SchedGroupRegistry()

    sched_config = SchedulerConfig(
        alpha               = args.alpha,
        t_half_interactive  = args.t_half_interactive,
        t_half_batch        = args.t_half_batch,
        utilization_window  = args.util_window,
        dispatch_k          = args.dispatch_k,
    )

    residency_profiles = {
        "interactive": ResidencyProfile(
            mean_pct = args.interactive_mean_pct,
            std_pct  = args.interactive_std_pct,
        ),
        "batch": ResidencyProfile(
            mean_pct = args.batch_mean_pct,
            std_pct  = args.batch_std_pct,
        ),
    }

    controller = LaneSchedulerController(
        core_v1                     = core_v1,
        registry                    = registry,
        sched_config                = sched_config,
        residency_profiles          = residency_profiles,
        prior_weight                = args.prior_weight,
        ewma_alpha                  = args.ewma_alpha,
        cycle_interval              = args.cycle_interval,
        wait_cache_interval         = args.wait_cache_interval,
        web_port                    = args.web_port,
        dry_run                     = args.dry_run,
        no_unknown_gpu_class_events = args.no_unknown_gpu_class_events,
        default_active_deadline     = args.default_active_deadline_seconds,
    )

    signal.signal(signal.SIGTERM, lambda *_: controller.stop())
    signal.signal(signal.SIGINT,  lambda *_: controller.stop())

    controller.run()


if __name__ == "__main__":
    main()
