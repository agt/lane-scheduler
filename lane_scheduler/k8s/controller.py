"""
Lane-based Priority Scheduler Controller
-----------------------------------------
Watches all pod and node events cluster-wide.  Pending pods that carry the
dsmlp/course label are held by the inhibitory node taint until our scheduler
selects them, at which point we patch in the matching toleration and let the
default Kubernetes scheduler handle actual placement.

Entry point:
    python controller.py [--config config.yaml]

Kubernetes RBAC required (see rbac.yaml):
    Pods  : get, list, watch, patch
    Nodes : get, list, watch
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

# kubernetes-client is the only external dependency
try:
    from kubernetes import client, config as k8s_config, watch
except ImportError:
    sys.exit("kubernetes package not found.  Run: pip install kubernetes")

from lane_scheduler.core.scheduler import Job, Lane, SchedulerConfig, Scheduler, initialise_lanes, lane_for_gpu_class, is_known_gpu_class
from lane_scheduler.core.course_registry import CourseRegistry
from lane_scheduler.core.node_capacity import (
    NodeCapacityTracker,
    INHIBIT_TAINT_KEY, INHIBIT_TAINT_VALUE, GPU_CLASS_LABEL_KEY,
)
from lane_scheduler.k8s.pod_translator import (
    NO_COURSE_LABEL,
    LABEL_COURSE,
    LABEL_BATCH,
    LABEL_GPU_CLASS,
    admission_patch,
    needs_scheduling,
    pod_to_job,
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

    Called once at startup, before initialise_lanes(), so that the Lane enum
    reflects the actual hardware inventory.  A controller restart is required
    for new GPU classes added after startup to be recognised.

    Returns a (possibly empty) sorted list of gpu-class strings.
    """
    from lane_scheduler.core.node_capacity import GPU_CLASS_LABEL_KEY

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
            "only the CPU lane will be available until a controller restart.",
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


WEB_PORT            = _env_int(  "LANE_WEB_PORT",           8080)   # 0 = disabled
CYCLE_INTERVAL      = _env_float("LANE_CYCLE_INTERVAL",   10.0)   # seconds
DISPATCH_K          = _env_int(  "LANE_DISPATCH_K",         8)
ALPHA               = _env_float("LANE_ALPHA",              1.0)
T_HALF_INTERACTIVE  = _env_float("LANE_T_HALF_INTERACTIVE", 600.0)
T_HALF_BATCH        = _env_float("LANE_T_HALF_BATCH",      7200.0)
EPSILON             = _env_float("LANE_EPSILON",            0.01)
UTIL_WINDOW         = _env_float("LANE_UTIL_WINDOW",        300.0)
COURSE_CSV          = os.environ.get("LANE_COURSE_CSV", "/etc/lane-scheduler/courses.csv")
RELOAD_INTERVAL     = _env_float("LANE_RELOAD_INTERVAL",  86400.0) # daily

# Residency distribution parameters (fraction of activeDeadlineSeconds)
INTERACTIVE_MEAN_PCT = _env_float("LANE_INTERACTIVE_MEAN_PCT", 0.4)
INTERACTIVE_STD_PCT  = _env_float("LANE_INTERACTIVE_STD_PCT",  0.2)
BATCH_MEAN_PCT       = _env_float("LANE_BATCH_MEAN_PCT",       0.7)
BATCH_STD_PCT        = _env_float("LANE_BATCH_STD_PCT",        0.15)
WAIT_CACHE_INTERVAL  = _env_float("LANE_WAIT_CACHE_INTERVAL",  60.0)  # seconds
PRIOR_WEIGHT         = _env_float("LANE_PRIOR_WEIGHT",          10.0)  # pseudo-count
EWMA_ALPHA           = _env_float("LANE_EWMA_ALPHA",             0.1)  # residency EWMA smoothing factor
NO_UNKNOWN_GPU_CLASS_EVENTS = os.environ.get(
    "LANE_NO_UNKNOWN_GPU_CLASS_EVENTS", ""
).lower() in ("1", "true", "yes")

# Environment variable fallbacks for flags that previously had no env-var equivalent
KUBECONFIG = os.environ.get("KUBECONFIG")                                    # standard k8s convention
LOG_LEVEL  = os.environ.get("LANE_LOG_LEVEL", "INFO").upper()
DRY_RUN    = os.environ.get("LANE_DRY_RUN", "").lower() in ("1", "true", "yes")

# Admission retry / circuit breaker
_MAX_ADMIT_ATTEMPTS = 5
_MAX_ADMIT_BACKOFF  = 300.0   # cap individual backoff sleep at 5 minutes

# Eviction TTL for completion context entries when a pod's completion event
# was missed (e.g. watch reconnection gap).  24 hours.
_RUNNING_CTX_TTL = 86400.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class LaneSchedulerController:
    """
    Orchestrates the pod/node watch loops and the scheduling cycle.

    Threading model:
        - pod_watch_thread   : streams pod events, maintains _pending set
        - node_watch_thread  : streams node events, updates NodeCapacityTracker
        - cycle_thread       : runs scheduler.cycle() every CYCLE_INTERVAL seconds
        - csv_reload_thread  : reloads the course CSV every RELOAD_INTERVAL seconds
    """

    def __init__(
        self,
        core_v1:             client.CoreV1Api,
        registry:            CourseRegistry,
        sched_config:        SchedulerConfig,
        residency_profiles:  dict[str, ResidencyProfile],
        prior_weight:        float = PRIOR_WEIGHT,
        ewma_alpha:          float = EWMA_ALPHA,
        course_csv:          Optional[Path] = None,
        cycle_interval:      float = CYCLE_INTERVAL,
        reload_interval:     float = RELOAD_INTERVAL,
        wait_cache_interval: float = WAIT_CACHE_INTERVAL,
        web_port:                     int   = 0,
        dry_run:                      bool  = False,
        no_unknown_gpu_class_events:  bool  = False,
    ):
        self.core_v1            = core_v1
        self.registry           = registry
        self.sched_config       = sched_config
        self.course_csv         = course_csv
        self.cycle_interval     = cycle_interval
        self.reload_interval    = reload_interval
        self.web_port           = web_port
        self.dry_run            = dry_run
        self.no_unknown_gpu_class_events = no_unknown_gpu_class_events

        if dry_run:
            logger.warning("DRY RUN mode enabled — pods will not be patched and events will not be created")

        self.node_tracker = NodeCapacityTracker()

        self.scheduler = Scheduler(
            lane_capacity={lane: 0.0 for lane in Lane},
            config=sched_config,
        )

        # Pods currently in our queue: uid → pod dict snapshot
        self._pending: dict[str, dict] = {}
        self._pending_lock = threading.Lock()

        # Pods with an unrecognised gpu-class that we are ignoring
        self._ignored_gpu_class: set[str] = set()

        # Pods we have already patched (avoid double-patching)
        self._admitted: set[str] = set()
        self._admitted_lock = threading.Lock()

        # Admitted pods not yet Running: uid → (lane, resource_units).
        # Used by the capacity gate so we don't over-commit a lane.
        self._admitted_resources: dict[str, tuple[object, float]] = {}
        self._admitted_resources_lock = threading.Lock()

        # Running pods per lane: {lane: {uid: RunningPod}}
        self._running: dict[object, dict[str, RunningPod]] = {}
        self._running_lock = threading.Lock()

        # Completion context: uid → (course_id, lane_name, batch, deadline, ctx_created_monotonic)
        # Needed to record residency when a running pod reaches a terminal phase.
        # ctx_created_monotonic is used by _sweep_running_ctx to evict stale
        # entries when a completion event is missed (e.g. watch gap).
        self._running_ctx: dict[str, tuple[str, str, bool, float, float]] = {}
        self._running_ctx_lock = threading.Lock()

        # Per-pod admission retry state for non-404 apiserver errors.
        # uid → (attempt_count, next_retry_monotonic).  Cleared on success or 404.
        self._admit_attempts: dict[str, tuple[int, float]] = {}
        self._admit_attempts_lock = threading.Lock()

        # Last-seen resourceVersion for each watch, used to resume after
        # disconnect without losing events.  Reset to None on a 410 ("Gone")
        # to force a fresh list.
        self._pod_resource_version:  Optional[str] = None
        self._node_resource_version: Optional[str] = None

        # Watch handles stored so stop() can interrupt blocking streams.
        self._pod_watch:  Optional["watch.Watch"] = None
        self._node_watch: Optional["watch.Watch"] = None

        # Background threads (populated in run()) so shutdown can join them.
        self._threads: list[threading.Thread] = []

        # Per-course residency statistics (Bayesian, updated on completions)
        self.residency_stats = ResidencyStats(
            interactive_prior = residency_profiles["interactive"],
            batch_prior       = residency_profiles["batch"],
            prior_weight      = prior_weight,
            ewma_alpha        = ewma_alpha,
        )

        # Background wait-time cache
        self.wait_cache = WaitTimeCache(
            snapshot_fn = self._build_wait_snapshot,
            interval    = wait_cache_interval,
        )

        # Kubernetes Event publisher (best-effort, non-blocking on errors)
        self.event_publisher = EventPublisher(
            core_v1,
            dry_run=dry_run,
            no_unknown_gpu_class_events=no_unknown_gpu_class_events,
        )

        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("Lane Scheduler starting")

        self.wait_cache.start()

        self._threads = [
            threading.Thread(target=self._pod_watch_loop,    name="pod-watch",    daemon=True),
            threading.Thread(target=self._node_watch_loop,   name="node-watch",   daemon=True),
            threading.Thread(target=self._cycle_loop,        name="cycle",        daemon=True),
            threading.Thread(target=self._csv_reload_loop,   name="csv-reload",   daemon=True),
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
        # Interrupt any in-progress watch.stream() calls so the loops can
        # observe _stop without waiting up to 60s for the next timeout.
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
        resp = self.core_v1.list_pod_for_all_namespaces()
        items = getattr(resp, "items", None) or []
        rv = None
        meta = getattr(resp, "metadata", None)
        if meta is not None:
            rv = getattr(meta, "resource_version", None)
        for item in items:
            self._handle_pod_event({"type": "ADDED", "object": item})
        self._pod_resource_version = rv

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
            pod = pod.to_dict()

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
                    self._upsert_running(uid, pod)
                elif phase in ("Succeeded", "Failed"):
                    self._record_completion(uid, pod)
                    self._remove_running(uid)
                else:
                    self._remove_running(uid)

        elif etype == "DELETED":
            self._dequeue(uid)
            self._remove_running(uid)

    def _running_pod_from_pod(self, uid: str, pod: dict) -> Optional[RunningPod]:
        """
        Build a RunningPod from a Kubernetes pod dict.
        Returns None if required fields are missing.
        """
        from lane_scheduler.k8s.pod_translator import _is_batch, _resource_units, _gpu_lane
        spec   = pod.get("spec")   or {}
        status = pod.get("status") or {}

        deadline = spec.get("activeDeadlineSeconds")
        if not deadline:
            return None   # unmetered pod — skip

        # start_time from status.startTime (ISO8601); convert to monotonic offset
        start_str = status.get("startTime")
        if not start_str:
            return None
        try:
            import datetime
            dt      = datetime.datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            wall_age = (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds()
            start_time = time.monotonic() - max(0.0, wall_age)
        except Exception:
            return None

        gpu_lane = _gpu_lane(pod)
        batch    = _is_batch(pod)

        from lane_scheduler.k8s.pod_translator import _resource_units as _ru
        units = _ru(pod, gpu_lane)

        job_stub = type("_J", (), {"batch": batch, "lane": gpu_lane})()
        from lane_scheduler.k8s.pod_translator import _gpu_lane as _gl
        lane = _gl(pod)
        from lane_scheduler.core.scheduler import Lane as _Lane
        lane = lane if lane is not None else _Lane.CPU

        return RunningPod(
            pod_uid                 = uid,
            start_time              = start_time,
            active_deadline_seconds = float(deadline),
            batch                   = batch,
            resource_units          = units,
        )

    def _upsert_running(self, uid: str, pod: dict) -> None:
        rp = self._running_pod_from_pod(uid, pod)
        if rp is None:
            return
        from lane_scheduler.k8s.pod_translator import _gpu_lane, _is_batch
        from lane_scheduler.core.scheduler import Lane as _Lane, LANE_NAMES
        gpu_lane  = _gpu_lane(pod)
        lane      = gpu_lane or _Lane.CPU
        lane_name = LANE_NAMES.get(lane, str(lane))
        course_id = (
            (pod.get("metadata") or {}).get("labels") or {}
        ).get(LABEL_COURSE, NO_COURSE_LABEL) or NO_COURSE_LABEL
        batch     = _is_batch(pod)
        deadline  = float((pod.get("spec") or {}).get("activeDeadlineSeconds") or 0)

        with self._running_lock:
            if lane not in self._running:
                self._running[lane] = {}
            self._running[lane][uid] = rp

        with self._running_ctx_lock:
            self._running_ctx[uid] = (
                course_id, lane_name, batch, deadline, time.monotonic(),
            )

    def _remove_running(self, uid: str) -> None:
        with self._running_lock:
            for lane_dict in self._running.values():
                lane_dict.pop(uid, None)
        with self._running_ctx_lock:
            self._running_ctx.pop(uid, None)

    def _record_completion(self, uid: str, pod: dict) -> None:
        """
        Record a pod completion into ResidencyStats.

        Residency is computed from pod status.startTime and
        status.containerStatuses[*].state.terminated.finishedAt.
        If the pod ran to its deadline (or finish time is unavailable),
        residency is treated as 1.0.
        """
        with self._running_ctx_lock:
            ctx = self._running_ctx.get(uid)
        if ctx is None:
            return   # pod wasn't tracked (no deadline, or never seen as Running)

        course_id, lane_name, batch, deadline, _ctx_created = ctx
        if deadline <= 0:
            return

        residency_pct = 1.0   # default: assume full deadline consumed

        try:
            import datetime as _dt
            status     = pod.get("status") or {}
            start_str  = status.get("startTime")

            # Find the latest finishedAt across all container statuses
            finish_str = None
            for cs in (status.get("containerStatuses") or []):
                term = (cs.get("state") or {}).get("terminated") or {}
                fat  = term.get("finishedAt")
                if fat:
                    if finish_str is None or fat > finish_str:
                        finish_str = fat

            if start_str and finish_str:
                def _parse(s: str) -> _dt.datetime:
                    return _dt.datetime.fromisoformat(
                        s.replace("Z", "+00:00")
                    )
                elapsed = (_parse(finish_str) - _parse(start_str)).total_seconds()
                if elapsed > 0:
                    raw = elapsed / deadline
                    # Cap at 1.0 — slight clock skew can push it marginally over
                    residency_pct = min(1.0, raw)
        except Exception as exc:
            logger.debug("Could not compute residency for pod %s: %s", uid, exc)
            # Fall through with residency_pct = 1.0

        self.residency_stats.record(
            course_id     = course_id,
            lane_name     = lane_name,
            batch         = batch,
            residency_pct = residency_pct,
        )
        logger.info(
            "Completion recorded [course=%s lane=%s batch=%s residency=%.3f n=%d]",
            course_id, lane_name, batch, residency_pct,
            self.residency_stats.observation_count(course_id, lane_name, batch),
        )

    def running_pods_for_lane(self, lane: object) -> list[RunningPod]:
        """Return a snapshot of RunningPod objects for a given lane."""
        with self._running_lock:
            return list((self._running.get(lane) or {}).values())

    def _enqueue(self, uid: str, pod: dict) -> None:
        # Reject pods whose gpu-class label is not managed by this controller.
        gpu_class = ((pod.get("metadata") or {}).get("labels") or {}).get(
            LABEL_GPU_CLASS, ""
        ).strip()
        if gpu_class and not is_known_gpu_class(gpu_class):
            if uid not in self._ignored_gpu_class:
                self._ignored_gpu_class.add(uid)
                logger.info(
                    "Ignoring pod %s/%s — gpu-class '%s' is not managed by this controller",
                    (pod.get("metadata") or {}).get("namespace", "?"),
                    (pod.get("metadata") or {}).get("name", "?"),
                    gpu_class,
                )
                self.event_publisher.warn_unknown_gpu_class(uid, pod)
            return

        with self._admitted_lock:
            if uid in self._admitted:
                return   # already handled

        with self._pending_lock:
            if uid in self._pending:
                return   # already queued

            course_id  = (pod.get("metadata", {}).get("labels") or {}).get(
                LABEL_COURSE, NO_COURSE_LABEL
            ).strip() or NO_COURSE_LABEL
            course     = self.registry.get(course_id)

            # Register course with scheduler if not yet known
            if not self.scheduler.has_class(course_id):
                self.scheduler.register_class(course)

            submit_time = time.monotonic()
            job = pod_to_job(pod, submit_time=submit_time)
            self.scheduler.submit(job)
            self._pending[uid] = pod
            self.event_publisher.register(uid, pod)

            logger.info(
                "Enqueued pod %s/%s [course=%s lane=%s]",
                (pod.get("metadata") or {}).get("namespace", "?"),
                (pod.get("metadata") or {}).get("name", "?"),
                course_id, job.lane.name,
            )

    def _dequeue(self, uid: str) -> None:
        with self._pending_lock:
            was_pending = self._pending.pop(uid, None) is not None
        with self._admitted_lock:
            self._admitted.discard(uid)
        with self._admitted_resources_lock:
            self._admitted_resources.pop(uid, None)
        self._ignored_gpu_class.discard(uid)
        with self._admit_attempts_lock:
            self._admit_attempts.pop(uid, None)
        # If the pod was still queued, also remove its Job from the scheduler
        # queue so it doesn't get dispatched against a no-longer-pending pod.
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
        """List all nodes and seed NodeCapacityTracker + resourceVersion."""
        resp = self.core_v1.list_node()
        items = getattr(resp, "items", None) or []
        rv = None
        meta = getattr(resp, "metadata", None)
        if meta is not None:
            rv = getattr(meta, "resource_version", None)
        for item in items:
            self._handle_node_event({"type": "ADDED", "object": item})
        self._sync_lane_capacity()
        self._node_resource_version = rv

    def _handle_node_event(self, event: dict) -> None:
        etype = event.get("type", "")
        node  = event.get("object", {})
        if hasattr(node, "to_dict"):
            node = node.to_dict()

        name = (node.get("metadata") or {}).get("name", "<unknown>")

        if etype in ("ADDED", "MODIFIED"):
            self.node_tracker.upsert(node)
        elif etype == "DELETED":
            self.node_tracker.remove(name)

    def _sync_lane_capacity(self) -> None:
        """Push current node capacity into the scheduler's utilization tracker."""
        caps = self.node_tracker.lane_capacity()
        self.scheduler.set_lane_capacity(caps)

    # ------------------------------------------------------------------
    # Scheduling cycle
    # ------------------------------------------------------------------

    def _cycle_loop(self) -> None:
        # Wait briefly for the watches to populate initial state
        self._stop.wait(max(self.cycle_interval, 5.0))

        while not self._stop.is_set():
            try:
                self._sweep_running_ctx()
                self._run_cycle()
            except Exception as exc:
                logger.error("Cycle error: %s", exc, exc_info=True)
            self._stop.wait(self.cycle_interval)

    def _sweep_running_ctx(self) -> None:
        """
        Evict completion-context entries that are older than _RUNNING_CTX_TTL
        AND whose uid is not currently in self._running.  Catches leaks from
        missed completion events (e.g. watch reconnection gaps).
        """
        now = time.monotonic()
        with self._running_lock:
            running_uids: set[str] = set()
            for lane_dict in self._running.values():
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
            logger.info(
                "Evicted %d stale completion-context entries (TTL exceeded)",
                len(stale),
            )

    def _run_cycle(self) -> None:
        caps = self.node_tracker.lane_capacity()
        if all(v == 0.0 for v in caps.values()):
            logger.debug("No node capacity known yet — skipping cycle")
            return

        now       = time.monotonic()

        # Snapshot running units per lane before the cycle so the gate has a
        # consistent baseline that is unaffected by concurrent pod-watch events.
        with self._running_lock:
            running_units: dict = {
                lane: sum(rp.resource_units for rp in pods.values())
                for lane, pods in self._running.items()
            }

        # Snapshot admitted-but-not-running units; augmented in the loop below
        # so jobs committed earlier in the same cycle are visible to later ones.
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
            # Honour retry backoff: if this pod recently failed admission,
            # do not try again until next_retry_monotonic has passed.
            with self._admit_attempts_lock:
                attempts_entry = self._admit_attempts.get(job.job_id)
            if attempts_entry is not None and attempts_entry[1] > now:
                # Re-queue the Job so it'll be considered again in a later cycle.
                with self._pending_lock:
                    pod = self._pending.get(job.job_id)
                if pod is not None:
                    # Job stays in _pending; re-submit so the scheduler still
                    # has it in its queue (cycle just popped it).
                    try:
                        self.scheduler.submit(job)
                    except ValueError:
                        pass
                continue

            # Capacity gate: only admit if genuine free capacity exists.
            lane_cap = caps.get(job.lane, 0.0)
            running  = running_units.get(job.lane, 0.0)
            admitted = admitted_units.get(job.lane, 0.0)
            free     = lane_cap - running - admitted
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
                    job.job_id, job.resource_units, free, job.lane.name,
                )
                continue

            # Speculatively record before the patch so later jobs in this cycle
            # see the reservation; rolled back below if _pop_pending fails.
            with self._admitted_resources_lock:
                self._admitted_resources[job.job_id] = (job.lane, job.resource_units)
            admitted_units[job.lane] = admitted + job.resource_units

            pod = self._pop_pending(job.job_id)
            if pod is None:
                logger.warning("Dispatched job %s has no matching pending pod", job.job_id)
                with self._admitted_resources_lock:
                    self._admitted_resources.pop(job.job_id, None)
                admitted_units[job.lane] -= job.resource_units
                continue
            self._admit_pod(pod, job)

        logger.info("Capacity: %s", self.node_tracker.summary())

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
            logger.debug("Pod %s/%s already has toleration — skipping patch", namespace, name)
            with self._admitted_lock:
                self._admitted.add(uid)
            with self._admit_attempts_lock:
                self._admit_attempts.pop(uid, None)
            return

        if self.dry_run:
            logger.info(
                "DRY RUN: would patch pod %s/%s to add scheduling-gate toleration "
                "[course=%s lane=%s wait=%.1fs]",
                namespace, name, job.class_id, job.lane.name,
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
                "Admitted pod %s/%s [course=%s lane=%s wait=%.1fs]",
                namespace, name, job.class_id, job.lane.name,
                job.wait_seconds(now=time.monotonic()),
            )
        except client.exceptions.ApiException as exc:
            # Patch failed — release the speculative capacity reservation.
            with self._admitted_resources_lock:
                self._admitted_resources.pop(uid, None)
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
            # Best-effort cleanup: drop any copies left in the scheduler queue
            # (a Job may have been re-submitted on each failed attempt),
            # remove controller state, and surface a Warning Event so the
            # student/operator sees it.
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
        # Put the pod back in _pending and re-submit the Job into the
        # scheduler queue so a later cycle can retry once backoff elapses.
        with self._pending_lock:
            self._pending[uid] = pod
        try:
            self.scheduler.submit(job)
        except ValueError:
            # Class was unregistered between cycles — nothing we can do.
            pass

    def get_wait_estimate(self, pod_uid: str) -> Optional[WaitEstimate]:
        """
        Return the most recent cached WaitEstimate for a queued pod, or None.

        Results are at most wait_cache_interval seconds stale.
        Returns None if the pod is unknown or the cache has not yet populated.
        """
        return self.wait_cache.get(pod_uid)

    def _build_wait_snapshot(self) -> dict[str, WaitEstimate]:
        """
        Compute WaitEstimates for every currently queued pod, then publish
        Kubernetes Events for pods whose emission schedule is due.

        Each pod's wait estimate uses a per-course ResidencyProfile blended
        from the cluster-wide prior and course-specific completion observations.
        """
        from lane_scheduler.core.scheduler import LANE_NAMES, Lane as _Lane
        from lane_scheduler.k8s.pod_translator import _is_batch

        now = time.monotonic()

        with self._pending_lock:
            pending_snapshot = dict(self._pending)
        with self._running_lock:
            running_snapshot = {
                lane: dict(pods)
                for lane, pods in self._running.items()
            }

        estimates:       dict[str, WaitEstimate] = {}
        lane_candidates: dict[object, list]      = {}

        for lane in _Lane:
            lane_name  = LANE_NAMES.get(lane, str(lane))
            running    = list((running_snapshot.get(lane) or {}).values())
            candidates = self.scheduler._scored_candidates(lane, now)
            lane_candidates[lane] = candidates

            for rank, (_, job) in enumerate(candidates, start=1):
                # Build a per-job profiles dict using the course-specific posterior
                profiles = {
                    "interactive": self.residency_stats.profile_for(
                        job.class_id, lane_name, batch=False),
                    "batch":       self.residency_stats.profile_for(
                        job.class_id, lane_name, batch=True),
                }
                est = estimate_wait(
                    queue_rank     = rank,
                    lane_name      = lane_name,
                    running        = running,
                    profiles       = profiles,
                    required_units = job.resource_units,
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

    # ------------------------------------------------------------------
    # CSV reload
    # ------------------------------------------------------------------

    def _csv_reload_loop(self) -> None:
        while not self._stop.is_set():
            if self.course_csv and self.course_csv.exists():
                try:
                    n = self.registry.load_csv(self.course_csv)
                    # Re-register any newly loaded courses
                    for course in self.registry.all_courses():
                        if not self.scheduler.has_class(course.class_id):
                            self.scheduler.register_class(course)
                    logger.info("CSV reload: %d courses", n)
                except Exception as exc:
                    logger.warning("CSV reload failed: %s", exc)
            self._stop.wait(self.reload_interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lane-based Priority Scheduler")

    # --- Kubernetes connection ---
    p.add_argument("--kubeconfig", default=KUBECONFIG,
                   help="Path to kubeconfig (default: $KUBECONFIG; omit for in-cluster config)")

    # --- Course registry ---
    p.add_argument("--course-csv", default=COURSE_CSV,
                   help="Path to registrar CSV (default: %(default)s)")
    p.add_argument("--reload-interval", type=float, default=RELOAD_INTERVAL,
                   help="CSV reload interval in seconds (default: %(default)s)")

    # --- Scheduling cycle ---
    p.add_argument("--cycle-interval", type=float, default=CYCLE_INTERVAL,
                   help="Scheduling cycle interval in seconds (default: %(default)s)")
    p.add_argument("--dispatch-k", type=int, default=DISPATCH_K,
                   help="Max jobs admitted per lane per cycle (default: %(default)s)")

    # --- Priority scoring formula ---
    p.add_argument("--alpha", type=float, default=ALPHA,
                   help="Aging boost scale α in P=W×Mode×Age/U (default: %(default)s)")
    p.add_argument("--t-half-interactive", type=float, default=T_HALF_INTERACTIVE,
                   help="Interactive aging half-life in seconds (default: %(default)s)")
    p.add_argument("--t-half-batch", type=float, default=T_HALF_BATCH,
                   help="Batch aging half-life in seconds (default: %(default)s)")
    p.add_argument("--epsilon", type=float, default=EPSILON,
                   help="Utilization floor epsilon (default: %(default)s)")
    p.add_argument("--util-window", type=float, default=UTIL_WINDOW,
                   help="Utilization rolling window in seconds (default: %(default)s)")

    # --- Residency priors ---
    p.add_argument("--interactive-mean-pct", type=float, default=INTERACTIVE_MEAN_PCT,
                   help="Interactive residency mean as fraction of deadline (default: %(default)s)")
    p.add_argument("--interactive-std-pct", type=float, default=INTERACTIVE_STD_PCT,
                   help="Interactive residency std as fraction of deadline (default: %(default)s)")
    p.add_argument("--batch-mean-pct", type=float, default=BATCH_MEAN_PCT,
                   help="Batch residency mean as fraction of deadline (default: %(default)s)")
    p.add_argument("--batch-std-pct", type=float, default=BATCH_STD_PCT,
                   help="Batch residency std as fraction of deadline (default: %(default)s)")
    p.add_argument("--prior-weight", type=float, default=PRIOR_WEIGHT,
                   help="Bayesian prior pseudo-count for per-course residency (default: %(default)s)")
    p.add_argument("--ewma-alpha", type=float, default=EWMA_ALPHA,
                   help="EWMA smoothing factor for per-course residency (0,1); higher = faster adaptation (default: %(default)s)")

    # --- Wait-time cache ---
    p.add_argument("--wait-cache-interval", type=float, default=WAIT_CACHE_INTERVAL,
                   help="Wait-time cache refresh interval in seconds (default: %(default)s)")

    # --- Kubernetes label/taint wiring ---
    p.add_argument("--course-label", default=LABEL_COURSE,
                   help="Pod label key for course identifier (default: %(default)s)")
    p.add_argument("--pod-gpu-class-label", default=LABEL_GPU_CLASS,
                   help="Pod label key for requested GPU class (default: %(default)s)")
    p.add_argument("--node-gpu-class-label", default=GPU_CLASS_LABEL_KEY,
                   help="Node label key for GPU class (default: %(default)s)")
    p.add_argument("--inhibit-taint-key", default=INHIBIT_TAINT_KEY,
                   help="Inhibitory scheduling-gate taint key (default: %(default)s)")
    p.add_argument("--inhibit-taint-value", default=INHIBIT_TAINT_VALUE,
                   help="Inhibitory scheduling-gate taint value (default: %(default)s)")

    # --- Operational ---
    p.add_argument("--web-port", type=int, default=WEB_PORT,
                   help="Port for the queue-snapshot dashboard (0 = disabled, default: %(default)s)")
    p.add_argument("--dry-run", action="store_true", default=DRY_RUN,
                   help="Log what would be done without patching pods or creating events "
                        "(also set via LANE_DRY_RUN=true)")
    p.add_argument("--no-unknown-gpu-class-events", action="store_true",
                   default=NO_UNKNOWN_GPU_CLASS_EVENTS,
                   help=(
                       "Suppress Warning events for pods whose gpu-class label is not "
                       "managed by this controller (useful when multiple schedulers or "
                       "ungated GPU classes coexist in the same cluster). "
                       "Also set via LANE_NO_UNKNOWN_GPU_CLASS_EVENTS=true."
                   ))
    p.add_argument("--log-level", default=LOG_LEVEL,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (also set via LANE_LOG_LEVEL, default: %(default)s)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
    )

    # Propagate CLI overrides for Kubernetes wiring to every module that reads them.
    # These modules capture env vars as module-level constants at import time; updating
    # the attributes here ensures CLI flags take precedence regardless of import order.
    import lane_scheduler.core.node_capacity as _nc
    import lane_scheduler.k8s.pod_translator as _pt
    _nc.INHIBIT_TAINT_KEY   = args.inhibit_taint_key
    _nc.INHIBIT_TAINT_VALUE = args.inhibit_taint_value
    _nc.GPU_CLASS_LABEL_KEY = args.node_gpu_class_label
    _pt.INHIBIT_TAINT_KEY   = args.inhibit_taint_key
    _pt.INHIBIT_TAINT_VALUE = args.inhibit_taint_value
    _pt.LABEL_COURSE        = args.course_label
    _pt.LABEL_GPU_CLASS     = args.pod_gpu_class_label
    # Also update the names imported into this module's own namespace, which are
    # referenced by _enqueue() and _upsert_running() as module globals.
    _g = globals()
    _g['LABEL_COURSE']    = args.course_label
    _g['LABEL_GPU_CLASS'] = args.pod_gpu_class_label

    # Load Kubernetes config
    if args.kubeconfig:
        k8s_config.load_kube_config(config_file=args.kubeconfig)
    else:
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            logger.info("Not in-cluster; falling back to default kubeconfig")
            k8s_config.load_kube_config()

    core_v1 = client.CoreV1Api()

    # ----------------------------------------------------------------
    # GPU class discovery — MUST happen before any Lane reference
    # ----------------------------------------------------------------
    gpu_classes = discover_gpu_classes(core_v1)
    initialise_lanes(gpu_classes)

    # Load course registry
    registry = CourseRegistry()
    csv_path = Path(args.course_csv)
    if csv_path.exists():
        registry.load_csv(csv_path)
        logger.info("Loaded %d courses from %s", len(registry), csv_path)
    else:
        logger.warning("Course CSV not found at %s — all courses will use fallback inference",
                       csv_path)

    sched_config = SchedulerConfig(
        alpha               = args.alpha,
        t_half_interactive  = args.t_half_interactive,
        t_half_batch        = args.t_half_batch,
        epsilon             = args.epsilon,
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
    logger.info(
        "Residency profiles — interactive: mean=%.0f%% std=%.0f%%  "
        "batch: mean=%.0f%% std=%.0f%%",
        args.interactive_mean_pct * 100, args.interactive_std_pct * 100,
        args.batch_mean_pct * 100,       args.batch_std_pct * 100,
    )

    controller = LaneSchedulerController(
        core_v1                      = core_v1,
        registry                     = registry,
        sched_config                 = sched_config,
        residency_profiles           = residency_profiles,
        prior_weight                 = args.prior_weight,
        ewma_alpha                   = args.ewma_alpha,
        course_csv                   = csv_path,
        cycle_interval               = args.cycle_interval,
        reload_interval              = args.reload_interval,
        wait_cache_interval          = args.wait_cache_interval,
        web_port                     = args.web_port,
        dry_run                      = args.dry_run,
        no_unknown_gpu_class_events  = args.no_unknown_gpu_class_events,
    )

    # Graceful shutdown on SIGTERM (Kubernetes) and SIGINT (local Ctrl-C)
    signal.signal(signal.SIGTERM, lambda *_: controller.stop())
    signal.signal(signal.SIGINT,  lambda *_: controller.stop())

    controller.run()


if __name__ == "__main__":
    main()
