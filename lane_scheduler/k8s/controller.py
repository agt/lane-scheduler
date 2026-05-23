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

from lane_scheduler.core.scheduler import Job, Lane, SchedulerConfig, Scheduler, initialise_lanes, lane_for_gpu_class
from lane_scheduler.core.course_registry import CourseRegistry
from lane_scheduler.core.node_capacity import NodeCapacityTracker
from lane_scheduler.k8s.pod_translator import (
    NO_COURSE_LABEL,
    LABEL_COURSE,
    LABEL_BATCH,
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
        course_csv:          Optional[Path] = None,
        cycle_interval:      float = CYCLE_INTERVAL,
        reload_interval:     float = RELOAD_INTERVAL,
        wait_cache_interval: float = WAIT_CACHE_INTERVAL,
        web_port:            int   = 0,
        dry_run:             bool  = False,
    ):
        self.core_v1            = core_v1
        self.registry           = registry
        self.sched_config       = sched_config
        self.course_csv         = course_csv
        self.cycle_interval     = cycle_interval
        self.reload_interval    = reload_interval
        self.web_port           = web_port
        self.dry_run            = dry_run

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

        # Pods we have already patched (avoid double-patching)
        self._admitted: set[str] = set()
        self._admitted_lock = threading.Lock()

        # Running pods per lane: {lane: {uid: RunningPod}}
        self._running: dict[object, dict[str, RunningPod]] = {}
        self._running_lock = threading.Lock()

        # Completion context: uid → (course_id, lane_name, batch, deadline)
        # Needed to record residency when a running pod reaches a terminal phase.
        self._running_ctx: dict[str, tuple[str, str, bool, float]] = {}
        self._running_ctx_lock = threading.Lock()

        # Per-course residency statistics (Bayesian, updated on completions)
        self.residency_stats = ResidencyStats(
            interactive_prior = residency_profiles["interactive"],
            batch_prior       = residency_profiles["batch"],
            prior_weight      = prior_weight,
        )

        # Background wait-time cache
        self.wait_cache = WaitTimeCache(
            snapshot_fn = self._build_wait_snapshot,
            interval    = wait_cache_interval,
        )

        # Kubernetes Event publisher (best-effort, non-blocking on errors)
        self.event_publisher = EventPublisher(core_v1, dry_run=dry_run)

        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("Lane Scheduler starting")

        self.wait_cache.start()

        threads = [
            threading.Thread(target=self._pod_watch_loop,    name="pod-watch",    daemon=True),
            threading.Thread(target=self._node_watch_loop,   name="node-watch",   daemon=True),
            threading.Thread(target=self._cycle_loop,        name="cycle",        daemon=True),
            threading.Thread(target=self._csv_reload_loop,   name="csv-reload",   daemon=True),
        ]
        for t in threads:
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
            self._stop.set()
            self.wait_cache.stop()
            logger.info("Lane Scheduler stopped")

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Pod watch
    # ------------------------------------------------------------------

    def _pod_watch_loop(self) -> None:
        w = watch.Watch()
        while not self._stop.is_set():
            try:
                logger.info("Starting pod watch")
                for event in w.stream(
                    self.core_v1.list_pod_for_all_namespaces,
                    timeout_seconds=60,
                ):
                    if self._stop.is_set():
                        break
                    self._handle_pod_event(event)
            except Exception as exc:
                logger.warning("Pod watch error: %s — reconnecting in 5s", exc)
                time.sleep(5.0)

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

        gpu_lane   = _gpu_lane(pod)
        _, has_gpu = _resource_units(pod, gpu_lane)
        batch      = _is_batch(pod)

        # resource_units — reuse pod_translator logic
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
            self._running_ctx[uid] = (course_id, lane_name, batch, deadline)

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

        course_id, lane_name, batch, deadline = ctx
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
            if course_id not in self.scheduler._classes:
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
            self._pending.pop(uid, None)
        with self._admitted_lock:
            self._admitted.discard(uid)
        self.event_publisher.deregister(uid)

    # ------------------------------------------------------------------
    # Node watch
    # ------------------------------------------------------------------

    def _node_watch_loop(self) -> None:
        w = watch.Watch()
        while not self._stop.is_set():
            try:
                logger.info("Starting node watch")
                for event in w.stream(
                    self.core_v1.list_node,
                    timeout_seconds=60,
                ):
                    if self._stop.is_set():
                        break
                    self._handle_node_event(event)
                    self._sync_lane_capacity()
            except Exception as exc:
                logger.warning("Node watch error: %s — reconnecting in 5s", exc)
                time.sleep(5.0)

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
        self.scheduler.lane_capacity = caps
        self.scheduler.util._lane_capacity = caps

    # ------------------------------------------------------------------
    # Scheduling cycle
    # ------------------------------------------------------------------

    def _cycle_loop(self) -> None:
        # Wait briefly for the watches to populate initial state
        time.sleep(max(self.cycle_interval, 5.0))

        while not self._stop.is_set():
            try:
                self._run_cycle()
            except Exception as exc:
                logger.error("Cycle error: %s", exc, exc_info=True)
            self._stop.wait(self.cycle_interval)

    def _run_cycle(self) -> None:
        caps = self.node_tracker.lane_capacity()
        if all(v == 0.0 for v in caps.values()):
            logger.debug("No node capacity known yet — skipping cycle")
            return

        now       = time.monotonic()
        dispatched = self.scheduler.cycle(now=now)

        if not dispatched:
            logger.debug("Cycle complete — no jobs dispatched")
            return

        logger.info("Cycle dispatching %d jobs", len(dispatched))

        for job in dispatched:
            pod = self._pop_pending(job.job_id)
            if pod is None:
                logger.warning("Dispatched job %s has no matching pending pod", job.job_id)
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
            return

        try:
            self.core_v1.patch_namespaced_pod(
                name      = name,
                namespace = namespace,
                body      = patch,
            )
            with self._admitted_lock:
                self._admitted.add(uid)
            logger.info(
                "Admitted pod %s/%s [course=%s lane=%s wait=%.1fs]",
                namespace, name, job.class_id, job.lane.name,
                job.wait_seconds(now=time.monotonic()),
            )
        except client.exceptions.ApiException as exc:
            if exc.status == 404:
                logger.info("Pod %s/%s vanished before admission — skipping", namespace, name)
            else:
                logger.error("Failed to patch pod %s/%s: %s", namespace, name, exc)
                # Re-enqueue so it is retried next cycle
                with self._pending_lock:
                    self._pending[uid] = pod
                self.scheduler.submit(job)

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
                        if course.class_id not in self.scheduler._classes:
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
    p.add_argument("--kubeconfig", default=None,
                   help="Path to kubeconfig (omit to use in-cluster config)")
    p.add_argument("--course-csv", default=COURSE_CSV,
                   help="Path to registrar CSV (default: %(default)s)")
    p.add_argument("--cycle-interval", type=float, default=CYCLE_INTERVAL,
                   help="Scheduling cycle interval in seconds (default: %(default)s)")
    p.add_argument("--reload-interval", type=float, default=RELOAD_INTERVAL,
                   help="CSV reload interval in seconds (default: %(default)s)")
    p.add_argument("--wait-cache-interval", type=float, default=WAIT_CACHE_INTERVAL,
                   help="Wait-time cache refresh interval in seconds (default: %(default)s)")
    p.add_argument("--prior-weight", type=float, default=PRIOR_WEIGHT,
                   help="Bayesian prior pseudo-count for per-course residency (default: %(default)s)")
    p.add_argument("--web-port", type=int, default=WEB_PORT,
                   help="Port for the queue-snapshot dashboard (0 = disabled, default: %(default)s)")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Log what would be done without patching pods or creating events")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
    )

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
        alpha               = ALPHA,
        t_half_interactive  = T_HALF_INTERACTIVE,
        t_half_batch        = T_HALF_BATCH,
        epsilon             = EPSILON,
        utilization_window  = UTIL_WINDOW,
        dispatch_k          = DISPATCH_K,
    )

    residency_profiles = {
        "interactive": ResidencyProfile(
            mean_pct = INTERACTIVE_MEAN_PCT,
            std_pct  = INTERACTIVE_STD_PCT,
        ),
        "batch": ResidencyProfile(
            mean_pct = BATCH_MEAN_PCT,
            std_pct  = BATCH_STD_PCT,
        ),
    }
    logger.info(
        "Residency profiles — interactive: mean=%.0f%% std=%.0f%%  "
        "batch: mean=%.0f%% std=%.0f%%",
        INTERACTIVE_MEAN_PCT * 100, INTERACTIVE_STD_PCT * 100,
        BATCH_MEAN_PCT * 100,       BATCH_STD_PCT * 100,
    )

    controller = LaneSchedulerController(
        core_v1             = core_v1,
        registry            = registry,
        sched_config        = sched_config,
        residency_profiles  = residency_profiles,
        prior_weight        = args.prior_weight,
        course_csv          = csv_path,
        cycle_interval      = args.cycle_interval,
        reload_interval     = args.reload_interval,
        wait_cache_interval = args.wait_cache_interval,
        web_port            = args.web_port,
        dry_run             = args.dry_run,
    )

    # Graceful shutdown on SIGTERM (standard in Kubernetes)
    signal.signal(signal.SIGTERM, lambda *_: controller.stop())

    controller.run()


if __name__ == "__main__":
    main()
