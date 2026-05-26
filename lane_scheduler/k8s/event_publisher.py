"""
Event Publisher
---------------
Publishes Kubernetes Event objects for queued pods describing their
queue position and estimated wait time.

Emission schedules
~~~~~~~~~~~~~~~~~~
Interactive pods (default):
    - Immediately at the next snapshot after the pod enters the queue
    - Every 60 s for the first 5 minutes  (emit_count 1–5)
    - Every 300 s thereafter              (emit_count 6+)

Batch pods (dsmlp/batch=true):
    - Immediately at the next snapshot (emit_count 0)
    - At 25%, 50%, 75% of the first median wait estimate (emit_count 1–3)
    - No further emissions after emit_count 3

Each Event is a fresh create (not a patch/update) so that kubectl and
the Kubernetes event stream show distinct timestamped entries.
Errors on individual creates are logged and skipped — event publication
is best-effort and must never block or crash the scheduling loop.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_EARLY_INTERVAL  =  60.0
_EARLY_COUNT     =   5
_LATE_INTERVAL   = 300.0

_BATCH_FRACTIONS = (0.25, 0.50, 0.75)
_BATCH_MAX_EMITS = 1 + len(_BATCH_FRACTIONS)

_EVENT_REASON         = "SchedulingQueued"
_EVENT_REASON_IGNORED = "UnknownGpuClass"
_EVENT_TYPE           = "Normal"
_EVENT_TYPE_WARNING   = "Warning"
_EVENT_COMPONENT      = "lane-scheduler"
_API_VERSION     = "v1"
_POD_KIND        = "Pod"


# ---------------------------------------------------------------------------
# Per-pod emission schedule
# ---------------------------------------------------------------------------

@dataclass
class PodEventSchedule:
    pod_uid:          str
    pod_name:         str
    namespace:        str
    resource_version: str
    batch:            bool  = False
    enqueued_at:      float = field(default_factory=time.monotonic)
    emit_count:       int   = 0
    next_emit_at:     float = 0.0
    _anchor_wait:     float = 0.0

    def due(self, now: float) -> bool:
        return now >= self.next_emit_at

    def advance(self, now: float, median_wait: float = 0.0) -> None:
        self.emit_count += 1

        if self.batch:
            if self.emit_count == 1:
                self._anchor_wait = max(median_wait, 0.0)
            elif (self._anchor_wait > 0
                  and median_wait > 2.0 * self._anchor_wait):
                self._anchor_wait = median_wait
            if self.emit_count < _BATCH_MAX_EMITS:
                fraction = _BATCH_FRACTIONS[self.emit_count - 1]
                self.next_emit_at = self.enqueued_at + fraction * self._anchor_wait
                if self.next_emit_at <= now:
                    self.next_emit_at = now
            else:
                self.next_emit_at = float("inf")
        else:
            if self.emit_count <= _EARLY_COUNT:
                self.next_emit_at = now + _EARLY_INTERVAL
            else:
                self.next_emit_at = now + _LATE_INTERVAL

    @property
    def exhausted(self) -> bool:
        return self.next_emit_at == float("inf")


# ---------------------------------------------------------------------------
# Group-scoped queue position tracker
# ---------------------------------------------------------------------------

def group_rank(
    pod_uid:         str,
    sched_group_id:  str,
    candidates:      "list[tuple[float, object]]",
) -> Optional[int]:
    """
    Return 1-based position of pod_uid within candidates that belong to
    sched_group_id.  Returns None if not found.
    """
    group_rank_val = 0
    for _, job in candidates:
        if job.sched_group_id == sched_group_id:
            group_rank_val += 1
            if job.job_id == pod_uid:
                return group_rank_val
    return None


# ---------------------------------------------------------------------------
# Event publisher
# ---------------------------------------------------------------------------

class EventPublisher:
    """
    Maintains per-pod emission schedules and publishes Kubernetes Events.

    Thread safety: all public methods are safe to call from any thread.
    """

    def __init__(
        self,
        core_v1:                      object,
        dry_run:                      bool = False,
        no_unknown_gpu_class_events:  bool = False,
    ) -> None:
        self._core_v1 = core_v1
        self._dry_run = dry_run
        self._no_unknown_gpu_class_events = no_unknown_gpu_class_events
        self._schedules: dict[str, PodEventSchedule] = {}
        self._unknown_gpu_warned: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, uid: str, pod: dict) -> None:
        """Register a newly queued pod.  Idempotent."""
        from lane_scheduler.k8s.pod_translator import _is_batch
        meta = pod.get("metadata") or {}
        with self._lock:
            if uid not in self._schedules:
                self._schedules[uid] = PodEventSchedule(
                    pod_uid          = uid,
                    pod_name         = meta.get("name", ""),
                    namespace        = meta.get("namespace", ""),
                    resource_version = meta.get("resourceVersion", ""),
                    batch            = _is_batch(pod),
                )

    def deregister(self, uid: str) -> None:
        """Remove a pod from the emission schedule."""
        with self._lock:
            self._schedules.pop(uid, None)

    # ------------------------------------------------------------------
    # Publication
    # ------------------------------------------------------------------

    def publish_due(
        self,
        estimates:        "dict[str, WaitEstimate]",
        lane_candidates:  "dict[object, list[tuple[float, object]]]",
        pending_snapshot: "dict[str, dict]",
        now:              float,
    ) -> int:
        """
        Publish events for all pods whose next_emit_at <= now.
        Returns the number of events successfully created.
        """
        with self._lock:
            due = [s for s in self._schedules.values()
                   if s.due(now) and not s.exhausted]

        published = 0
        for sched in due:
            uid = sched.pod_uid
            est = estimates.get(uid)
            if est is None:
                self.deregister(uid)
                continue

            pod = pending_snapshot.get(uid)
            g_rank = None
            if pod is not None:
                for lane, candidates in lane_candidates.items():
                    for _, job in candidates:
                        if job.job_id == uid:
                            g_rank = group_rank(uid, job.sched_group_id, candidates)
                            break
                    if g_rank is not None:
                        break

            message = _format_message(est, g_rank)
            ok = self._create_event(sched, message, now)
            if ok:
                with self._lock:
                    if uid in self._schedules:
                        self._schedules[uid].advance(
                            now, median_wait=est.median_seconds
                        )
                published += 1

        return published

    # ------------------------------------------------------------------
    # One-shot warning events
    # ------------------------------------------------------------------

    def warn_unknown_gpu_class(self, uid: str, pod: dict) -> None:
        """
        Emit a single Warning event for a pod whose gpu-class label is not
        managed by this controller.  Idempotent.
        """
        if self._no_unknown_gpu_class_events:
            return
        with self._lock:
            if uid in self._unknown_gpu_warned:
                return
            self._unknown_gpu_warned.add(uid)

        meta      = pod.get("metadata") or {}
        labels    = meta.get("labels") or {}
        gpu_class = labels.get("gpu-class", "?")
        namespace = meta.get("namespace", "")
        pod_name  = meta.get("name", "")
        res_ver   = meta.get("resourceVersion", "")

        message = (
            f"gpu-class '{gpu_class}' is not managed by this lane scheduler "
            f"and will be ignored. If this class should be gated, restart the "
            f"controller after adding the node label."
        )

        import datetime as dt
        now_dt     = dt.datetime.now(dt.timezone.utc)
        microtime  = now_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        timestamp  = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        event_name = (
            f"{pod_name}.unknown-gpu"
            .lower()
            .replace("_", "-")[:253]
        )
        body = {
            "apiVersion": _API_VERSION,
            "kind": "Event",
            "metadata": {"name": event_name, "namespace": namespace},
            "involvedObject": {
                "apiVersion":      _API_VERSION,
                "kind":            _POD_KIND,
                "name":            pod_name,
                "namespace":       namespace,
                "uid":             uid,
                "resourceVersion": res_ver,
            },
            "reason":             _EVENT_REASON_IGNORED,
            "action":             "Ignored",
            "message":            message,
            "type":               _EVENT_TYPE_WARNING,
            "eventTime":          microtime,
            "firstTimestamp":     timestamp,
            "lastTimestamp":      timestamp,
            "reportingComponent": _EVENT_COMPONENT,
            "reportingInstance":  _EVENT_COMPONENT,
            "source": {"component": _EVENT_COMPONENT},
            "count": 1,
        }

        if self._dry_run:
            logger.info(
                "DRY RUN: would create UnknownGpuClass event for pod %s/%s: %s",
                namespace, pod_name, message,
            )
            return

        try:
            self._core_v1.create_namespaced_event(namespace=namespace, body=body)
        except Exception as exc:
            logger.warning(
                "Failed to create UnknownGpuClass event for pod %s/%s: %s",
                namespace, pod_name, exc,
            )

    def clear_unknown_gpu_warning(self, uid: str) -> None:
        with self._lock:
            self._unknown_gpu_warned.discard(uid)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_event(self, sched: PodEventSchedule, message: str,
                      now: float) -> bool:
        import datetime as dt
        now_dt    = dt.datetime.now(dt.timezone.utc)
        microtime = now_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        timestamp = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        event_name = (
            f"{sched.pod_name}.queue.{sched.emit_count:04d}"
            .lower()
            .replace("_", "-")[:253]
        )

        body = {
            "apiVersion": _API_VERSION,
            "kind": "Event",
            "metadata": {
                "name":      event_name,
                "namespace": sched.namespace,
            },
            "involvedObject": {
                "apiVersion":      _API_VERSION,
                "kind":            _POD_KIND,
                "name":            sched.pod_name,
                "namespace":       sched.namespace,
                "uid":             sched.pod_uid,
                "resourceVersion": sched.resource_version,
            },
            "reason":              _EVENT_REASON,
            "action":              "Queued",
            "message":             message,
            "type":                _EVENT_TYPE,
            "eventTime":           microtime,
            "firstTimestamp":      timestamp,
            "lastTimestamp":       timestamp,
            "reportingComponent":  _EVENT_COMPONENT,
            "reportingInstance":   _EVENT_COMPONENT,
            "source": {
                "component": _EVENT_COMPONENT,
            },
            "count": 1,
        }

        if self._dry_run:
            logger.info(
                "DRY RUN: would create event for pod %s/%s (emit #%d): %s",
                sched.namespace, sched.pod_name, sched.emit_count + 1, message,
            )
            return True

        try:
            self._core_v1.create_namespaced_event(
                namespace = sched.namespace,
                body      = body,
            )
            logger.debug(
                "Event created for pod %s/%s (emit #%d): %s",
                sched.namespace, sched.pod_name, sched.emit_count + 1, message,
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed to create event for pod %s/%s: %s",
                sched.namespace, sched.pod_name, exc,
            )
            return False


# ---------------------------------------------------------------------------
# Message formatter
# ---------------------------------------------------------------------------

def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"


def _format_message(est: "WaitEstimate", group_rank_val: Optional[int]) -> str:
    """
    Build the Event message string shown in 'kubectl describe pod'.

    Example:
        Queue position: #3 overall in gpu-medium lane, #1 in CSE234_SP26_A00.
        Estimated wait: 8.2m (optimistic 4.1m, pessimistic 14.7m).
    """
    rank_str = f"#{est.queue_rank} overall in {est.lane_name} lane"
    if group_rank_val is not None:
        rank_str += f", #{group_rank_val} within scheduling group"

    wait_str = (
        f"Estimated wait: {_fmt_seconds(est.median_seconds)} "
        f"(optimistic {_fmt_seconds(est.p20_seconds)}, "
        f"pessimistic {_fmt_seconds(est.p80_seconds)})"
    )
    return f"Queue position: {rank_str}. {wait_str}."
