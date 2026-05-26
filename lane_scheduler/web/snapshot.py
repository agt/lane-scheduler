"""
Queue-snapshot aggregator for the web dashboard.

build_snapshot() reads controller state (read-only, respecting existing locks)
and returns a JSON-serializable dict covering:
  - lanes   : capacity, utilization, queue depth, drain estimate
  - courses : per-(course, lane) running/queued counts and wait estimates
  - system  : health / operational metadata

No student identifiers (namespaces / pod UIDs) appear in the output.
"""
from __future__ import annotations

import datetime
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lane_scheduler.k8s.controller import LaneSchedulerController

logger = logging.getLogger(__name__)

# Course ID assigned to pods that carry no dsmlp/course label — excluded from output.
_NO_COURSE = "__unlabelled__"


def _fmt_wait(est) -> Optional[dict]:
    if est is None:
        return None
    return {
        "median_s": round(est.median_seconds, 1),
        "p20_s":    round(est.p20_seconds, 1),
        "p80_s":    round(est.p80_seconds, 1),
        "rank":     est.queue_rank,
    }


def build_snapshot(ctrl: "LaneSchedulerController") -> dict:
    """
    Aggregate controller state into a JSON-serializable snapshot.

    Accesses controller internals read-only using the same locking discipline
    as _build_wait_snapshot().  Safe to call from any thread concurrently
    with the scheduling cycle.
    """
    from lane_scheduler.core.scheduler import Lane
    from lane_scheduler.estimation.wait_estimator import estimate_wait

    now = time.monotonic()

    # ------------------------------------------------------------------
    # Consistent snapshots of shared mutable state
    # ------------------------------------------------------------------
    lane_capacity = ctrl.node_tracker.lane_capacity()   # {Lane: float}
    node_counts   = ctrl.node_tracker.node_count()      # {Lane: int}

    with ctrl._pending_lock:
        pending_count = len(ctrl._pending)
    with ctrl._running_lock:
        running_snap = {
            lane: dict(pods)
            for lane, pods in ctrl._running.items()
        }
        student_snap = dict(ctrl._running_student)
    with ctrl._running_ctx_lock:
        # uid → (course_id, lane_name, batch, deadline, ctx_created_monotonic)
        ctx_snap = dict(ctrl._running_ctx)

    queue_depths  = ctrl.scheduler.queue_depths()   # {lane_name: {class_id: count}}
    cached_ests   = ctrl.wait_cache.all_estimates()  # {pod_uid: WaitEstimate}

    # ------------------------------------------------------------------
    # Running-pod aggregates
    # ------------------------------------------------------------------
    running_by_course_lane: dict[tuple, int]   = defaultdict(int)
    running_units_by_lane:  dict[str, float]   = defaultdict(float)
    running_count_by_lane:  dict[str, int]     = defaultdict(int)
    # uid → RunningPod lookup (flat, for resource_units attribution)
    rp_by_uid: dict[str, object] = {}
    for lane, pods in running_snap.items():
        for uid, rp in pods.items():
            rp_by_uid[uid] = rp

    for lane, pods in running_snap.items():
        lane_name = lane
        for uid, rp in pods.items():
            running_units_by_lane[lane_name] += rp.resource_units
            running_count_by_lane[lane_name] += 1

    for uid, ctx in ctx_snap.items():
        course_id, lane_name = ctx[0], ctx[1]
        running_by_course_lane[(course_id, lane_name)] += 1

    # ------------------------------------------------------------------
    # Scored candidates per lane — maps (lane_name, class_id) → top-candidate info
    # ------------------------------------------------------------------
    course_top: dict[tuple, dict] = {}
    for lane in sorted(Lane or []):
        lane_name    = lane
        running_pods = list(running_snap.get(lane, {}).values())
        try:
            candidates = ctrl.scheduler._scored_candidates(lane, now)
        except Exception:
            candidates = []
        for rank, (score, job) in enumerate(candidates, start=1):
            key = (lane_name, job.class_id)
            course_top[key] = {
                "rank":         rank,
                "job":          job,
                "running_pods": running_pods,
                "cached_est":   cached_ests.get(job.job_id),
            }

    # ------------------------------------------------------------------
    # Lane rows (View 1)
    # ------------------------------------------------------------------
    lanes_out = []
    for lane in sorted(Lane or []):
        lane_name    = lane
        capacity     = lane_capacity.get(lane, 0.0)
        nodes        = node_counts.get(lane, 0)
        run_units    = running_units_by_lane.get(lane_name, 0.0)
        run_count    = running_count_by_lane.get(lane_name, 0)
        queued_total = sum(queue_depths.get(lane_name, {}).values())

        # P80 drain estimate: how long until a hypothetical new arrival at the
        # back of the lane would be dispatched (uses cluster-wide prior profiles).
        drain_p80_s: Optional[float] = None
        if queued_total > 0:
            running_pods = list(running_snap.get(lane, {}).values())
            try:
                profiles = {
                    "interactive": ctrl.residency_stats.profile_for(
                        _NO_COURSE, lane_name, batch=False),
                    "batch": ctrl.residency_stats.profile_for(
                        _NO_COURSE, lane_name, batch=True),
                }
                drain_est = estimate_wait(
                    queue_rank=queued_total,
                    lane_name=lane_name,
                    running=running_pods,
                    profiles=profiles,
                    required_units=1.0,
                    now=now,
                )
                drain_p80_s = round(drain_est.p80_seconds, 1)
            except Exception:
                pass

        lanes_out.append({
            "name":           lane_name,
            "node_count":     nodes,
            "capacity_units": round(capacity, 1),
            "running_count":  run_count,
            "running_units":  round(run_units, 1),
            "queued_count":   queued_total,
            "drain_p80_s":    drain_p80_s,
        })

    # ------------------------------------------------------------------
    # Course rows (View 2)
    # ------------------------------------------------------------------
    all_course_ids: set[str] = set()
    for lane_name, cm in queue_depths.items():
        all_course_ids.update(cm.keys())
    for (course_id, _) in running_by_course_lane:
        all_course_ids.add(course_id)
    all_course_ids.discard(_NO_COURSE)

    courses_out = []
    for course_id in sorted(all_course_ids):
        course = ctrl.registry.get(course_id)
        course_lanes = []

        for lane in sorted(Lane or []):
            lane_name = lane
            queued  = queue_depths.get(lane_name, {}).get(course_id, 0)
            running = running_by_course_lane.get((course_id, lane_name), 0)
            if queued == 0 and running == 0:
                continue

            top_info  = course_top.get((lane_name, course_id))
            top_wait  = None
            tail_wait = None

            if top_info:
                top_wait = _fmt_wait(top_info["cached_est"])

                # Tail estimate: approximate the wait for the last job in this
                # course's sub-queue as rank = (top-candidate lane rank) + (depth - 1).
                # This is conservative (treats subsequent course candidates as
                # non-overlapping with other courses) but gives a useful upper bound.
                if queued > 1:
                    tail_rank = top_info["rank"] + queued - 1
                    try:
                        profiles = {
                            "interactive": ctrl.residency_stats.profile_for(
                                course_id, lane_name, batch=False),
                            "batch": ctrl.residency_stats.profile_for(
                                course_id, lane_name, batch=True),
                        }
                        tail_est = estimate_wait(
                            queue_rank=tail_rank,
                            lane_name=lane_name,
                            running=top_info["running_pods"],
                            profiles=profiles,
                            required_units=1.0,
                            now=now,
                        )
                        tail_wait = _fmt_wait(tail_est)
                    except Exception:
                        pass

            course_lanes.append({
                "lane_name":     lane_name,
                "running_count": running,
                "queued_count":  queued,
                "top_wait":      top_wait,
                "tail_wait":     tail_wait,
            })

        if not course_lanes:
            continue

        courses_out.append({
            "course_id": course_id,
            "weight":    round(course.class_weight, 4),
            "lanes":     course_lanes,
        })

    # Most-queued courses first
    courses_out.sort(
        key=lambda c: sum(ln["queued_count"] for ln in c["lanes"]),
        reverse=True,
    )

    # ------------------------------------------------------------------
    # Running-pod detail rows (debug view)
    # ------------------------------------------------------------------
    running_pods_out = []
    for lane, pods in running_snap.items():
        for uid, rp in pods.items():
            ctx       = ctx_snap.get(uid)
            course_id = ctx[0] if ctx else _NO_COURSE
            batch     = bool(ctx[2]) if ctx else rp.batch
            namespace = student_snap.get(uid, "")
            running_s = max(0, round(now - rp.start_time))
            running_pods_out.append({
                "uid":            uid,
                "namespace":      namespace,
                "lane":           lane,
                "resource_units": rp.resource_units,
                "course_id":      course_id if course_id != _NO_COURSE else None,
                "batch":          batch,
                "running_s":      running_s,
                "deadline_s":     round(rp.active_deadline_seconds),
            })
    running_pods_out.sort(key=lambda p: (p["lane"], -p["running_s"]))

    cache_age = ctrl.wait_cache.snapshot_age()

    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "system": {
            "cycle_interval_s":      ctrl.cycle_interval,
            "wait_cache_age_s":      round(cache_age, 1) if cache_age is not None else None,
            "wait_cache_duration_s": round(ctrl.wait_cache.last_duration, 2),
            "course_count":          len(ctrl.registry),
            "pending_count":         pending_count,
        },
        "lanes":        lanes_out,
        "courses":      courses_out,
        "running_pods": running_pods_out,
    }
