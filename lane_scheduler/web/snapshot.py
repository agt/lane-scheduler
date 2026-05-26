"""
Queue-snapshot aggregator for the web dashboard.

build_snapshot() reads controller state (read-only, respecting existing locks)
and returns a JSON-serializable dict covering:
  - lanes        : capacity, utilization, queue depth, drain estimate
  - sched_groups : per-(sched_group, lane) running/queued counts and wait estimates
  - system       : health / operational metadata

No user identifiers (namespaces / pod UIDs) appear in the output.
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

_NO_SCHED_GROUP = "__unlabelled__"


def _fmt_wait(est) -> Optional[dict]:
    if est is None:
        return None
    return {
        "median_s": round(est.median_seconds, 1),
        "p20_s":    round(est.p20_seconds, 1),
        "p80_s":    round(est.p80_seconds, 1),
        "rank":     est.queue_rank,
    }


def build_nodes_snapshot(ctrl: "LaneSchedulerController") -> dict:
    """
    Dump the live NodeCapacityTracker state for debugging.
    Returns every node the tracker knows about (managed GPU pool only),
    regardless of readiness.
    """
    now_str = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()
    with ctrl.node_tracker._lock:
        nodes = [
            {
                "name":        info.name,
                "lane":        info.lane,
                "gpu_count":   info.gpu_count,
                "ready":       info.ready,
                "schedulable": info.schedulable,
                "active":      info.active,
            }
            for info in sorted(
                ctrl.node_tracker._nodes.values(), key=lambda n: n.name
            )
        ]
    return {"generated_at": now_str, "nodes": nodes}


def build_snapshot(ctrl: "LaneSchedulerController") -> dict:
    """
    Aggregate controller state into a JSON-serializable snapshot.

    Accesses controller internals read-only using the same locking discipline
    as _build_wait_snapshot().  Safe to call from any thread concurrently
    with the scheduling cycle.
    """
    from lane_scheduler.core.scheduler import Lane
    from lane_scheduler.estimation.wait_estimator import (
        estimate_soonest_completion,
        estimate_wait,
    )

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
            for lane, pods in ctrl._kubernetes_running.items()
        }
        user_snap = dict(ctrl._kubernetes_running_user)

    with ctrl._kubernetes_pending_lock:
        pending_snap = {
            lane: dict(pods)
            for lane, pods in ctrl._kubernetes_pending.items()
        }

    with ctrl._running_ctx_lock:
        # uid → (sched_group_id, lane_name, batch, deadline_seconds, created_at)
        ctx_snap = dict(ctrl._running_ctx)

    with ctrl._admitted_resources_lock:
        admitted_by_lane: dict = {}
        for _uid, (lane, units) in ctrl._admitted_resources.items():
            admitted_by_lane[lane] = admitted_by_lane.get(lane, 0.0) + units

    pending_units_by_lane: dict = {}
    for lane, pods in pending_snap.items():
        for rp in pods.values():
            pending_units_by_lane[lane] = (
                pending_units_by_lane.get(lane, 0.0) + rp.resource_units
            )

    queue_depths = ctrl.scheduler.queue_depths()    # {lane_name: {sched_group_id: count}}
    cached_ests  = ctrl.wait_cache.all_estimates()  # {pod_uid: WaitEstimate}

    # ------------------------------------------------------------------
    # Running-pod aggregates
    # ------------------------------------------------------------------
    running_by_sched_group_lane: dict[tuple, int] = defaultdict(int)
    running_units_by_lane: dict[str, float]       = defaultdict(float)
    running_count_by_lane: dict[str, int]         = defaultdict(int)

    for lane, pods in running_snap.items():
        lane_name = lane
        for _uid, rp in pods.items():
            running_units_by_lane[lane_name] += rp.resource_units
            running_count_by_lane[lane_name] += 1

    for uid, ctx in ctx_snap.items():
        sched_group_id, lane_name = ctx[0], ctx[1]
        running_by_sched_group_lane[(sched_group_id, lane_name)] += 1

    # ------------------------------------------------------------------
    # Scored candidates per lane
    # ------------------------------------------------------------------
    group_top: dict[tuple, dict] = {}
    for lane in sorted(Lane or []):
        lane_name    = lane
        running_pods = list(running_snap.get(lane, {}).values())
        try:
            candidates = ctrl.scheduler._scored_candidates(lane, now)
        except Exception:
            candidates = []
        for rank, (score, job) in enumerate(candidates, start=1):
            key = (lane_name, job.sched_group_id)
            group_top[key] = {
                "rank":         rank,
                "job":          job,
                "running_pods": running_pods,
                "cached_est":   cached_ests.get(job.job_id),
            }

    # ------------------------------------------------------------------
    # Lane rows
    # ------------------------------------------------------------------
    lanes_out = []
    for lane in sorted(Lane or []):
        lane_name    = lane
        capacity     = lane_capacity.get(lane, 0.0)
        nodes        = node_counts.get(lane, 0)
        run_units    = running_units_by_lane.get(lane_name, 0.0)
        run_count    = running_count_by_lane.get(lane_name, 0)
        queued_total = sum(queue_depths.get(lane_name, {}).values())

        running_pods = list(running_snap.get(lane, {}).values())
        drain_p80_s: Optional[float] = None
        soonest_completion: Optional[dict] = None

        lane_profiles = None
        if running_pods or queued_total > 0:
            try:
                lane_profiles = {
                    "interactive": ctrl.residency_stats.profile_for(
                        _NO_SCHED_GROUP, lane_name, batch=False),
                    "batch": ctrl.residency_stats.profile_for(
                        _NO_SCHED_GROUP, lane_name, batch=True),
                }
            except Exception:
                pass

        if running_pods and lane_profiles is not None:
            try:
                sc_est = estimate_soonest_completion(
                    lane_name=lane_name,
                    running=running_pods,
                    profiles=lane_profiles,
                    now=now,
                )
                soonest_completion = _fmt_wait(sc_est)
            except Exception:
                pass

        if queued_total > 0 and lane_profiles is not None:
            k8s_pend_u = pending_units_by_lane.get(lane, 0.0)
            free_u     = max(
                0.0,
                capacity - run_units - k8s_pend_u - admitted_by_lane.get(lane, 0.0)
            )
            try:
                drain_est = estimate_wait(
                    queue_rank=queued_total,
                    lane_name=lane_name,
                    running=running_pods,
                    profiles=lane_profiles,
                    required_units=1.0,
                    free_units=free_u,
                    now=now,
                )
                drain_p80_s = round(drain_est.p80_seconds, 1)
            except Exception:
                pass

        lanes_out.append({
            "name":               lane_name,
            "node_count":         nodes,
            "capacity_units":     round(capacity, 1),
            "running_count":      run_count,
            "running_units":      round(run_units, 1),
            "queued_count":       queued_total,
            "drain_p80_s":        drain_p80_s,
            "soonest_completion": soonest_completion,
        })

    # ------------------------------------------------------------------
    # Scheduling-group rows
    # ------------------------------------------------------------------
    all_sched_group_ids: set[str] = set()
    for lane_name, gm in queue_depths.items():
        all_sched_group_ids.update(gm.keys())
    for (sched_group_id, _) in running_by_sched_group_lane:
        all_sched_group_ids.add(sched_group_id)
    all_sched_group_ids.discard(_NO_SCHED_GROUP)

    sched_groups_out = []
    for sched_group_id in sorted(all_sched_group_ids):
        group = ctrl.registry.get(sched_group_id)
        group_lanes = []

        for lane in sorted(Lane or []):
            lane_name = lane
            queued  = queue_depths.get(lane_name, {}).get(sched_group_id, 0)
            running = running_by_sched_group_lane.get((sched_group_id, lane_name), 0)
            if queued == 0 and running == 0:
                continue

            top_info  = group_top.get((lane_name, sched_group_id))
            top_wait  = None
            tail_wait = None

            if top_info:
                top_wait = _fmt_wait(top_info["cached_est"])

                if queued > 1:
                    tail_rank  = top_info["rank"] + queued - 1
                    run_u      = running_units_by_lane.get(lane_name, 0.0)
                    cap        = lane_capacity.get(lane, 0.0)
                    k8s_pend_u = pending_units_by_lane.get(lane, 0.0)
                    free_u     = max(
                        0.0,
                        cap - run_u - k8s_pend_u - admitted_by_lane.get(lane, 0.0)
                    )
                    try:
                        profiles = {
                            "interactive": ctrl.residency_stats.profile_for(
                                sched_group_id, lane_name, batch=False),
                            "batch": ctrl.residency_stats.profile_for(
                                sched_group_id, lane_name, batch=True),
                        }
                        tail_est = estimate_wait(
                            queue_rank=tail_rank,
                            lane_name=lane_name,
                            running=top_info["running_pods"],
                            profiles=profiles,
                            required_units=1.0,
                            free_units=free_u,
                            now=now,
                        )
                        tail_wait = _fmt_wait(tail_est)
                    except Exception:
                        pass

            group_lanes.append({
                "lane_name":     lane_name,
                "running_count": running,
                "queued_count":  queued,
                "top_wait":      top_wait,
                "tail_wait":     tail_wait,
            })

        if not group_lanes:
            continue

        sched_groups_out.append({
            "sched_group_id": sched_group_id,
            "weight":         round(group.weight, 4),
            "lanes":          group_lanes,
        })

    # Most-queued scheduling groups first
    sched_groups_out.sort(
        key=lambda g: sum(ln["queued_count"] for ln in g["lanes"]),
        reverse=True,
    )

    # ------------------------------------------------------------------
    # Running-pod detail rows (debug view)
    # ------------------------------------------------------------------
    running_pods_out = []
    for lane, pods in running_snap.items():
        for uid, rp in pods.items():
            ctx            = ctx_snap.get(uid)
            sched_group_id = ctx[0] if ctx else _NO_SCHED_GROUP
            batch          = bool(ctx[2]) if ctx else rp.batch
            username       = user_snap.get(uid, "")
            running_s      = max(0, round(now - rp.start_time))
            running_pods_out.append({
                "uid":            uid,
                "username":       username,
                "lane":           lane,
                "resource_units": rp.resource_units,
                "sched_group_id": sched_group_id if sched_group_id != _NO_SCHED_GROUP else None,
                "batch":          batch,
                "running_s":      running_s,
                "deadline_s":     round(rp.active_deadline_seconds),
            })
    running_pods_out.sort(key=lambda p: (p["lane"], -p["running_s"]))

    cache_age         = ctrl.wait_cache.snapshot_age()
    sched_group_count = len(all_sched_group_ids)

    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "system": {
            "cycle_interval_s":       ctrl.cycle_interval,
            "wait_cache_age_s":       round(cache_age, 1) if cache_age is not None else None,
            "wait_cache_duration_s":  round(ctrl.wait_cache.last_duration, 2),
            "sched_group_count":      sched_group_count,
            "pending_count":          pending_count,
        },
        "lanes":        lanes_out,
        "sched_groups": sched_groups_out,
        "running_pods": running_pods_out,
    }
