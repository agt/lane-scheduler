"""
Cluster Priority Scheduler
Weighted fair-share scheduling across heterogeneous resource lanes,
with per-group weight-based priority, wait-time aging,
and within-group per-user fairness via minimum-running-then-oldest-job ordering.

Lane model
~~~~~~~~~~
    One lane exists per physical GPU class discovered at controller startup.
    Lanes are plain strings: "gpu-<class>" for each GPU class (e.g. "gpu-small").
    initialise_lanes() must be called before any Scheduler is constructed.

    Batch vs interactive is a *scoring modifier*, not a lane split.
    Batch jobs receive a Mode penalty (default 0.3) so interactive jobs
    are preferred within the same gpu-class lane, but batch jobs age up
    and drain overnight when interactive load falls away.

Startup contract
~~~~~~~~~~~~~~~~
    controller.py MUST call initialise_lanes(gpu_classes) before constructing
    any Scheduler.  All other modules import lane helpers from here.
"""

from __future__ import annotations

import math
import threading
import time
import logging
from dataclasses import dataclass, field
from collections import defaultdict

from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lane strings — built dynamically at startup
# ---------------------------------------------------------------------------

# Populated by initialise_lanes().
Lane: Optional[frozenset] = None   # all lane strings: {"gpu-small", …}

# Internal set: known raw gpu-class strings (e.g. "small", "medium")
_KNOWN_GPU_CLASSES: frozenset = frozenset()


def initialise_lanes(gpu_classes: list[str]) -> frozenset:
    """
    Build the lane string set from *gpu_classes* and populate all derived
    module globals.  Must be called before any Scheduler is constructed.

    Returns the full lane frozenset for convenience.
    Duplicate and empty class strings are silently dropped.
    """
    global Lane, _KNOWN_GPU_CLASSES

    unique             = frozenset(c.strip().lower() for c in gpu_classes if c.strip())
    _KNOWN_GPU_CLASSES = unique
    Lane               = frozenset(f"gpu-{c}" for c in unique)

    logger.info("Lanes initialised: %s", ", ".join(sorted(Lane)))
    return Lane


def lane_for_gpu_class(gpu_class: str, strict: bool = False) -> Optional[str]:
    """
    Return the lane string for *gpu_class* (case-insensitive), or None.

    If the class is unrecognised:
      - strict=True  → return None silently
      - strict=False → return None with a warning logged
    """
    if Lane is None:
        raise RuntimeError("initialise_lanes() has not been called")

    key = gpu_class.strip().lower()
    if key in _KNOWN_GPU_CLASSES:
        return f"gpu-{key}"

    if not strict:
        logger.warning(
            "Unrecognised gpu-class %r. "
            "Controller restart required to pick up new GPU classes.",
            gpu_class,
        )
    return None


def best_fallback_gpu_lane(lane_capacity: dict) -> Optional[str]:
    """Return the GPU lane with the most allocatable capacity."""
    gpu_lanes = sorted(lane_capacity)
    if not gpu_lanes:
        return None
    return max(gpu_lanes, key=lambda lane: lane_capacity.get(lane, 0.0))


def is_known_gpu_class(gpu_class: str) -> bool:
    """Return True if gpu_class is in the current known set."""
    return gpu_class.strip().lower() in _KNOWN_GPU_CLASSES


# ---------------------------------------------------------------------------
# Tuning defaults
# ---------------------------------------------------------------------------

EPSILON = 0.01  # utilization floor to prevent div-by-zero; not operator-tunable

DEFAULTS = dict(
    alpha              = 1.0,    # urgency scaling factor
    t_half_interactive = 600.0,  # 10 min — interactive job aging half-life (s)
    t_half_batch       = 7200.0, # 2 hr  — batch job aging half-life (s)
    batch_mode_penalty = 0.3,    # batch jobs score at 30% of interactive baseline
    dispatch_k         = 8,      # max jobs dispatched per lane per cycle
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    alpha:              float = DEFAULTS["alpha"]
    t_half_interactive: float = DEFAULTS["t_half_interactive"]
    t_half_batch:       float = DEFAULTS["t_half_batch"]
    batch_mode_penalty: float = DEFAULTS["batch_mode_penalty"]
    dispatch_k:         int   = DEFAULTS["dispatch_k"]

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.t_half_interactive <= 0:
            raise ValueError(f"t_half_interactive must be > 0, got {self.t_half_interactive}")
        if self.t_half_batch <= 0:
            raise ValueError(f"t_half_batch must be > 0, got {self.t_half_batch}")
        if self.batch_mode_penalty <= 0:
            raise ValueError(f"batch_mode_penalty must be > 0, got {self.batch_mode_penalty}")
        if self.dispatch_k <= 0:
            raise ValueError(f"dispatch_k must be > 0, got {self.dispatch_k}")

    def t_half(self, batch: bool) -> float:
        return self.t_half_batch if batch else self.t_half_interactive

    def mode_weight(self, batch: bool) -> float:
        return self.batch_mode_penalty if batch else 1.0


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------

@dataclass
class SchedGroup:
    """A scheduling group registered with the scheduler."""
    sched_group_id: str
    weight:         float  # scheduling weight (default 1.0 from SchedGroupRegistry)


@dataclass
class Job:
    """A single schedulable unit submitted by a user."""
    job_id:         str
    sched_group_id: str
    username:       str
    lane:           str
    batch:          bool  = False
    submit_time:    float = field(default_factory=time.monotonic)
    dispatch_time:  Optional[float] = field(default=None, repr=False)
    complete_time:  Optional[float] = field(default=None, repr=False)
    resource_units: float = 1.0

    def wait_seconds(self, now: Optional[float] = None) -> float:
        t = now if now is not None else time.monotonic()
        return t - self.submit_time


# ---------------------------------------------------------------------------
# Utilization tracker (rolling window)
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------

class PriorityScorer:

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config

    def age_boost(self, job: Job, now: float) -> float:
        """Age(j) = 1 + α × log(1 + wait / t_half)"""
        wait   = job.wait_seconds(now)
        t_half = self.config.t_half(job.batch)
        return 1.0 + self.config.alpha * math.log1p(wait / t_half)

    def score(self, job: Job, group: SchedGroup,
              utilization: float, now: float) -> float:
        """
        P(j, l) = W(g) × Mode(j) × Age(j) / U(g, l)
        Higher score → higher priority.
        """
        u = max(utilization, EPSILON)
        return (group.weight
                * self.config.mode_weight(job.batch)
                * self.age_boost(job, now)
                / u)


# ---------------------------------------------------------------------------
# Main Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """
    Priority scheduler with:
    - Per-lane independent queues (one per GPU class)
    - Per-group scheduling weights (supplied by SchedGroupRegistry; default 1.0)
    - Log-aging wait boost with batch/interactive half-lives
    - Batch mode penalty keeps interactive jobs preferred
    - Fewest-running-then-oldest-submit student ordering within each class
    - Live utilization derived from running pod resource units
    """

    def __init__(
        self,
        lane_capacity: dict,
        config: Optional[SchedulerConfig] = None,
    ) -> None:
        if Lane is None:
            raise RuntimeError("initialise_lanes() must be called before Scheduler()")
        self.lane_capacity = lane_capacity
        self.config  = config or SchedulerConfig()
        self.scorer  = PriorityScorer(self.config)
        self._classes: dict[str, CourseClass] = {}

        # Per-lane per-user count of currently running pods, set each cycle
        # by the controller before calling cycle().  Used by _top_user().
        self._running_counts: dict = {}  # {lane: {username: count}}

        # Per-lane per-class running resource units, pushed each cycle by the
        # controller from its live _running dict.  Used to compute U(c, lane).
        self._running_utilization: dict = {}  # {lane: {class_id: float}}

        # Single re-entrant lock protects _classes, _queues, _running_counts,
        # and _running_utilization.  Acquisition order: any controller lock
        # (_pending_lock, _admitted_lock, _running_lock, _running_ctx_lock) is
        # acquired BEFORE Scheduler._lock, never after.
        self._lock = threading.RLock()

        # {lane: {sched_group_id: {username: [Job, ...]}}}
        self._queues: dict = {
            lane: defaultdict(lambda: defaultdict(list))
            for lane in Lane
        }

    def set_lane_capacity(self, lane_capacity: dict) -> None:
        """Atomically replace the lane-capacity view used for scoring."""
        with self._lock:
            self.lane_capacity = lane_capacity

    def update_running_counts(self, counts: dict) -> None:
        """
        Replace the per-lane per-user running-pod count snapshot.
        counts: {lane_str: {username: int}}
        Called by the controller at the start of each cycle, before cycle().
        """
        with self._lock:
            self._running_counts = counts

    def update_running_utilization(self, utilization: dict) -> None:
        """
        Replace the per-lane per-class running resource units snapshot.

        Called by the controller at the start of each cycle, before cycle().
        utilization: {lane: {class_id: total_resource_units}}
        """
        with self._lock:
            self._running_utilization = utilization

    def _top_student(self, active_students: set[str], lane: str,
                     student_map: dict) -> str:
        """
        Return the highest-priority user among *active_users* for *lane*.

        Priority rules (applied in order):
          1. Fewest currently running pods in this lane.
          2. Oldest pending job submit_time (FIFO among equally-loaded users).
        """
        running_in_lane = self._running_counts.get(lane, {})
        min_running = min(running_in_lane.get(u, 0) for u in active_users)
        candidates = [u for u in active_users
                      if running_in_lane.get(u, 0) == min_running]
        return min(candidates, key=lambda u: user_map[u][0].submit_time)

    def _iter_top_candidates(
        self, lane: str, now: float
    ) -> Iterator[tuple[str, float, Job, SchedGroup]]:
        """Yield (sched_group_id, score, job, group) for the top user of each
        active group in *lane*.  Must be called while holding self._lock."""
        for sched_group_id, user_map in self._queues[lane].items():
            group        = self._groups[sched_group_id]
            active_users = {u for u, jobs in user_map.items() if jobs}
            if not active_users:
                continue
            student_id   = self._top_student(active_students, lane, student_map)
            job          = student_map[student_id][0]
            running      = self._running_utilization.get(lane, {}).get(class_id, 0.0)
            cap          = self.lane_capacity.get(lane, 1.0)
            util         = running / cap if cap > 0 else 0.0
            score        = self.scorer.score(job, course, util, now)
            yield class_id, score, job, course

    def register_group(self, group: SchedGroup) -> None:
        with self._lock:
            self._groups[group.sched_group_id] = group
        logger.info(
            "Registered scheduling group %s (weight=%.4f)",
            group.sched_group_id, group.weight,
        )

    def has_group(self, sched_group_id: str) -> bool:
        with self._lock:
            return sched_group_id in self._groups

    def submit(self, job: Job) -> None:
        with self._lock:
            if job.sched_group_id not in self._groups:
                raise ValueError(f"Unknown sched_group_id: {job.sched_group_id!r}")
            self._queues[job.lane][job.sched_group_id][job.username].append(job)
        logger.debug(
            "Queued job %s [lane=%s user=%s group=%s]",
            job.job_id, job.lane, job.username, job.sched_group_id,
        )

    def remove_job(self, job_id: str) -> Optional[Job]:
        """
        Remove the queued job whose job_id matches and return it, or None.
        Walks all lanes; used for pod DELETED events.
        """
        with self._lock:
            for lane in Lane:
                lane_queue = self._queues[lane]
                for sched_group_id, user_map in list(lane_queue.items()):
                    for username, jobs in list(user_map.items()):
                        for idx, job in enumerate(jobs):
                            if job.job_id == job_id:
                                del jobs[idx]
                                if not jobs:
                                    del user_map[username]
                                if not user_map:
                                    del lane_queue[sched_group_id]
                                return job
        return None

    def cycle(self, now: Optional[float] = None) -> list[Job]:
        """
        Run one scheduling cycle.  Returns dispatched jobs.

        1. Per lane: select one candidate per group via _top_user().
        2. Score candidates; dispatch top-K.
        """
        now = now if now is not None else time.monotonic()
        dispatched: list[Job] = []
        log_entries: list[tuple] = []

        with self._lock:
            for lane in Lane:
                lane_queue = self._queues[lane]
                if not lane_queue:
                    continue

                candidates: list[tuple[float, Job, SchedGroup]] = [
                    (score, job, group)
                    for _, score, job, group in self._iter_top_candidates(lane, now)
                ]

                if not candidates:
                    continue

                candidates.sort(key=lambda x: x[0], reverse=True)
                for score, job, group in candidates[: self.config.dispatch_k]:
                    job.dispatch_time = now
                    student_jobs = self._queues[lane][job.class_id][job.student_id]
                    student_jobs.pop(0)
                    if not student_jobs:
                        del self._queues[lane][job.class_id][job.student_id]
                    if not self._queues[lane][job.class_id]:
                        del self._queues[lane][job.class_id]

                    dispatched.append(job)
                    log_entries.append((job, lane, score))

        for job, lane, score in log_entries:
            mode = "batch" if job.batch else "interactive"
            logger.info(
                "Dispatched job %s [lane=%s mode=%s score=%.4f "
                "group=%s user=%s wait=%.1fs]",
                job.job_id, lane, mode, score,
                job.sched_group_id, job.username, job.wait_seconds(now),
            )
        return dispatched

    def _scored_candidates(self, lane: str,
                           now: Optional[float] = None) -> list[tuple[float, Job]]:
        """
        Return all per-group top candidates for *lane*, scored and sorted
        descending.  Used by the wait-time snapshot builder.
        """
        now = now if now is not None else time.monotonic()
        with self._lock:
            candidates = sorted(
                ((score, job)
                 for _, score, job, _ in self._iter_top_candidates(lane, now)),
                key=lambda x: x[0], reverse=True,
            )
            return candidates

    def queue_rank(self, job_uid: str, lane: str,
                   now: Optional[float] = None) -> Optional[int]:
        """
        Return the 1-based position of job_uid in the scored lane queue,
        or None if not found.
        """
        now = now if now is not None else time.monotonic()
        with self._lock:
            candidates = sorted(
                ((score, job.job_id)
                 for _, score, job, _ in self._iter_top_candidates(lane, now)),
                key=lambda x: x[0], reverse=True,
            )
            for rank, (_, uid) in enumerate(candidates, start=1):
                if uid == job_uid:
                    return rank
            return None

    def queue_depths(self) -> dict[str, dict[str, int]]:
        """Returns {lane_name: {sched_group_id: job_count}} for monitoring."""
        result: dict[str, dict[str, int]] = {}
        with self._lock:
            for lane in Lane:
                counts = {
                    sched_group_id: sum(len(jobs) for jobs in user_map.values())
                    for sched_group_id, user_map in self._queues[lane].items()
                }
                if counts:
                    result[lane] = counts
        return result

    def group_scores(self, lane: str,
                     now: Optional[float] = None) -> dict[str, float]:
        """Current top-candidate score per group in a lane.  For dashboards."""
        now = now if now is not None else time.monotonic()
        with self._lock:
            return {
                sched_group_id: score
                for sched_group_id, score, _, _ in self._iter_top_candidates(lane, now)
            }
