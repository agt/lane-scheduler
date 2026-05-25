"""
Cluster Priority Scheduler
Weighted fair-share scheduling across heterogeneous resource lanes,
with per-class weight-based priority, wait-time aging,
and within-class fair scheduling via minimum-running-then-oldest-job ordering.

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

    Callers that need a runtime fallback (e.g. based on current capacity)
    should call best_fallback_gpu_lane(lane_capacity) separately.
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
    """
    Return the GPU lane with the most allocatable capacity.
    Lexically earliest lane name breaks ties.
    Returns None if no GPU lanes are known.
    """
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
class CourseClass:
    """A class section registered with the scheduler."""
    class_id:     str
    class_weight: float  # scheduling weight supplied directly from the course CSV


@dataclass
class Job:
    """A single schedulable unit submitted by a student."""
    job_id:        str
    class_id:      str
    student_id:    str
    lane:          str
    batch:         bool  = False
    submit_time:   float = field(default_factory=time.monotonic)
    dispatch_time: Optional[float] = field(default=None, repr=False)
    complete_time: Optional[float] = field(default=None, repr=False)
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

    def score(self, job: Job, course: CourseClass,
              utilization: float, now: float) -> float:
        """
        P(j, l) = W(c) × Mode(j) × Age(j) / U(c, l)
        Higher score → higher priority.
        """
        u = max(utilization, EPSILON)
        return (course.class_weight
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
    - Per-class scheduling weights supplied externally (from course CSV)
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

        # Per-lane per-student count of currently running pods, set each cycle
        # by the controller before calling cycle().  Used by _top_student().
        self._running_counts: dict = {}  # {lane: {student_id: count}}

        # Per-lane per-class running resource units, pushed each cycle by the
        # controller from its live _running dict.  Used to compute U(c, lane).
        self._running_utilization: dict = {}  # {lane: {class_id: float}}

        # Single re-entrant lock protects _classes, _queues, _running_counts,
        # and _running_utilization.  Acquisition order: any controller lock
        # (_pending_lock, _admitted_lock, _running_lock, _running_ctx_lock) is
        # acquired BEFORE Scheduler._lock, never after.
        self._lock = threading.RLock()

        # {lane: {class_id: {student_id: [Job, ...]}}}
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
        Replace the per-lane per-student running-pod count snapshot.

        Called by the controller at the start of each cycle, before cycle().
        counts: {lane_str: {student_id: int}}
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
        Return the highest-priority student among *active_students* for *lane*.

        Priority rules (applied in order):
          1. Fewest currently running pods in this lane.
          2. Oldest pending job submit_time (FIFO among equally-loaded students).
        """
        running_in_lane = self._running_counts.get(lane, {})
        min_running = min(running_in_lane.get(s, 0) for s in active_students)
        candidates = [s for s in active_students
                      if running_in_lane.get(s, 0) == min_running]
        return min(candidates, key=lambda s: student_map[s][0].submit_time)

    def _iter_top_candidates(
        self, lane: str, now: float
    ) -> Iterator[tuple[str, float, Job, CourseClass]]:
        """Yield (class_id, score, job, course) for the top student of each active
        class in *lane*.  Must be called while holding self._lock."""
        for class_id, student_map in self._queues[lane].items():
            course          = self._classes[class_id]
            active_students = {s for s, jobs in student_map.items() if jobs}
            if not active_students:
                continue
            student_id   = self._top_student(active_students, lane, student_map)
            job          = student_map[student_id][0]
            running      = self._running_utilization.get(lane, {}).get(class_id, 0.0)
            cap          = self.lane_capacity.get(lane, 1.0)
            util         = running / cap if cap > 0 else 0.0
            score        = self.scorer.score(job, course, util, now)
            yield class_id, score, job, course

    def register_class(self, course: CourseClass) -> None:
        with self._lock:
            self._classes[course.class_id] = course
        logger.info(
            "Registered class %s (weight=%.4f)",
            course.class_id, course.class_weight,
        )

    def has_class(self, class_id: str) -> bool:
        with self._lock:
            return class_id in self._classes

    def submit(self, job: Job) -> None:
        with self._lock:
            if job.class_id not in self._classes:
                raise ValueError(f"Unknown class_id: {job.class_id!r}")
            self._queues[job.lane][job.class_id][job.student_id].append(job)
        logger.debug(
            "Queued job %s [lane=%s student=%s class=%s]",
            job.job_id, job.lane,
            job.student_id, job.class_id,
        )

    def remove_job(self, job_id: str) -> Optional[Job]:
        """
        Remove the queued job whose job_id matches and return it, or None.

        Walks all lanes; intended for handling pod DELETED events so an
        orphan Job doesn't sit in the scheduler queue after the pod is gone.
        Does not touch utilization tracking (the job never ran).
        """
        with self._lock:
            for lane in Lane:
                lane_queue = self._queues[lane]
                for class_id, student_map in list(lane_queue.items()):
                    for student_id, jobs in list(student_map.items()):
                        for idx, job in enumerate(jobs):
                            if job.job_id == job_id:
                                del jobs[idx]
                                if not jobs:
                                    del student_map[student_id]
                                if not student_map:
                                    del lane_queue[class_id]
                                return job
        return None

    def cycle(self, now: Optional[float] = None) -> list[Job]:
        """
        Run one scheduling cycle.  Returns dispatched jobs.

        1. Per lane: select one candidate per class via _top_student().
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

                # Step 1 — one candidate per class
                candidates: list[tuple[float, Job, CourseClass]] = [
                    (score, job, course)
                    for _, score, job, course in self._iter_top_candidates(lane, now)
                ]

                if not candidates:
                    continue

                # Step 2 — dispatch top-K
                candidates.sort(key=lambda x: x[0], reverse=True)
                for score, job, course in candidates[: self.config.dispatch_k]:
                    job.dispatch_time = now
                    student_jobs = self._queues[lane][job.class_id][job.student_id]
                    student_jobs.pop(0)
                    if not student_jobs:
                        del self._queues[lane][job.class_id][job.student_id]
                    if not self._queues[lane][job.class_id]:
                        del self._queues[lane][job.class_id]

                    dispatched.append(job)
                    log_entries.append((job, lane, score))

        # Logging outside the lock — formatting is non-trivial and the
        # dispatched job objects we hold are safe to read after release.
        for job, lane, score in log_entries:
            mode = "batch" if job.batch else "interactive"
            logger.info(
                "Dispatched job %s [lane=%s mode=%s score=%.4f "
                "class=%s student=%s wait=%.1fs]",
                job.job_id, lane, mode, score,
                job.class_id, job.student_id, job.wait_seconds(now),
            )
        return dispatched

    def _scored_candidates(self, lane: str,
                           now: Optional[float] = None) -> list[tuple[float, Job]]:
        """
        Return all per-class top candidates for *lane*, scored and sorted
        descending.  Used by the wait-time snapshot to compute ranks in one
        pass rather than calling queue_rank() per pod (O(Q) vs O(Q²)).

        Acquires self._lock for the full scoring pass; callers on other threads
        (e.g. the wait-cache background thread) will contend with cycle() at
        the per-lane granularity.
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
        or None if the job is not found.

        Rank 1 means the job is next to be dispatched from its lane.
        This re-scores all candidates, so it should be called sparingly
        (e.g. on-demand per student request, not every cycle).
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
        """Returns {lane_name: {class_id: job_count}} for monitoring."""
        result: dict[str, dict[str, int]] = {}
        with self._lock:
            for lane in Lane:
                counts = {
                    class_id: sum(len(jobs) for jobs in student_map.values())
                    for class_id, student_map in self._queues[lane].items()
                }
                if counts:
                    result[lane] = counts
        return result

    def class_scores(self, lane: str,
                     now: Optional[float] = None) -> dict[str, float]:
        """Current top-candidate score per class in a lane.  For dashboards."""
        now = now if now is not None else time.monotonic()
        with self._lock:
            return {
                class_id: score
                for class_id, score, _, _ in self._iter_top_candidates(lane, now)
            }
