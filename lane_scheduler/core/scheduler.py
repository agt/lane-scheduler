"""
Cluster Priority Scheduler
Weighted fair-share scheduling across heterogeneous resource lanes,
with per-class tier weighting, enrollment normalization, wait-time aging,
and within-class fair scheduling via minimum-running-then-oldest-job ordering.

Lane model
~~~~~~~~~~
    One lane exists per physical GPU class discovered at controller startup,
    plus a single CPU lane for all non-GPU pods.  Lanes are plain strings:
    "cpu" for the CPU lane, "gpu-<class>" for each GPU class (e.g. "gpu-small").
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
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lane strings — built dynamically at startup
# ---------------------------------------------------------------------------

CPU_LANE = "cpu"

# Populated by initialise_lanes().
Lane:       Optional[frozenset] = None   # all lane strings: {"cpu", "gpu-small", …}
GPU_LANES:  frozenset           = frozenset()  # Lane minus CPU_LANE

# Internal set: known raw gpu-class strings (e.g. "small", "medium")
_KNOWN_GPU_CLASSES: frozenset = frozenset()


def initialise_lanes(gpu_classes: list[str]) -> frozenset:
    """
    Build the lane string set from *gpu_classes* and populate all derived
    module globals.  Must be called before any Scheduler is constructed.

    Returns the full lane frozenset for convenience.
    Duplicate and empty class strings are silently dropped.
    """
    global Lane, GPU_LANES, _KNOWN_GPU_CLASSES

    unique             = frozenset(c.strip().lower() for c in gpu_classes if c.strip())
    _KNOWN_GPU_CLASSES = unique
    GPU_LANES          = frozenset(f"gpu-{c}" for c in unique)
    Lane               = frozenset([CPU_LANE]) | GPU_LANES

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
    gpu_lanes = sorted(lane for lane in lane_capacity if lane != CPU_LANE)
    if not gpu_lanes:
        return None
    return max(gpu_lanes, key=lambda lane: lane_capacity.get(lane, 0.0))


def is_known_gpu_class(gpu_class: str) -> bool:
    """Return True if gpu_class is in the current known set."""
    return gpu_class.strip().lower() in _KNOWN_GPU_CLASSES


# ---------------------------------------------------------------------------
# Tier
# ---------------------------------------------------------------------------

class Tier(IntEnum):
    INTRO     = 1
    UPPER_DIV = 2
    GRAD      = 3


# ---------------------------------------------------------------------------
# Tuning defaults
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    alpha              = 1.0,    # urgency scaling factor
    t_half_interactive = 600.0,  # 10 min — interactive job aging half-life (s)
    t_half_batch       = 7200.0, # 2 hr  — batch job aging half-life (s)
    batch_mode_penalty = 0.3,    # batch jobs score at 30% of interactive baseline
    epsilon            = 0.01,   # utilization floor to prevent div-by-zero
    utilization_window = 300.0,  # rolling utilization window (s)
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
    epsilon:            float = DEFAULTS["epsilon"]
    utilization_window: float = DEFAULTS["utilization_window"]
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
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.utilization_window <= 0:
            raise ValueError(f"utilization_window must be > 0, got {self.utilization_window}")
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
    class_id:   str
    tier:       Tier
    enrollment: int

    @property
    def tier_weight(self) -> float:
        return float(self.tier.value)   # 1.0 / 2.0 / 3.0

    @property
    def class_weight(self) -> float:
        """W(c) = tier_weight / sqrt(enrollment)"""
        return self.tier_weight / math.sqrt(max(1, self.enrollment))


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

class UtilizationTracker:
    """
    Tracks resource consumption per (class_id, lane) over a rolling window.
    Usage events are timestamped; expired events are purged on read.
    """

    def __init__(self, window: float, lane_capacity: dict) -> None:
        self._window = window
        self._lane_capacity = lane_capacity
        self._events: dict[tuple, list] = defaultdict(list)

    def record(self, class_id: str, lane: str, units: float,
               now: Optional[float] = None) -> None:
        ts = now if now is not None else time.monotonic()
        self._events[(class_id, lane)].append((ts, units))

    def utilization(self, class_id: str, lane: str,
                    now: Optional[float] = None) -> float:
        ts     = now if now is not None else time.monotonic()
        cutoff = ts - self._window
        key    = (class_id, lane)
        filtered = [(t, u) for t, u in self._events[key] if t >= cutoff]
        if filtered:
            self._events[key] = filtered
        else:
            # Drop the key entirely to bound long-term memory growth
            self._events.pop(key, None)
        total    = sum(u for _, u in filtered)
        capacity = self._lane_capacity.get(lane, 1.0)
        return total / (capacity * self._window) if capacity > 0 else 0.0

    def reset(self, class_id: str, lane: str) -> None:
        self._events.pop((class_id, lane), None)




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
        u = max(utilization, self.config.epsilon)
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
    - Per-lane independent queues (one per GPU class + CPU)
    - Tier-weighted, enrollment-normalized class weights
    - Log-aging wait boost with batch/interactive half-lives
    - Batch mode penalty keeps interactive jobs preferred
    - Fewest-running-then-oldest-submit student ordering within each class
    - Rolling-window utilization tracking
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
        self.util    = UtilizationTracker(self.config.utilization_window, lane_capacity)

        # Per-lane per-student count of currently running pods, set each cycle
        # by the controller before calling cycle().  Used by _top_student().
        self._running_counts: dict = {}  # {lane: {student_id: count}}

        # Single re-entrant lock protects _classes, _queues, _running_counts,
        # and the internal state of self.util (which is not accessed from outside
        # Scheduler).  Acquisition order: any controller lock (_pending_lock,
        # _admitted_lock, _running_lock, _running_ctx_lock) is acquired BEFORE
        # Scheduler._lock, never after.
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
            self.util._lane_capacity = lane_capacity

    def update_running_counts(self, counts: dict) -> None:
        """
        Replace the per-lane per-student running-pod count snapshot.

        Called by the controller at the start of each cycle, before cycle().
        counts: {lane_str: {student_id: int}}
        """
        with self._lock:
            self._running_counts = counts

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

    def register_class(self, course: CourseClass) -> None:
        with self._lock:
            self._classes[course.class_id] = course
        logger.info(
            "Registered class %s (tier=%s enrollment=%d weight=%.4f)",
            course.class_id, course.tier.name,
            course.enrollment, course.class_weight,
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
        3. Update utilization tracker.
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
                candidates: list[tuple[float, Job, CourseClass]] = []
                for class_id, student_map in lane_queue.items():
                    course          = self._classes[class_id]
                    active_students = {s for s, jobs in student_map.items() if jobs}
                    if not active_students:
                        continue
                    student_id = self._top_student(active_students, lane, student_map)
                    job        = student_map[student_id][0]
                    util       = self.util.utilization(class_id, lane, now)
                    score      = self.scorer.score(job, course, util, now)
                    candidates.append((score, job, course))

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

                    self.util.record(job.class_id, lane, job.resource_units, now)
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
        """
        now = now if now is not None else time.monotonic()
        with self._lock:
            candidates: list[tuple[float, Job]] = []
            for class_id, student_map in self._queues[lane].items():
                course          = self._classes[class_id]
                active_students = {s for s, jobs in student_map.items() if jobs}
                if not active_students:
                    continue
                student_id = self._top_student(active_students, lane, student_map)
                job        = student_map[student_id][0]
                util       = self.util.utilization(class_id, lane, now)
                score      = self.scorer.score(job, course, util, now)
                candidates.append((score, job))
            candidates.sort(key=lambda x: x[0], reverse=True)
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
            candidates: list[tuple[float, str]] = []   # (score, job_uid)

            for class_id, student_map in self._queues[lane].items():
                course          = self._classes[class_id]
                active_students = {s for s, jobs in student_map.items() if jobs}
                if not active_students:
                    continue
                student_id = self._top_student(active_students, lane, student_map)
                job        = student_map[student_id][0]
                util       = self.util.utilization(class_id, lane, now)
                score      = self.scorer.score(job, course, util, now)
                candidates.append((score, job.job_id))

            candidates.sort(key=lambda x: x[0], reverse=True)
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
        now    = now if now is not None else time.monotonic()
        scores = {}
        with self._lock:
            for class_id, student_map in self._queues[lane].items():
                course  = self._classes[class_id]
                active  = {s for s, jobs in student_map.items() if jobs}
                if not active:
                    continue
                student_id = self._top_student(active, lane, student_map)
                job        = student_map[student_id][0]
                util       = self.util.utilization(class_id, lane, now)
                scores[class_id] = self.scorer.score(job, course, util, now)
        return scores
