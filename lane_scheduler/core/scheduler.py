"""
Cluster Priority Scheduler
Weighted fair-share scheduling across heterogeneous resource lanes,
with per-class tier weighting, enrollment normalization, wait-time aging,
and within-class deficit round-robin.

Lane model
~~~~~~~~~~
    One lane exists per physical GPU class discovered at controller startup,
    plus a single CPU lane for all non-GPU pods.  The Lane enum is built
    dynamically via initialise_lanes(); all other modules import Lane from
    here after that call has returned.

    Batch vs interactive is a *scoring modifier*, not a lane split.
    Batch jobs receive a Mode penalty (default 0.3) so interactive jobs
    are preferred within the same gpu-class lane, but batch jobs age up
    and drain overnight when interactive load falls away.

Startup contract
~~~~~~~~~~~~~~~~
    controller.py MUST call initialise_lanes(gpu_classes) before importing
    any module that references Lane.  All other modules do:

        from scheduler import Lane, lane_for_gpu_class

    and will receive the populated enum.
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
# Lane enum — built dynamically at startup
# ---------------------------------------------------------------------------

# Populated by initialise_lanes(); referenced as a module-level name so that
# "from scheduler import Lane" works after initialisation.
Lane: type[IntEnum] = None  # type: ignore[assignment]

# Derived helpers — rebuilt by initialise_lanes() alongside Lane.
LANE_NAMES: dict = {}          # Lane member → human-readable string
GPU_LANES:  frozenset = frozenset()

# Internal map: lowercase gpu-class string → Lane member
_GPU_CLASS_TO_LANE: dict[str, IntEnum] = {}


def build_lanes(gpu_classes: list[str]) -> type[IntEnum]:
    """
    Construct and return a fresh Lane IntEnum from a list of gpu-class strings.

    CPU is always value 0.  GPU lanes are assigned values 1…N in
    lexicographic order of their class name — this is stable under additions
    (new classes sort into the sequence) but not renames.  The sort is
    intentionally lexicographic, not semantic; xsmall < xlarge alphabetically,
    which differs from size order, but integer values are internal only.

    Duplicate and empty strings are silently dropped.
    """
    unique = sorted({c.strip().lower() for c in gpu_classes if c.strip()})
    members = {"CPU": 0}
    for i, cls in enumerate(unique, start=1):
        members[f"GPU_{cls.upper()}"] = i
    return IntEnum("Lane", members)  # type: ignore[return-value]


def initialise_lanes(gpu_classes: list[str]) -> type[IntEnum]:
    """
    Build the Lane enum from *gpu_classes* and populate all derived module
    globals.  Must be called exactly once, before any other module imports Lane.

    Returns the constructed enum for convenience.
    """
    global Lane, LANE_NAMES, GPU_LANES, _GPU_CLASS_TO_LANE

    Lane = build_lanes(gpu_classes)

    LANE_NAMES = {}
    for member in Lane:
        if member.value == 0:
            LANE_NAMES[member] = "cpu"
        else:
            # GPU_XSMALL → "gpu-xsmall"
            LANE_NAMES[member] = member.name.lower().replace("_", "-", 1)

    GPU_LANES = frozenset(m for m in Lane if m.value != 0)

    _GPU_CLASS_TO_LANE = {
        member.name[4:].lower(): member   # strip "GPU_" prefix
        for member in Lane
        if member.value != 0
    }

    logger.info(
        "Lane enum initialised: %s",
        ", ".join(f"{LANE_NAMES[m]}={m.value}" for m in Lane),
    )
    return Lane


def lane_for_gpu_class(gpu_class: str, fallback_name: str = "small",
                       strict: bool = False) -> "Optional[IntEnum]":
    """
    Return the Lane member for *gpu_class* (case-insensitive).

    If the class is unrecognised:
      - strict=True  → return None (caller decides what to do; no warning logged)
      - strict=False → return the Lane for *fallback_name* if it exists,
                       otherwise the lowest-valued GPU lane, with a warning.
    """
    if Lane is None:
        raise RuntimeError("initialise_lanes() has not been called")

    key = gpu_class.strip().lower()
    lane = _GPU_CLASS_TO_LANE.get(key)
    if lane is not None:
        return lane

    if strict:
        return None

    fallback = _GPU_CLASS_TO_LANE.get(fallback_name.lower())
    if fallback is None and GPU_LANES:
        fallback = min(GPU_LANES, key=lambda m: m.value)

    logger.warning(
        "Unrecognised gpu-class %r — using lane %s. "
        "Controller restart required to pick up new GPU classes.",
        gpu_class, LANE_NAMES.get(fallback, "?"),
    )
    return fallback


def is_known_gpu_class(gpu_class: str) -> bool:
    """Return True if gpu_class maps to a lane in the current enum."""
    return gpu_class.strip().lower() in _GPU_CLASS_TO_LANE


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
    lane:          "IntEnum"   # Lane member; typed loosely for dynamic enum compat
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

    def record(self, class_id: str, lane: "IntEnum", units: float,
               now: Optional[float] = None) -> None:
        ts = now if now is not None else time.monotonic()
        self._events[(class_id, lane)].append((ts, units))

    def utilization(self, class_id: str, lane: "IntEnum",
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

    def reset(self, class_id: str, lane: "IntEnum") -> None:
        self._events.pop((class_id, lane), None)


# ---------------------------------------------------------------------------
# Deficit tracker (within-class round-robin)
# ---------------------------------------------------------------------------

class DeficitTracker:
    """
    Accrues credit to students while they have waiting jobs,
    and debits on dispatch.  Highest deficit → promoted next.
    """

    def __init__(self) -> None:
        self._deficit: dict[tuple, float] = defaultdict(float)

    def accrue(self, student_id: str, lane: "IntEnum",
               class_weight: float, dt: float) -> None:
        self._deficit[(student_id, lane)] += class_weight * dt

    def debit(self, student_id: str, lane: "IntEnum", units: float) -> None:
        self._deficit[(student_id, lane)] -= units

    def deficit(self, student_id: str, lane: "IntEnum") -> float:
        return self._deficit[(student_id, lane)]

    def top_student(self, student_ids: set[str], lane: "IntEnum") -> Optional[str]:
        if not student_ids:
            return None
        return max(student_ids, key=lambda s: self._deficit[(s, lane)])


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
    - Deficit round-robin within each class
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
        self.deficit = DeficitTracker()

        # Single re-entrant lock protects _classes, _queues, _last_cycle, and
        # the internal state of self.util and self.deficit (which are not
        # accessed from outside Scheduler).  Acquisition order: any controller
        # lock (_pending_lock, _admitted_lock, _running_lock, _running_ctx_lock)
        # is acquired BEFORE Scheduler._lock, never after.
        self._lock = threading.RLock()

        # {lane: {class_id: {student_id: [Job, ...]}}}
        self._queues: dict = {
            lane: defaultdict(lambda: defaultdict(list))
            for lane in Lane
        }
        self._last_cycle: float = time.monotonic()

    def set_lane_capacity(self, lane_capacity: dict) -> None:
        """Atomically replace the lane-capacity view used for scoring."""
        with self._lock:
            self.lane_capacity = lane_capacity
            self.util._lane_capacity = lane_capacity

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
            job.job_id, LANE_NAMES.get(job.lane, str(job.lane)),
            job.student_id, job.class_id,
        )

    def remove_job(self, job_id: str) -> Optional[Job]:
        """
        Remove the queued job whose job_id matches and return it, or None.

        Walks all lanes; intended for handling pod DELETED events so an
        orphan Job doesn't sit in the scheduler queue after the pod is gone.
        Does not touch utilization or deficit (the job never ran).
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

        1. Compute dt; accrue deficits for students with waiting jobs.
        2. Per lane: select one candidate per class via deficit round-robin.
        3. Score candidates; dispatch top-K.
        4. Update utilization and deficit trackers.
        """
        now = now if now is not None else time.monotonic()
        dispatched: list[Job] = []
        log_entries: list[tuple] = []

        with self._lock:
            dt  = now - self._last_cycle
            self._last_cycle = now

            for lane in Lane:
                lane_queue = self._queues[lane]
                if not lane_queue:
                    continue

                # Step 1 — accrue deficits
                for class_id, student_map in lane_queue.items():
                    course = self._classes[class_id]
                    for student_id, jobs in list(student_map.items()):
                        if jobs:
                            self.deficit.accrue(student_id, lane, course.class_weight, dt)

                # Step 2 — one candidate per class
                candidates: list[tuple[float, Job, CourseClass]] = []
                for class_id, student_map in lane_queue.items():
                    course          = self._classes[class_id]
                    active_students = {s for s, jobs in student_map.items() if jobs}
                    if not active_students:
                        continue
                    student_id = self.deficit.top_student(active_students, lane)
                    job        = student_map[student_id][0]
                    util       = self.util.utilization(class_id, lane, now)
                    score      = self.scorer.score(job, course, util, now)
                    candidates.append((score, job, course))

                if not candidates:
                    continue

                # Step 3 — dispatch top-K
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
                    self.deficit.debit(job.student_id, lane, job.resource_units)
                    dispatched.append(job)
                    log_entries.append((job, lane, score))

        # Logging outside the lock — formatting is non-trivial and the
        # dispatched job objects we hold are safe to read after release.
        for job, lane, score in log_entries:
            mode = "batch" if job.batch else "interactive"
            logger.info(
                "Dispatched job %s [lane=%s mode=%s score=%.4f "
                "class=%s student=%s wait=%.1fs]",
                job.job_id, LANE_NAMES.get(lane, str(lane)), mode, score,
                job.class_id, job.student_id, job.wait_seconds(now),
            )
        return dispatched

    def _scored_candidates(self, lane: "IntEnum",
                           now: Optional[float] = None) -> list[tuple[float, "Job"]]:
        """
        Return all per-class top candidates for *lane*, scored and sorted
        descending.  Used by the wait-time snapshot to compute ranks in one
        pass rather than calling queue_rank() per pod (O(Q) vs O(Q²)).
        """
        now = now if now is not None else time.monotonic()
        with self._lock:
            candidates: list[tuple[float, "Job"]] = []
            for class_id, student_map in self._queues[lane].items():
                course          = self._classes[class_id]
                active_students = {s for s, jobs in student_map.items() if jobs}
                if not active_students:
                    continue
                student_id = self.deficit.top_student(active_students, lane)
                job        = student_map[student_id][0]
                util       = self.util.utilization(class_id, lane, now)
                score      = self.scorer.score(job, course, util, now)
                candidates.append((score, job))
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates

    def queue_rank(self, job_uid: str, lane: "IntEnum",
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
                student_id = self.deficit.top_student(active_students, lane)
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
                    result[LANE_NAMES.get(lane, str(lane))] = counts
        return result

    def class_scores(self, lane: "IntEnum",
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
                student_id = self.deficit.top_student(active, lane)
                job        = student_map[student_id][0]
                util       = self.util.utilization(class_id, lane, now)
                scores[class_id] = self.scorer.score(job, course, util, now)
        return scores
