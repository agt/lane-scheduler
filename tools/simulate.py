"""
Simulation harness for the cluster priority scheduler.
Runs a synthetic workload and prints fairness/throughput statistics.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

# Lanes are built dynamically by initialise_lanes(); we pick a representative
# set for the simulator and rebind module-level Lane after initialisation.
from lane_scheduler.core.scheduler import initialise_lanes

_GPU_CLASSES = ["small", "medium", "large"]
initialise_lanes(_GPU_CLASSES)

from lane_scheduler.core.scheduler import (  # noqa: E402 — after initialise_lanes
    CourseClass, Job, CPU_LANE, Scheduler, SchedulerConfig,
)


# ---------------------------------------------------------------------------
# Simulation config
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    seed:           int   = 42
    duration:       float = 3600.0   # simulated seconds (1 hour)
    cycle_interval: float = 10.0     # scheduling cycle every N seconds
    dispatch_k:     int   = 8

    # Lane capacities (arbitrary resource units)
    lane_capacity: dict = field(default_factory=lambda: {
        "cpu":         200.0,
        "gpu-small":    20.0,
        "gpu-medium":   10.0,
        "gpu-large":     5.0,
    })


# ---------------------------------------------------------------------------
# Synthetic class definitions
# ---------------------------------------------------------------------------

CLASSES = [
    # (class_id, tier, enrollment)
    ("INTRO-101",    1, 220),
    ("INTRO-102",    1, 180),
    ("UPPER-201",    2,  60),
    ("UPPER-202",    2,  45),
    ("GRAD-301",     3,  15),
    ("GRAD-302",     3,  10),
]

_TIER_PROFILES = {
    1: dict(
        rate=2.0,
        lane_probs={"cpu": 0.90, "gpu-small": 0.10},
        resource_range=(0.5, 2.0),
        batch_prob=0.0,
    ),
    2: dict(
        rate=3.0,
        lane_probs={"cpu": 0.70, "gpu-small": 0.20, "gpu-medium": 0.10},
        resource_range=(1.0, 4.0),
        batch_prob=0.1,
    ),
    3: dict(
        rate=4.0,
        lane_probs={"cpu": 0.50, "gpu-small": 0.15,
                    "gpu-medium": 0.20, "gpu-large": 0.15},
        resource_range=(2.0, 8.0),
        batch_prob=0.35,
    ),
}


def pick_lane(probs: dict, rng: random.Random):
    r = rng.random()
    cumulative = 0.0
    for lane, p in probs.items():
        cumulative += p
        if r < cumulative:
            return lane
    return CPU_LANE


# ---------------------------------------------------------------------------
# Statistics collector
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    dispatched_by_class:    dict = field(default_factory=lambda: defaultdict(int))
    wait_by_class:          dict = field(default_factory=lambda: defaultdict(list))
    dispatched_by_lane:     dict = field(default_factory=lambda: defaultdict(int))
    dispatched_by_tier:     dict = field(default_factory=lambda: defaultdict(int))


def record(stats: Stats, job: Job, course: CourseClass) -> None:
    wait = (job.dispatch_time - job.submit_time)
    stats.dispatched_by_class[job.class_id] += 1
    stats.wait_by_class[job.class_id].append(wait)
    stats.dispatched_by_lane[job.lane] += 1
    stats.dispatched_by_tier[course.tier.name] += 1


def print_report(stats: Stats, classes: dict[str, CourseClass],
                 total_submitted: dict[str, int], duration: float) -> None:

    print("\n" + "=" * 70)
    print("SIMULATION REPORT")
    print("=" * 70)

    # Per-class table
    print(f"\n{'Class':<14} {'Tier':<10} {'Enroll':>6} {'Weight':>7} "
          f"{'Submit':>7} {'Dispatched':>10} {'Dispatch%':>10} "
          f"{'AvgWait(s)':>11} {'P95Wait(s)':>11}")
    print("-" * 90)

    for class_id, course in sorted(classes.items()):
        submitted  = total_submitted.get(class_id, 0)
        dispatched = stats.dispatched_by_class.get(class_id, 0)
        waits      = stats.wait_by_class.get(class_id, [0])
        pct        = 100 * dispatched / submitted if submitted else 0
        avg_wait   = statistics.mean(waits) if waits else 0
        p95_wait   = sorted(waits)[int(0.95 * len(waits))] if waits else 0

        print(f"{class_id:<14} {course.tier:<10} {course.enrollment:>6} "
              f"{course.class_weight:>7.4f} {submitted:>7} {dispatched:>10} "
              f"{pct:>9.1f}% {avg_wait:>11.1f} {p95_wait:>11.1f}")

    # By tier
    print(f"\n{'--- By Tier ---'}")
    for tier_name, count in sorted(stats.dispatched_by_tier.items()):
        print(f"  {tier_name:<12}: {count} jobs dispatched")

    # By lane
    print(f"\n{'--- By Lane ---'}")
    for lane_name, count in sorted(stats.dispatched_by_lane.items()):
        print(f"  {lane_name:<18}: {count} jobs dispatched")

    print(f"\nSimulation duration: {duration:.0f}s")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run(sim_cfg: SimConfig | None = None) -> None:
    sim_cfg = sim_cfg if sim_cfg is not None else SimConfig()
    rng = random.Random(sim_cfg.seed)

    sched_cfg = SchedulerConfig(dispatch_k=sim_cfg.dispatch_k)
    scheduler = Scheduler(lane_capacity=sim_cfg.lane_capacity, config=sched_cfg)

    courses: dict[str, CourseClass] = {}
    for class_id, tier, enrollment in CLASSES:
        c = CourseClass(class_id=class_id, tier=tier, enrollment=enrollment)
        courses[class_id] = c
        scheduler.register_class(c)

    # Pre-generate job submission events
    # Each student submits jobs at a Poisson rate based on their tier profile
    events: list[tuple[float, Job]] = []
    job_counter = 0

    for class_id, tier, enrollment in CLASSES:
        profile = _TIER_PROFILES[tier]
        rate_per_second = profile["rate"] / 3600.0

        for student_idx in range(enrollment):
            student_id = f"{class_id}-S{student_idx:04d}"
            t = rng.expovariate(rate_per_second)
            while t < sim_cfg.duration:
                lane = pick_lane(profile["lane_probs"], rng)
                units = rng.uniform(*profile["resource_range"])
                batch = rng.random() < profile["batch_prob"]
                job = Job(
                    job_id=f"J{job_counter:06d}",
                    class_id=class_id,
                    student_id=student_id,
                    lane=lane,
                    batch=batch,
                    submit_time=t,
                    resource_units=units,
                )
                events.append((t, job))
                job_counter += 1
                t += rng.expovariate(rate_per_second)

    events.sort(key=lambda x: x[0])
    total_students = sum(e for _, _, e in CLASSES)
    print(f"Generated {len(events)} job submissions across "
          f"{total_students} students.")

    # Count submissions per class for reporting
    total_submitted: dict[str, int] = defaultdict(int)
    for _, job in events:
        total_submitted[job.class_id] += 1

    # Run simulation
    stats = Stats()
    event_idx = 0
    t = 0.0

    while t < sim_cfg.duration:
        # Submit all jobs due by current time
        while event_idx < len(events) and events[event_idx][0] <= t:
            _, job = events[event_idx]
            scheduler.submit(job)
            event_idx += 1

        # Run scheduling cycle
        dispatched = scheduler.cycle(now=t)
        for job in dispatched:
            record(stats, job, courses[job.class_id])

        t += sim_cfg.cycle_interval

    # Print remaining queue depths
    depths = scheduler.queue_depths()
    if depths:
        print(f"\nRemaining queue depths at end of simulation:")
        for lane_name, counts in depths.items():
            for class_id, n in counts.items():
                print(f"  [{lane_name}] {class_id}: {n} jobs waiting")

    print_report(stats, courses, total_submitted, sim_cfg.duration)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    run()
