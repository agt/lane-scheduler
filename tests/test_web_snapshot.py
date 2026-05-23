"""
Tests for lane_scheduler.web.snapshot.build_snapshot().

Uses a real Scheduler + CourseRegistry + ResidencyStats/WaitTimeCache but
mocks the Kubernetes client so no cluster is needed.
"""

import time
import unittest
from unittest.mock import MagicMock

from lane_scheduler.core.scheduler import (
    CourseClass, Job, SchedulerConfig, Scheduler, Tier,
    initialise_lanes, lane_for_gpu_class,
)
from lane_scheduler.core.course_registry import CourseRegistry
from lane_scheduler.core.node_capacity import NodeCapacityTracker
from lane_scheduler.estimation.wait_estimator import ResidencyProfile, WaitTimeCache
from lane_scheduler.estimation.residency_stats import ResidencyStats


def setUpModule():
    initialise_lanes(["small"])


def _cpu():
    from lane_scheduler.core.scheduler import Lane
    return Lane.CPU


def _small():
    return lane_for_gpu_class("small")


def _make_controller(cycle_interval=10.0):
    """Build a LaneSchedulerController with a mock K8s client."""
    from lane_scheduler.k8s.controller import LaneSchedulerController

    core_v1 = MagicMock()
    registry = CourseRegistry()
    sched_config = SchedulerConfig()
    residency_profiles = {
        "interactive": ResidencyProfile(mean_pct=0.4, std_pct=0.2),
        "batch":       ResidencyProfile(mean_pct=0.7, std_pct=0.15),
    }
    ctrl = LaneSchedulerController(
        core_v1=core_v1,
        registry=registry,
        sched_config=sched_config,
        residency_profiles=residency_profiles,
        cycle_interval=cycle_interval,
        web_port=0,
    )
    return ctrl


def _register(ctrl, class_id, tier=Tier.INTRO, enrollment=100):
    course = CourseClass(class_id=class_id, tier=tier, enrollment=enrollment)
    ctrl.registry._courses[class_id] = course
    ctrl.scheduler.register_class(course)
    return course


def _submit(ctrl, class_id, lane, job_id="J1", student_id="s1", batch=False):
    job = Job(
        job_id=job_id,
        class_id=class_id,
        student_id=student_id,
        lane=lane,
        batch=batch,
        resource_units=1.0,
    )
    job.submit_time = time.monotonic() - 30.0  # 30s old
    ctrl.scheduler.submit(job)
    return job


class TestBuildSnapshotStructure(unittest.TestCase):
    """Verify the top-level shape and required keys are always present."""

    def setUp(self):
        self.ctrl = _make_controller()

    def test_required_top_level_keys(self):
        from lane_scheduler.web.snapshot import build_snapshot
        snap = build_snapshot(self.ctrl)
        for key in ("generated_at", "system", "lanes", "courses"):
            self.assertIn(key, snap)

    def test_system_keys(self):
        from lane_scheduler.web.snapshot import build_snapshot
        sys = build_snapshot(self.ctrl)["system"]
        for key in ("cycle_interval_s", "wait_cache_age_s",
                    "wait_cache_duration_s", "course_count"):
            self.assertIn(key, sys)

    def test_cycle_interval_propagated(self):
        from lane_scheduler.web.snapshot import build_snapshot
        ctrl = _make_controller(cycle_interval=30.0)
        snap = build_snapshot(ctrl)
        self.assertEqual(snap["system"]["cycle_interval_s"], 30.0)

    def test_lanes_list_has_cpu_and_gpu(self):
        from lane_scheduler.web.snapshot import build_snapshot
        snap = build_snapshot(self.ctrl)
        names = {ln["name"] for ln in snap["lanes"]}
        self.assertIn("cpu", names)
        self.assertIn("gpu-small", names)

    def test_lane_row_keys(self):
        from lane_scheduler.web.snapshot import build_snapshot
        snap = build_snapshot(self.ctrl)
        for row in snap["lanes"]:
            for key in ("name", "node_count", "capacity_units",
                        "running_count", "running_units",
                        "queued_count", "drain_p80_s"):
                self.assertIn(key, row, f"missing key {key!r} in lane row")


class TestBuildSnapshotEmpty(unittest.TestCase):
    """Empty cluster — no queued or running jobs."""

    def setUp(self):
        self.ctrl = _make_controller()

    def test_no_courses_in_output(self):
        from lane_scheduler.web.snapshot import build_snapshot
        snap = build_snapshot(self.ctrl)
        self.assertEqual(snap["courses"], [])

    def test_queued_count_zero(self):
        from lane_scheduler.web.snapshot import build_snapshot
        snap = build_snapshot(self.ctrl)
        for row in snap["lanes"]:
            self.assertEqual(row["queued_count"], 0)
            self.assertIsNone(row["drain_p80_s"])


class TestBuildSnapshotWithJobs(unittest.TestCase):
    """Queue with a couple of courses and jobs."""

    def setUp(self):
        self.ctrl = _make_controller()
        _register(self.ctrl, "CSE101", tier=Tier.INTRO, enrollment=200)
        _register(self.ctrl, "CSE250", tier=Tier.GRAD,  enrollment=40)

    def test_queued_course_appears(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE101", _cpu(), job_id="J1", student_id="s1")
        snap = build_snapshot(self.ctrl)
        course_ids = {c["course_id"] for c in snap["courses"]}
        self.assertIn("CSE101", course_ids)

    def test_queued_count_matches(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE101", _cpu(), job_id="J1", student_id="s1")
        _submit(self.ctrl, "CSE101", _cpu(), job_id="J2", student_id="s2")
        snap = build_snapshot(self.ctrl)
        cse101 = next(c for c in snap["courses"] if c["course_id"] == "CSE101")
        cpu_lane = next(l for l in cse101["lanes"] if l["lane_name"] == "cpu")
        self.assertEqual(cpu_lane["queued_count"], 2)

    def test_lane_queued_count_matches(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE101", _cpu(), job_id="J1", student_id="s1")
        _submit(self.ctrl, "CSE250", _cpu(), job_id="J2", student_id="s2")
        snap = build_snapshot(self.ctrl)
        cpu_row = next(r for r in snap["lanes"] if r["name"] == "cpu")
        self.assertEqual(cpu_row["queued_count"], 2)

    def test_course_metadata(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE250", _small(), job_id="J1", student_id="s1")
        snap = build_snapshot(self.ctrl)
        cse250 = next(c for c in snap["courses"] if c["course_id"] == "CSE250")
        self.assertEqual(cse250["tier"], "GRAD")
        self.assertEqual(cse250["enrollment"], 40)
        self.assertAlmostEqual(cse250["weight"], 3.0 / (40 ** 0.5), places=3)

    def test_tail_wait_absent_for_single_job(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE101", _cpu(), job_id="J1", student_id="s1")
        snap = build_snapshot(self.ctrl)
        cse101 = next(c for c in snap["courses"] if c["course_id"] == "CSE101")
        cpu_lane = next(l for l in cse101["lanes"] if l["lane_name"] == "cpu")
        self.assertEqual(cpu_lane["queued_count"], 1)
        self.assertIsNone(cpu_lane["tail_wait"])

    def test_tail_wait_present_for_multiple_jobs(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE101", _cpu(), job_id="J1", student_id="s1")
        _submit(self.ctrl, "CSE101", _cpu(), job_id="J2", student_id="s2")
        snap = build_snapshot(self.ctrl)
        cse101 = next(c for c in snap["courses"] if c["course_id"] == "CSE101")
        cpu_lane = next(l for l in cse101["lanes"] if l["lane_name"] == "cpu")
        self.assertEqual(cpu_lane["queued_count"], 2)
        # tail_wait may be None if no running pods to base estimate on,
        # but the dict key must be present
        self.assertIn("tail_wait", cpu_lane)

    def test_sorted_by_queued_descending(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE250", _cpu(), job_id="J1", student_id="s1")
        _submit(self.ctrl, "CSE101", _cpu(), job_id="J2", student_id="s2")
        _submit(self.ctrl, "CSE101", _cpu(), job_id="J3", student_id="s3")
        snap = build_snapshot(self.ctrl)
        queued = [
            sum(l["queued_count"] for l in c["lanes"])
            for c in snap["courses"]
        ]
        self.assertEqual(queued, sorted(queued, reverse=True))


class TestNoPrivateIdentifiers(unittest.TestCase):
    """Ensure no student/pod identifiers leak into the snapshot."""

    def setUp(self):
        self.ctrl = _make_controller()
        _register(self.ctrl, "CSE101", tier=Tier.INTRO, enrollment=100)

    def test_no_student_ids(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE101", _cpu(), job_id="pod-uid-abc", student_id="alice")
        snap = build_snapshot(self.ctrl)
        dump = str(snap)
        self.assertNotIn("alice", dump)
        self.assertNotIn("pod-uid-abc", dump)

    def test_unlabelled_course_excluded(self):
        from lane_scheduler.web.snapshot import build_snapshot
        from lane_scheduler.k8s.pod_translator import NO_COURSE_LABEL
        ctrl = _make_controller()
        course = CourseClass(class_id=NO_COURSE_LABEL, tier=Tier.INTRO, enrollment=1)
        ctrl.scheduler._classes[NO_COURSE_LABEL] = course
        job = Job(job_id="J-unlabelled", class_id=NO_COURSE_LABEL,
                  student_id="s1", lane=_cpu())
        job.submit_time = time.monotonic()
        ctrl.scheduler._queues[_cpu()][NO_COURSE_LABEL]["s1"].append(job)
        snap = build_snapshot(ctrl)
        course_ids = {c["course_id"] for c in snap["courses"]}
        self.assertNotIn(NO_COURSE_LABEL, course_ids)


class TestWaitCacheAllEstimates(unittest.TestCase):
    """WaitTimeCache.all_estimates() returns a copy of the current cache."""

    def test_empty_cache(self):
        cache = WaitTimeCache(snapshot_fn=lambda: {})
        self.assertEqual(cache.all_estimates(), {})

    def test_populated_cache(self):
        from lane_scheduler.estimation.wait_estimator import WaitEstimate
        est = WaitEstimate(median_seconds=60.0, p20_seconds=30.0,
                           p80_seconds=120.0, queue_rank=1, lane_name="cpu")
        cache = WaitTimeCache(snapshot_fn=lambda: {"uid-1": est})
        # Manually inject into cache to avoid background thread timing
        from lane_scheduler.estimation.wait_estimator import CacheEntry
        import threading
        cache._cache = {"uid-1": CacheEntry(estimate=est, computed_at=time.monotonic())}
        result = cache.all_estimates()
        self.assertIn("uid-1", result)
        self.assertIs(result["uid-1"], est)

    def test_returns_copy(self):
        from lane_scheduler.estimation.wait_estimator import WaitEstimate, CacheEntry
        est = WaitEstimate(median_seconds=60.0, p20_seconds=30.0,
                           p80_seconds=120.0, queue_rank=1, lane_name="cpu")
        cache = WaitTimeCache(snapshot_fn=lambda: {})
        cache._cache = {"uid-1": CacheEntry(estimate=est, computed_at=time.monotonic())}
        r1 = cache.all_estimates()
        cache._cache = {}
        r2 = cache.all_estimates()
        self.assertIn("uid-1", r1)
        self.assertEqual(r2, {})


if __name__ == "__main__":
    unittest.main()
