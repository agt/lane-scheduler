"""
Unit tests for core scheduler components.
"""

import math
import unittest

from lane_scheduler.core.scheduler import (
    CourseClass, DeficitTracker, Job, PriorityScorer,
    Scheduler, SchedulerConfig, Tier, UtilizationTracker,
    initialise_lanes, lane_for_gpu_class, is_known_gpu_class,
)

# ---------------------------------------------------------------------------
# Module-level setup: build the Lane enum before any test references it
# ---------------------------------------------------------------------------

def setUpModule():
    initialise_lanes(["xsmall", "small", "medium", "large", "xlarge"])

# Convenience aliases resolved after setUpModule() — accessed via functions
# rather than module-level names so they're evaluated lazily after init.
def _cpu():        from lane_scheduler.core.scheduler import Lane; return Lane.CPU
def _gpu(cls):     return lane_for_gpu_class(cls)

CAPACITY = lambda: {
    _cpu():        100.0,
    _gpu("xsmall"):  4.0,
    _gpu("small"):   8.0,
    _gpu("medium"):  8.0,
    _gpu("large"):   4.0,
    _gpu("xlarge"):  2.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_course(class_id="TEST-101", tier=Tier.INTRO, enrollment=100):
    return CourseClass(class_id=class_id, tier=tier, enrollment=enrollment)


def make_job(job_id="J001", class_id="TEST-101", student_id="S001",
             lane=None, submit_time=0.0, resource_units=1.0, batch=False):
    if lane is None:
        from lane_scheduler.core.scheduler import Lane
        lane = Lane.CPU
    j = Job(job_id=job_id, class_id=class_id, student_id=student_id,
            lane=lane, batch=batch, resource_units=resource_units)
    j.submit_time = submit_time
    return j


# ---------------------------------------------------------------------------
# CourseClass weight tests
# ---------------------------------------------------------------------------

class TestCourseWeight(unittest.TestCase):

    def test_tier_ordering(self):
        intro = make_course(tier=Tier.INTRO,     enrollment=100)
        upper = make_course(tier=Tier.UPPER_DIV, enrollment=100)
        grad  = make_course(tier=Tier.GRAD,      enrollment=100)
        self.assertLess(intro.class_weight, upper.class_weight)
        self.assertLess(upper.class_weight, grad.class_weight)

    def test_enrollment_penalty(self):
        small = make_course(tier=Tier.INTRO, enrollment=20)
        large = make_course(tier=Tier.INTRO, enrollment=200)
        self.assertGreater(small.class_weight, large.class_weight)

    def test_sqrt_scaling(self):
        c = make_course(tier=Tier.GRAD, enrollment=25)
        self.assertAlmostEqual(c.class_weight, 3.0 / math.sqrt(25))


# ---------------------------------------------------------------------------
# PriorityScorer tests
# ---------------------------------------------------------------------------

class TestPriorityScorer(unittest.TestCase):

    def setUp(self):
        self.cfg    = SchedulerConfig()
        self.scorer = PriorityScorer(self.cfg)
        self.course = make_course(tier=Tier.GRAD, enrollment=16)

    def test_age_boost_increases_with_wait(self):
        j0 = make_job(submit_time=1000.0)
        j1 = make_job(submit_time=0.0)
        self.assertGreater(self.scorer.age_boost(j1, now=1000.0),
                           self.scorer.age_boost(j0, now=1000.0))

    def test_age_boost_minimum_is_one(self):
        j = make_job(submit_time=100.0)
        self.assertAlmostEqual(self.scorer.age_boost(j, now=100.0), 1.0)

    def test_higher_utilization_lowers_score(self):
        j = make_job(submit_time=0.0)
        self.assertGreater(
            self.scorer.score(j, self.course, utilization=0.1, now=100.0),
            self.scorer.score(j, self.course, utilization=0.9, now=100.0),
        )

    def test_epsilon_floor_prevents_division_by_zero(self):
        j = make_job(submit_time=0.0)
        score = self.scorer.score(j, self.course, utilization=0.0, now=0.0)
        self.assertTrue(math.isfinite(score))
        self.assertGreater(score, 0)

    def test_batch_scores_lower_than_interactive(self):
        ji = make_job(submit_time=0.0, batch=False)
        jb = make_job(submit_time=0.0, batch=True)
        self.assertGreater(
            self.scorer.score(ji, self.course, 0.1, 100.0),
            self.scorer.score(jb, self.course, 0.1, 100.0),
        )


# ---------------------------------------------------------------------------
# UtilizationTracker tests
# ---------------------------------------------------------------------------

class TestUtilizationTracker(unittest.TestCase):

    def test_zero_when_no_events(self):
        tracker = UtilizationTracker(window=300.0, lane_capacity=CAPACITY())
        self.assertEqual(tracker.utilization("C", _cpu(), now=100.0), 0.0)

    def test_event_contributes_to_utilization(self):
        tracker = UtilizationTracker(window=300.0, lane_capacity=CAPACITY())
        tracker.record("C", _cpu(), units=50.0, now=100.0)
        self.assertGreater(tracker.utilization("C", _cpu(), now=100.0), 0.0)

    def test_expired_events_are_purged(self):
        tracker = UtilizationTracker(window=60.0, lane_capacity=CAPACITY())
        tracker.record("C", _cpu(), units=50.0, now=0.0)
        self.assertGreater(tracker.utilization("C", _cpu(), now=0.0), 0.0)
        self.assertEqual(tracker.utilization("C", _cpu(), now=200.0), 0.0)


# ---------------------------------------------------------------------------
# DeficitTracker tests
# ---------------------------------------------------------------------------

class TestDeficitTracker(unittest.TestCase):

    def test_accrual(self):
        dt = DeficitTracker()
        dt.accrue("S1", _cpu(), class_weight=1.0, dt=10.0)
        self.assertAlmostEqual(dt.deficit("S1", _cpu()), 10.0)

    def test_debit(self):
        dt = DeficitTracker()
        dt.accrue("S1", _cpu(), class_weight=1.0, dt=10.0)
        dt.debit("S1", _cpu(), units=3.0)
        self.assertAlmostEqual(dt.deficit("S1", _cpu()), 7.0)

    def test_top_student(self):
        dt = DeficitTracker()
        dt.accrue("S1", _cpu(), class_weight=1.0, dt=5.0)
        dt.accrue("S2", _cpu(), class_weight=1.0, dt=20.0)
        dt.accrue("S3", _cpu(), class_weight=1.0, dt=10.0)
        self.assertEqual(dt.top_student({"S1", "S2", "S3"}, _cpu()), "S2")


# ---------------------------------------------------------------------------
# Scheduler integration tests
# ---------------------------------------------------------------------------

class TestScheduler(unittest.TestCase):

    def setUp(self):
        self.sched = Scheduler(
            lane_capacity=CAPACITY(),
            config=SchedulerConfig(dispatch_k=4),
        )
        self.intro = make_course("INTRO-101", Tier.INTRO,  200)
        self.grad  = make_course("GRAD-301",  Tier.GRAD,    10)
        self.sched.register_class(self.intro)
        self.sched.register_class(self.grad)

    def _submit_n(self, class_id, n, lane=None, t=0.0):
        if lane is None:
            lane = _cpu()
        for i in range(n):
            j = make_job(
                job_id=f"{class_id}-J{i}",
                class_id=class_id,
                student_id=f"{class_id}-S{i}",   # unique students
                lane=lane, submit_time=t,
            )
            self.sched.submit(j)

    def test_dispatch_returns_jobs(self):
        self._submit_n("INTRO-101", 5)
        self.assertGreater(len(self.sched.cycle(now=0.0)), 0)

    def test_grad_class_not_starved_by_intro(self):
        self._submit_n("INTRO-101", 50, t=0.0)
        self._submit_n("GRAD-301",  10, t=0.0)
        grad_count = intro_count = 0
        for cycle in range(20):
            for job in self.sched.cycle(now=cycle * 10.0):
                if job.class_id == "GRAD-301":
                    grad_count += 1
                else:
                    intro_count += 1
        self.assertGreater(grad_count,  0, "Grad class was completely starved")
        self.assertGreater(intro_count, 0)

    def test_wait_time_boosts_priority(self):
        scorer = PriorityScorer(SchedulerConfig())
        course = make_course(tier=Tier.GRAD, enrollment=9)
        old_job   = make_job(submit_time=0.0)
        fresh_job = make_job(submit_time=7200.0)
        self.assertGreater(
            scorer.score(old_job,   course, 0.1, now=7200.0),
            scorer.score(fresh_job, course, 0.1, now=7200.0),
        )

    def test_queue_depths_cpu(self):
        self._submit_n("INTRO-101", 3)
        depths = self.sched.queue_depths()
        self.assertIn("cpu", depths)
        self.assertEqual(depths["cpu"]["INTRO-101"], 3)

    def test_queue_depths_gpu(self):
        self._submit_n("GRAD-301", 2, lane=_gpu("medium"))
        depths = self.sched.queue_depths()
        self.assertIn("gpu-medium", depths)
        self.assertEqual(depths["gpu-medium"]["GRAD-301"], 2)


# ---------------------------------------------------------------------------
# Dynamic Lane enum tests
# ---------------------------------------------------------------------------

class TestDynamicLane(unittest.TestCase):

    def test_known_classes_resolve(self):
        for cls in ("xsmall", "small", "medium", "large", "xlarge"):
            lane = lane_for_gpu_class(cls)
            self.assertIsNotNone(lane)
            self.assertNotEqual(lane, _cpu())

    def test_case_insensitive(self):
        self.assertEqual(lane_for_gpu_class("Medium"), lane_for_gpu_class("medium"))
        self.assertEqual(lane_for_gpu_class("LARGE"),  lane_for_gpu_class("large"))

    def test_unknown_class_returns_fallback(self):
        lane = lane_for_gpu_class("supergpu")
        self.assertIsNotNone(lane)
        self.assertIn(lane, lane_for_gpu_class.__globals__.get(
            "GPU_LANES", set()) or set() or {lane})

    def test_gpu_lanes_distinct_from_cpu(self):
        from lane_scheduler.core.scheduler import GPU_LANES
        self.assertNotIn(_cpu(), GPU_LANES)
        for cls in ("xsmall", "small", "medium", "large", "xlarge"):
            self.assertIn(lane_for_gpu_class(cls), GPU_LANES)

    def test_build_lanes_idempotent_values(self):
        """Same input → same integer values regardless of call order."""
        from lane_scheduler.core.scheduler import build_lanes
        la = build_lanes(["small", "large", "medium"])
        lb = build_lanes(["large", "small", "medium"])
        self.assertEqual(
            {m.name: m.value for m in la},
            {m.name: m.value for m in lb},
        )

    def test_new_class_appended(self):
        """A new gpu-class sorts into the existing sequence without renumbering."""
        from lane_scheduler.core.scheduler import build_lanes
        base    = build_lanes(["small", "medium", "large"])
        extended = build_lanes(["small", "medium", "large", "xlarge"])
        for member in base:
            self.assertEqual(member.value, extended[member.name].value)


class TestIsKnownGpuClass(unittest.TestCase):
    def test_known_classes_return_true(self):
        for cls in ("xsmall", "small", "medium", "large", "xlarge"):
            self.assertTrue(is_known_gpu_class(cls))

    def test_case_insensitive(self):
        self.assertTrue(is_known_gpu_class("Small"))
        self.assertTrue(is_known_gpu_class("LARGE"))

    def test_unknown_class_returns_false(self):
        self.assertFalse(is_known_gpu_class("xyz"))
        self.assertFalse(is_known_gpu_class("supergpu"))

    def test_empty_string_returns_false(self):
        self.assertFalse(is_known_gpu_class(""))
        self.assertFalse(is_known_gpu_class("  "))


if __name__ == "__main__":
    unittest.main(verbosity=2)
