"""
Unit tests for core scheduler components.
"""

import math
import unittest

from lane_scheduler.core.scheduler import (
    CourseClass, Job, PriorityScorer,
    Scheduler, SchedulerConfig, UtilizationTracker,
    initialise_lanes, lane_for_gpu_class, is_known_gpu_class,
)

# ---------------------------------------------------------------------------
# Module-level setup: build the Lane enum before any test references it
# ---------------------------------------------------------------------------

def setUpModule():
    initialise_lanes(["xsmall", "small", "medium", "large", "xlarge"])

# Convenience aliases resolved after setUpModule() — accessed via functions
# rather than module-level names so they're evaluated lazily after init.
def _gpu(cls):     return lane_for_gpu_class(cls)

CAPACITY = lambda: {
    _gpu("xsmall"):  4.0,
    _gpu("small"):   8.0,
    _gpu("medium"):  8.0,
    _gpu("large"):   4.0,
    _gpu("xlarge"):  2.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_course(class_id="TEST-101", weight=1.0):
    return CourseClass(class_id=class_id, class_weight=weight)


def make_job(job_id="J001", class_id="TEST-101", student_id="S001",
             lane=None, submit_time=0.0, resource_units=1.0, batch=False):
    if lane is None:
        lane = _gpu("small")
    j = Job(job_id=job_id, class_id=class_id, student_id=student_id,
            lane=lane, batch=batch, resource_units=resource_units)
    j.submit_time = submit_time
    return j


# ---------------------------------------------------------------------------
# CourseClass weight tests
# ---------------------------------------------------------------------------

class TestCourseWeight(unittest.TestCase):

    def test_weight_stored_directly(self):
        c = make_course(weight=0.75)
        self.assertAlmostEqual(c.class_weight, 0.75)

    def test_weight_ordering(self):
        low  = make_course(weight=0.1)
        mid  = make_course(weight=0.5)
        high = make_course(weight=1.0)
        self.assertLess(low.class_weight, mid.class_weight)
        self.assertLess(mid.class_weight, high.class_weight)


# ---------------------------------------------------------------------------
# PriorityScorer tests
# ---------------------------------------------------------------------------

class TestPriorityScorer(unittest.TestCase):

    def setUp(self):
        self.cfg    = SchedulerConfig()
        self.scorer = PriorityScorer(self.cfg)
        self.course = make_course(weight=0.75)

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
        self.assertEqual(tracker.utilization("C", _gpu("small"), now=100.0), 0.0)

    def test_event_contributes_to_utilization(self):
        tracker = UtilizationTracker(window=300.0, lane_capacity=CAPACITY())
        tracker.record("C", _gpu("small"), units=50.0, now=100.0)
        self.assertGreater(tracker.utilization("C", _gpu("small"), now=100.0), 0.0)

    def test_expired_events_are_purged(self):
        tracker = UtilizationTracker(window=60.0, lane_capacity=CAPACITY())
        tracker.record("C", _gpu("small"), units=50.0, now=0.0)
        self.assertGreater(tracker.utilization("C", _gpu("small"), now=0.0), 0.0)
        self.assertEqual(tracker.utilization("C", _gpu("small"), now=200.0), 0.0)


# ---------------------------------------------------------------------------
# Student prioritization tests
# ---------------------------------------------------------------------------

class TestStudentPrioritization(unittest.TestCase):
    """Tests for _top_student: fewest running → oldest pending job."""

    def _sched_with_counts(self, running_counts):
        sched = Scheduler(lane_capacity=CAPACITY(), config=SchedulerConfig())
        sched.update_running_counts(running_counts)
        return sched

    def _student_map(self, entries):
        """entries: [(student_id, submit_time), ...]"""
        sm = {}
        for sid, t in entries:
            j = make_job(job_id=sid, student_id=sid, submit_time=t)
            sm[sid] = [j]
        return sm

    def test_no_running_picks_oldest(self):
        sched = self._sched_with_counts({})
        sm = self._student_map([("S1", 100.0), ("S2", 50.0), ("S3", 75.0)])
        # all have 0 running; S2 submitted earliest
        result = sched._top_student(set(sm), _gpu("small"), sm)
        self.assertEqual(result, "S2")

    def test_fewest_running_wins_over_older_submit(self):
        sched = self._sched_with_counts({_gpu("small"): {"S1": 2, "S2": 1, "S3": 0}})
        # S1 has oldest job but most running; S3 has fewest running
        sm = self._student_map([("S1", 10.0), ("S2", 20.0), ("S3", 30.0)])
        result = sched._top_student(set(sm), _gpu("small"), sm)
        self.assertEqual(result, "S3")

    def test_tie_in_running_broken_by_submit_time(self):
        sched = self._sched_with_counts({_gpu("small"): {"S1": 1, "S2": 1, "S3": 0, "S4": 0}})
        sm = self._student_map([("S1", 10.0), ("S2", 20.0), ("S3", 40.0), ("S4", 30.0)])
        # S3 and S4 tied at 0 running; S4 submitted earlier
        result = sched._top_student({"S3", "S4"}, _gpu("small"), sm)
        self.assertEqual(result, "S4")

    def test_single_student_always_selected(self):
        sched = self._sched_with_counts({_gpu("small"): {"S1": 5}})
        sm = self._student_map([("S1", 100.0)])
        result = sched._top_student({"S1"}, _gpu("small"), sm)
        self.assertEqual(result, "S1")

    def test_unknown_student_treated_as_zero_running(self):
        # S2 appears in running_counts with 1; S1 has no entry → 0 running
        sched = self._sched_with_counts({_gpu("small"): {"S2": 1}})
        sm = self._student_map([("S1", 200.0), ("S2", 10.0)])
        result = sched._top_student(set(sm), _gpu("small"), sm)
        self.assertEqual(result, "S1")

    def test_different_lane_counts_not_mixed(self):
        # S1 has 3 running in gpu-small but 0 in gpu-medium; S2 has 0 in both
        sched = self._sched_with_counts({_gpu("small"): {"S1": 3}})
        sm = self._student_map([("S1", 10.0), ("S2", 20.0)])
        # Both have 0 running in gpu-medium; S1 wins on submit time
        result = sched._top_student(set(sm), _gpu("medium"), sm)
        self.assertEqual(result, "S1")

    def test_cycle_respects_running_counts(self):
        """Scheduler.cycle() selects the student with fewest running pods."""
        sched = Scheduler(lane_capacity=CAPACITY(), config=SchedulerConfig(dispatch_k=1))
        course = make_course()
        sched.register_class(course)

        # S1 submitted first but has a running pod; S2 has none
        j1 = make_job("J1", student_id="S1", submit_time=0.0)
        j2 = make_job("J2", student_id="S2", submit_time=5.0)
        sched.submit(j1)
        sched.submit(j2)

        sched.update_running_counts({_gpu("small"): {"S1": 1}})
        dispatched = sched.cycle(now=100.0)

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].student_id, "S2")


# ---------------------------------------------------------------------------
# Scheduler integration tests
# ---------------------------------------------------------------------------

class TestScheduler(unittest.TestCase):

    def setUp(self):
        self.sched = Scheduler(
            lane_capacity=CAPACITY(),
            config=SchedulerConfig(dispatch_k=4),
        )
        self.intro = make_course("INTRO-101", weight=0.07)
        self.grad  = make_course("GRAD-301",  weight=0.77)
        self.sched.register_class(self.intro)
        self.sched.register_class(self.grad)

    def _submit_n(self, class_id, n, lane=None, t=0.0):
        if lane is None:
            lane = _gpu("small")
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
        course = make_course(weight=1.0)
        old_job   = make_job(submit_time=0.0)
        fresh_job = make_job(submit_time=7200.0)
        self.assertGreater(
            scorer.score(old_job,   course, 0.1, now=7200.0),
            scorer.score(fresh_job, course, 0.1, now=7200.0),
        )

    def test_queue_depths_gpu_small(self):
        self._submit_n("INTRO-101", 3)
        depths = self.sched.queue_depths()
        self.assertIn("gpu-small", depths)
        self.assertEqual(depths["gpu-small"]["INTRO-101"], 3)

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

    def test_case_insensitive(self):
        self.assertEqual(lane_for_gpu_class("Medium"), lane_for_gpu_class("medium"))
        self.assertEqual(lane_for_gpu_class("LARGE"),  lane_for_gpu_class("large"))

    def test_unknown_class_returns_none(self):
        lane = lane_for_gpu_class("supergpu", strict=True)
        self.assertIsNone(lane)

    def test_unknown_class_non_strict_returns_none(self):
        lane = lane_for_gpu_class("supergpu")
        self.assertIsNone(lane)

    def test_lane_strings_have_expected_format(self):
        for cls in ("xsmall", "small", "medium", "large", "xlarge"):
            lane = lane_for_gpu_class(cls)
            self.assertEqual(lane, f"gpu-{cls}")


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


class TestSchedulerConfigValidation(unittest.TestCase):
    """SchedulerConfig.__post_init__ rejects invalid values."""

    def test_valid_defaults(self):
        SchedulerConfig()   # must not raise

    def test_negative_alpha(self):
        with self.assertRaises(ValueError):
            SchedulerConfig(alpha=-0.1)

    def test_zero_t_half_interactive(self):
        with self.assertRaises(ValueError):
            SchedulerConfig(t_half_interactive=0.0)

    def test_zero_t_half_batch(self):
        with self.assertRaises(ValueError):
            SchedulerConfig(t_half_batch=0.0)

    def test_zero_batch_mode_penalty(self):
        with self.assertRaises(ValueError):
            SchedulerConfig(batch_mode_penalty=0.0)

    def test_zero_utilization_window(self):
        with self.assertRaises(ValueError):
            SchedulerConfig(utilization_window=0.0)

    def test_zero_dispatch_k(self):
        with self.assertRaises(ValueError):
            SchedulerConfig(dispatch_k=0)


class TestRemoveJob(unittest.TestCase):
    """Scheduler.remove_job clears orphan Jobs from the queue."""

    def setUp(self):
        self.s = Scheduler(lane_capacity=CAPACITY())
        self.s.register_class(make_course("CSE-100"))

    def test_remove_unknown_returns_none(self):
        self.assertIsNone(self.s.remove_job("does-not-exist"))

    def test_remove_existing_returns_job_and_clears_queue(self):
        job = make_job(job_id="JOB-A", class_id="CSE-100", student_id="S1")
        self.s.submit(job)
        removed = self.s.remove_job("JOB-A")
        self.assertIs(removed, job)
        # Queue path should be cleaned up
        self.assertEqual(self.s.queue_depths(), {})

    def test_remove_preserves_other_jobs(self):
        j1 = make_job(job_id="JOB-A", class_id="CSE-100", student_id="S1")
        j2 = make_job(job_id="JOB-B", class_id="CSE-100", student_id="S2")
        self.s.submit(j1)
        self.s.submit(j2)
        self.s.remove_job("JOB-A")
        depths = self.s.queue_depths()
        self.assertEqual(sum(sum(cm.values()) for cm in depths.values()), 1)


class TestUtilizationPruning(unittest.TestCase):
    """UtilizationTracker drops empty keys to bound memory."""

    def test_empty_key_pruned_after_expiry(self):
        u = UtilizationTracker(window=10.0, lane_capacity={_gpu("small"): 100.0})
        u.record("CSE-100", _gpu("small"), 1.0, now=0.0)
        # Far future — all events expired
        u.utilization("CSE-100", _gpu("small"), now=1000.0)
        self.assertNotIn(("CSE-100", _gpu("small")), u._events)


class TestConcurrentSubmitCycle(unittest.TestCase):
    """Scheduler.submit and Scheduler.cycle from concurrent threads must
    not crash and must preserve queue invariants."""

    def test_no_exceptions_under_contention(self):
        import threading

        s = Scheduler(lane_capacity=CAPACITY())
        for cid in ("CSE-100", "CSE-200", "CSE-300"):
            s.register_class(make_course(cid))

        stop = threading.Event()
        errors: list[Exception] = []

        def submitter(prefix: str):
            i = 0
            try:
                while not stop.is_set():
                    job = make_job(
                        job_id=f"{prefix}-{i}",
                        class_id="CSE-100",
                        student_id=f"S{i % 5}",
                    )
                    s.submit(job)
                    i += 1
            except Exception as e:
                errors.append(e)

        def cycler():
            try:
                while not stop.is_set():
                    s.cycle()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=submitter, args=("A",)),
            threading.Thread(target=submitter, args=("B",)),
            threading.Thread(target=cycler),
        ]
        for t in threads:
            t.start()
        import time as _t
        _t.sleep(0.5)
        stop.set()
        for t in threads:
            t.join(timeout=5.0)

        self.assertFalse(errors, f"thread errors: {errors!r}")


class TestLaneForGpuClassStrict(unittest.TestCase):
    """strict=True returns None for unknown classes; default still falls back."""

    def test_strict_returns_none_for_unknown(self):
        self.assertIsNone(lane_for_gpu_class("h200", strict=True))

    def test_strict_returns_lane_for_known(self):
        self.assertEqual(lane_for_gpu_class("medium", strict=True), "gpu-medium")

    def test_non_strict_unknown_returns_none(self):
        self.assertIsNone(lane_for_gpu_class("h200"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
