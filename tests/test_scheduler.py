"""
Unit tests for core scheduler components.
"""

import math
import unittest

from lane_scheduler.core.scheduler import (
    SchedGroup, Job, PriorityScorer,
    Scheduler, SchedulerConfig,
    initialise_lanes, lane_for_gpu_class, is_known_gpu_class,
)


def setUpModule():
    initialise_lanes(["xsmall", "small", "medium", "large", "xlarge"])


def _gpu(cls):
    return lane_for_gpu_class(cls)


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

def make_sched_group(sched_group_id="TEST-101", weight=1.0):
    return SchedGroup(sched_group_id=sched_group_id, weight=weight)


def make_job(job_id="J001", sched_group_id="TEST-101", username="S001",
             lane=None, submit_time=0.0, resource_units=1.0, batch=False):
    if lane is None:
        lane = _gpu("small")
    j = Job(job_id=job_id, sched_group_id=sched_group_id, username=username,
            lane=lane, batch=batch, resource_units=resource_units)
    j.submit_time = submit_time
    return j


# ---------------------------------------------------------------------------
# SchedGroup weight tests
# ---------------------------------------------------------------------------

class TestSchedGroupWeight(unittest.TestCase):

    def test_weight_stored_directly(self):
        g = make_sched_group(weight=0.75)
        self.assertAlmostEqual(g.weight, 0.75)

    def test_weight_ordering(self):
        low  = make_sched_group(weight=0.1)
        mid  = make_sched_group(weight=0.5)
        high = make_sched_group(weight=1.0)
        self.assertLess(low.weight, mid.weight)
        self.assertLess(mid.weight, high.weight)


# ---------------------------------------------------------------------------
# PriorityScorer tests
# ---------------------------------------------------------------------------

class TestPriorityScorer(unittest.TestCase):

    def setUp(self):
        self.cfg    = SchedulerConfig()
        self.scorer = PriorityScorer(self.cfg)
        self.group  = make_sched_group(weight=0.75)

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
            self.scorer.score(j, self.group, utilization=0.1, now=100.0),
            self.scorer.score(j, self.group, utilization=0.9, now=100.0),
        )

    def test_epsilon_floor_prevents_division_by_zero(self):
        j = make_job(submit_time=0.0)
        score = self.scorer.score(j, self.group, utilization=0.0, now=0.0)
        self.assertTrue(math.isfinite(score))
        self.assertGreater(score, 0)

    def test_batch_scores_lower_than_interactive(self):
        ji = make_job(submit_time=0.0, batch=False)
        jb = make_job(submit_time=0.0, batch=True)
        self.assertGreater(
            self.scorer.score(ji, self.group, 0.1, 100.0),
            self.scorer.score(jb, self.group, 0.1, 100.0),
        )


# ---------------------------------------------------------------------------
# Running-utilization snapshot tests
# ---------------------------------------------------------------------------

class TestRunningUtilization(unittest.TestCase):
    """Scheduler.update_running_utilization feeds U(g, lane) for scoring."""

    def setUp(self):
        self.sched = Scheduler(lane_capacity=CAPACITY())
        self.sched.register_group(make_sched_group("CSE-100"))
        self.sched.register_group(make_sched_group("CSE-200"))

    def test_idle_group_uses_epsilon_floor(self):
        # No running pods → U=0 → EPSILON floor; job still dispatches.
        self.sched.update_running_utilization({})
        j = make_job(sched_group_id="CSE-100", username="S1", submit_time=0.0)
        self.sched.submit(j)
        self.assertEqual(len(self.sched.cycle(now=0.0)), 1)

    def test_higher_running_units_lowers_score(self):
        # CSE-100 has more running units → lower U-adjusted score → dispatched after CSE-200.
        lane = _gpu("small")
        self.sched.update_running_utilization({lane: {"CSE-100": 6.0, "CSE-200": 1.0}})
        j1 = make_job(job_id="J1", sched_group_id="CSE-100", username="S1",
                      submit_time=0.0, lane=lane)
        j2 = make_job(job_id="J2", sched_group_id="CSE-200", username="S2",
                      submit_time=0.0, lane=lane)
        self.sched.submit(j1)
        self.sched.submit(j2)
        dispatched = self.sched.cycle(now=0.0)
        # Both dispatched (dispatch_k=8); CSE-200 (lower util) must rank first.
        self.assertEqual(len(dispatched), 2)
        self.assertEqual(dispatched[0].sched_group_id, "CSE-200")

    def test_zero_running_units_behaves_like_idle(self):
        # Explicit zero in the map → same as absence (EPSILON floor applies).
        lane = _gpu("small")
        self.sched.update_running_utilization({lane: {"CSE-100": 0.0}})
        j = make_job(sched_group_id="CSE-100", username="S1", submit_time=0.0, lane=lane)
        self.sched.submit(j)
        self.assertEqual(len(self.sched.cycle(now=0.0)), 1)


# ---------------------------------------------------------------------------
# User prioritization tests
# ---------------------------------------------------------------------------

class TestUserPrioritization(unittest.TestCase):
    """Tests for _top_user: fewest running → oldest pending job."""

    def _sched_with_counts(self, running_counts):
        sched = Scheduler(lane_capacity=CAPACITY(), config=SchedulerConfig())
        sched.update_running_counts(running_counts)
        return sched

    def _user_map(self, entries):
        """entries: [(username, submit_time), ...]"""
        um = {}
        for uid, t in entries:
            j = make_job(job_id=uid, username=uid, submit_time=t)
            um[uid] = [j]
        return um

    def test_no_running_picks_oldest(self):
        sched = self._sched_with_counts({})
        um = self._user_map([("S1", 100.0), ("S2", 50.0), ("S3", 75.0)])
        result = sched._top_user(set(um), _gpu("small"), um)
        self.assertEqual(result, "S2")

    def test_fewest_running_wins_over_older_submit(self):
        sched = self._sched_with_counts({_gpu("small"): {"S1": 2, "S2": 1, "S3": 0}})
        um = self._user_map([("S1", 10.0), ("S2", 20.0), ("S3", 30.0)])
        result = sched._top_user(set(um), _gpu("small"), um)
        self.assertEqual(result, "S3")

    def test_tie_in_running_broken_by_submit_time(self):
        sched = self._sched_with_counts({_gpu("small"): {"S1": 1, "S2": 1, "S3": 0, "S4": 0}})
        um = self._user_map([("S1", 10.0), ("S2", 20.0), ("S3", 40.0), ("S4", 30.0)])
        result = sched._top_user({"S3", "S4"}, _gpu("small"), um)
        self.assertEqual(result, "S4")

    def test_single_user_always_selected(self):
        sched = self._sched_with_counts({_gpu("small"): {"S1": 5}})
        um = self._user_map([("S1", 100.0)])
        result = sched._top_user({"S1"}, _gpu("small"), um)
        self.assertEqual(result, "S1")

    def test_unknown_user_treated_as_zero_running(self):
        sched = self._sched_with_counts({_gpu("small"): {"S2": 1}})
        um = self._user_map([("S1", 200.0), ("S2", 10.0)])
        result = sched._top_user(set(um), _gpu("small"), um)
        self.assertEqual(result, "S1")

    def test_different_lane_counts_not_mixed(self):
        sched = self._sched_with_counts({_gpu("small"): {"S1": 3}})
        um = self._user_map([("S1", 10.0), ("S2", 20.0)])
        result = sched._top_user(set(um), _gpu("medium"), um)
        self.assertEqual(result, "S1")

    def test_cycle_respects_running_counts(self):
        """Scheduler.cycle() selects the user with fewest running pods."""
        sched = Scheduler(lane_capacity=CAPACITY(), config=SchedulerConfig(dispatch_k=1))
        group = make_sched_group()
        sched.register_group(group)

        j1 = make_job("J1", username="S1", submit_time=0.0)
        j2 = make_job("J2", username="S2", submit_time=5.0)
        sched.submit(j1)
        sched.submit(j2)

        sched.update_running_counts({_gpu("small"): {"S1": 1}})
        dispatched = sched.cycle(now=100.0)

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].username, "S2")


# ---------------------------------------------------------------------------
# Scheduler integration tests
# ---------------------------------------------------------------------------

class TestScheduler(unittest.TestCase):

    def setUp(self):
        self.sched = Scheduler(
            lane_capacity=CAPACITY(),
            config=SchedulerConfig(dispatch_k=4),
        )
        self.intro = make_sched_group("INTRO-101", weight=0.07)
        self.grad  = make_sched_group("GRAD-301",  weight=0.77)
        self.sched.register_group(self.intro)
        self.sched.register_group(self.grad)

    def _submit_n(self, sched_group_id, n, lane=None, t=0.0):
        if lane is None:
            lane = _gpu("small")
        for i in range(n):
            j = make_job(
                job_id=f"{sched_group_id}-J{i}",
                sched_group_id=sched_group_id,
                username=f"{sched_group_id}-S{i}",
                lane=lane, submit_time=t,
            )
            self.sched.submit(j)

    def test_dispatch_returns_jobs(self):
        self._submit_n("INTRO-101", 5)
        self.assertGreater(len(self.sched.cycle(now=0.0)), 0)

    def test_grad_group_not_starved_by_intro(self):
        self._submit_n("INTRO-101", 50, t=0.0)
        self._submit_n("GRAD-301",  10, t=0.0)
        grad_count = intro_count = 0
        for cycle in range(20):
            for job in self.sched.cycle(now=cycle * 10.0):
                if job.sched_group_id == "GRAD-301":
                    grad_count += 1
                else:
                    intro_count += 1
        self.assertGreater(grad_count,  0, "Grad group was completely starved")
        self.assertGreater(intro_count, 0)

    def test_wait_time_boosts_priority(self):
        scorer = PriorityScorer(SchedulerConfig())
        group = make_sched_group(weight=1.0)
        old_job   = make_job(submit_time=0.0)
        fresh_job = make_job(submit_time=7200.0)
        self.assertGreater(
            scorer.score(old_job,   group, 0.1, now=7200.0),
            scorer.score(fresh_job, group, 0.1, now=7200.0),
        )

    def test_queue_depths_gpu_small(self):
        self._submit_n("INTRO-101", 3)
        depths = self.sched.queue_depths()
        self.assertIn("gpu-small", depths)
        self.assertEqual(depths["gpu-small"]["INTRO-101"], 3)

    def test_queue_depths_gpu_medium(self):
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

    def test_valid_defaults(self):
        SchedulerConfig()

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

    def test_zero_dispatch_k(self):
        with self.assertRaises(ValueError):
            SchedulerConfig(dispatch_k=0)


class TestRemoveJob(unittest.TestCase):

    def setUp(self):
        self.s = Scheduler(lane_capacity=CAPACITY())
        self.s.register_group(make_sched_group("CSE-100"))

    def test_remove_unknown_returns_none(self):
        self.assertIsNone(self.s.remove_job("does-not-exist"))

    def test_remove_existing_returns_job_and_clears_queue(self):
        job = make_job(job_id="JOB-A", sched_group_id="CSE-100", username="S1")
        self.s.submit(job)
        removed = self.s.remove_job("JOB-A")
        self.assertIs(removed, job)
        self.assertEqual(self.s.queue_depths(), {})

    def test_remove_preserves_other_jobs(self):
        j1 = make_job(job_id="JOB-A", sched_group_id="CSE-100", username="S1")
        j2 = make_job(job_id="JOB-B", sched_group_id="CSE-100", username="S2")
        self.s.submit(j1)
        self.s.submit(j2)
        self.s.remove_job("JOB-A")
        depths = self.s.queue_depths()
        self.assertEqual(sum(sum(cm.values()) for cm in depths.values()), 1)



class TestConcurrentSubmitCycle(unittest.TestCase):

    def test_no_exceptions_under_contention(self):
        import threading

        s = Scheduler(lane_capacity=CAPACITY())
        for gid in ("CSE-100", "CSE-200", "CSE-300"):
            s.register_group(make_sched_group(gid))

        stop = threading.Event()
        errors: list = []

        def submitter(prefix: str):
            i = 0
            try:
                while not stop.is_set():
                    job = make_job(
                        job_id=f"{prefix}-{i}",
                        sched_group_id="CSE-100",
                        username=f"S{i % 5}",
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

    def test_strict_returns_none_for_unknown(self):
        self.assertIsNone(lane_for_gpu_class("h200", strict=True))

    def test_strict_returns_lane_for_known(self):
        self.assertEqual(lane_for_gpu_class("medium", strict=True), "gpu-medium")

    def test_non_strict_unknown_returns_none(self):
        self.assertIsNone(lane_for_gpu_class("h200"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
