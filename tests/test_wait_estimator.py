"""
Unit tests for wait_estimator.py

All tests use stdlib only and do not require a live cluster or the
dynamic Lane enum (wait_estimator is Lane-agnostic).
"""

import math
import time
import unittest

from lane_scheduler.estimation.wait_estimator import (
    ResidencyProfile,
    RunningPod,
    WaitEstimate,
    estimate_wait,
    format_wait,
    _phi,
    _finish_prob,
    _expected_slots,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile(mean=0.4, std=0.2) -> ResidencyProfile:
    return ResidencyProfile(mean_pct=mean, std_pct=std)


PROFILES = {
    "interactive": _profile(mean=0.4, std=0.2),
    "batch":       _profile(mean=0.7, std=0.15),
}


def _pod(deadline=3600.0, age=0.0, batch=False, units=1.0) -> RunningPod:
    now = time.monotonic()
    rp  = RunningPod(
        pod_uid                 = "uid",
        start_time              = now - age,
        active_deadline_seconds = deadline,
        batch                   = batch,
        resource_units          = units,
    )
    return rp


# ---------------------------------------------------------------------------
# Truncated-normal math
# ---------------------------------------------------------------------------

class TestPhi(unittest.TestCase):

    def test_phi_at_zero(self):
        self.assertAlmostEqual(_phi(0.0), 0.5, places=6)

    def test_phi_large_positive(self):
        self.assertAlmostEqual(_phi(10.0), 1.0, places=6)

    def test_phi_large_negative(self):
        self.assertAlmostEqual(_phi(-10.0), 0.0, places=6)

    def test_phi_symmetry(self):
        for x in (0.5, 1.0, 1.96, 2.5):
            self.assertAlmostEqual(_phi(x) + _phi(-x), 1.0, places=10)


class TestFinishProb(unittest.TestCase):

    def test_zero_time_gives_zero_prob(self):
        p = _finish_prob(t=0.0, mu=1800.0, sigma=360.0, age=0.0)
        self.assertAlmostEqual(p, 0.0, places=4)

    def test_long_time_gives_high_prob(self):
        p = _finish_prob(t=86400.0, mu=1800.0, sigma=360.0, age=0.0)
        self.assertGreater(p, 0.99)

    def test_prob_increases_with_t(self):
        mu, sigma, age = 1800.0, 360.0, 0.0
        p1 = _finish_prob(600.0,  mu, sigma, age)
        p2 = _finish_prob(1800.0, mu, sigma, age)
        p3 = _finish_prob(3600.0, mu, sigma, age)
        self.assertLess(p1, p2)
        self.assertLess(p2, p3)

    def test_old_pod_has_high_finish_prob(self):
        # Pod that has survived far past expected lifetime
        p = _finish_prob(t=60.0, mu=1800.0, sigma=360.0, age=7200.0)
        self.assertGreater(p, 0.99)

    def test_zero_sigma_deterministic(self):
        # Pod finishes exactly at mu
        p_before = _finish_prob(t=1799.0, mu=1800.0, sigma=0.0, age=0.0)
        p_after  = _finish_prob(t=1801.0, mu=1800.0, sigma=0.0, age=0.0)
        self.assertAlmostEqual(p_before, 0.0)
        self.assertAlmostEqual(p_after,  1.0)


# ---------------------------------------------------------------------------
# Expected slots
# ---------------------------------------------------------------------------

class TestExpectedSlots(unittest.TestCase):

    def test_no_running_pods_zero_slots(self):
        mean, var = _expected_slots(t=600.0, running=[], profiles=PROFILES, now=time.monotonic())
        self.assertAlmostEqual(mean, 0.0)
        self.assertAlmostEqual(var,  0.0)

    def test_single_pod_near_deadline_finishes_before_deadline(self):
        # Pod at 99% of deadline — remaining_max is ~36s, so within 36s it must finish
        deadline = 3600.0
        pod = _pod(deadline=deadline, age=deadline * 0.99)
        mean, _ = _expected_slots(t=pod.remaining_max() + 1.0,
                                   running=[pod], profiles=PROFILES,
                                   now=time.monotonic())
        # By remaining_max, finish probability must be 1.0 (hard deadline)
        self.assertAlmostEqual(mean, 1.0, places=3)

    def test_single_fresh_pod_unlikely_to_finish_quickly(self):
        pod  = _pod(deadline=3600.0, age=0.0)
        mean, _ = _expected_slots(t=30.0, running=[pod], profiles=PROFILES,
                                  now=time.monotonic())
        self.assertLess(mean, 0.1)

    def test_slots_increase_with_t(self):
        pods = [_pod(deadline=3600.0, age=1000.0) for _ in range(5)]
        now  = time.monotonic()
        m1, _ = _expected_slots(600.0,  pods, PROFILES, now)
        m2, _ = _expected_slots(3600.0, pods, PROFILES, now)
        self.assertLess(m1, m2)

    def test_resource_units_scale_slots(self):
        """A pod consuming 2 units contributes 2× to expected slot count."""
        pod1 = _pod(deadline=3600.0, age=1800.0, units=1.0)
        pod2 = _pod(deadline=3600.0, age=1800.0, units=2.0)
        now  = time.monotonic()
        m1, _ = _expected_slots(600.0, [pod1], PROFILES, now)
        m2, _ = _expected_slots(600.0, [pod2], PROFILES, now)
        self.assertAlmostEqual(m2, 2 * m1, places=5)

    def test_variance_nonzero_for_uncertain_pods(self):
        pod  = _pod(deadline=3600.0, age=1800.0)
        _, var = _expected_slots(600.0, [pod], PROFILES, now=time.monotonic())
        self.assertGreater(var, 0.0)


# ---------------------------------------------------------------------------
# estimate_wait
# ---------------------------------------------------------------------------

class TestEstimateWait(unittest.TestCase):

    def test_rank_1_no_running_is_immediate(self):
        est = estimate_wait(
            queue_rank=1, lane_name="cpu",
            running=[], profiles=PROFILES,
            required_units=1.0,
        )
        self.assertAlmostEqual(est.median_seconds, 0.0, places=1)
        self.assertAlmostEqual(est.p20_seconds,    0.0, places=1)

    def test_rank_2_with_congested_lane_has_positive_wait(self):
        # Rank 2 must wait for 1 slot to free — with fresh pods that takes time
        running = [_pod(deadline=3600.0, age=0.0) for _ in range(8)]
        est = estimate_wait(
            queue_rank=2, lane_name="gpu-medium",
            running=running, profiles=PROFILES,
        )
        self.assertGreater(est.median_seconds, 0.0)

    def test_higher_rank_means_longer_wait(self):
        running = [_pod(deadline=3600.0, age=600.0) for _ in range(10)]
        est1 = estimate_wait(queue_rank=1, lane_name="cpu", running=running, profiles=PROFILES)
        est5 = estimate_wait(queue_rank=5, lane_name="cpu", running=running, profiles=PROFILES)
        self.assertLess(est1.median_seconds, est5.median_seconds)

    def test_p20_le_median_le_p80(self):
        running = [_pod(deadline=3600.0, age=900.0) for _ in range(4)]
        est = estimate_wait(queue_rank=2, lane_name="gpu-small", running=running, profiles=PROFILES)
        self.assertLessEqual(est.p20_seconds,    est.median_seconds)
        self.assertLessEqual(est.median_seconds,  est.p80_seconds)

    def test_old_running_pods_predict_shorter_wait(self):
        """Pods near their deadline should free slots sooner → shorter wait."""
        fresh = [_pod(deadline=3600.0, age=100.0)  for _ in range(4)]
        old   = [_pod(deadline=3600.0, age=3400.0) for _ in range(4)]
        est_fresh = estimate_wait(queue_rank=2, lane_name="cpu", running=fresh, profiles=PROFILES)
        est_old   = estimate_wait(queue_rank=2, lane_name="cpu", running=old,   profiles=PROFILES)
        self.assertLess(est_old.median_seconds, est_fresh.median_seconds)

    def test_batch_pods_predicted_to_run_longer(self):
        """Batch profile has higher mean_pct → slots free later → longer wait."""
        interactive_pods = [_pod(deadline=7200.0, age=100.0, batch=False) for _ in range(4)]
        batch_pods       = [_pod(deadline=7200.0, age=100.0, batch=True)  for _ in range(4)]
        est_i = estimate_wait(queue_rank=2, lane_name="gpu-large", running=interactive_pods, profiles=PROFILES)
        est_b = estimate_wait(queue_rank=2, lane_name="gpu-large", running=batch_pods,       profiles=PROFILES)
        self.assertLess(est_i.median_seconds, est_b.median_seconds)

    def test_returns_correct_metadata(self):
        est = estimate_wait(queue_rank=3, lane_name="gpu-xlarge", running=[], profiles=PROFILES)
        self.assertEqual(est.queue_rank, 3)
        self.assertEqual(est.lane_name, "gpu-xlarge")

    def test_t_max_ceiling_respected(self):
        """With no running pods and rank > 1, wait should hit t_max ceiling."""
        est = estimate_wait(
            queue_rank=5, lane_name="cpu",
            running=[], profiles=PROFILES,
            t_max=120.0,
        )
        self.assertLessEqual(est.median_seconds, 120.0)


# ---------------------------------------------------------------------------
# format_wait
# ---------------------------------------------------------------------------

class TestFormatWait(unittest.TestCase):

    def test_imminent_dispatch(self):
        est = WaitEstimate(median_seconds=0.0, p20_seconds=0.0, p80_seconds=0.0,
                           queue_rank=1, lane_name="cpu")
        self.assertIn("imminent", format_wait(est))

    def test_seconds_format(self):
        est = WaitEstimate(median_seconds=45.0, p20_seconds=20.0, p80_seconds=90.0,
                           queue_rank=2, lane_name="gpu-small")
        out = format_wait(est)
        self.assertIn("45s", out)

    def test_minutes_format(self):
        est = WaitEstimate(median_seconds=300.0, p20_seconds=120.0, p80_seconds=600.0,
                           queue_rank=3, lane_name="gpu-medium")
        out = format_wait(est)
        self.assertIn("5.0m", out)

    def test_hours_format(self):
        est = WaitEstimate(median_seconds=7200.0, p20_seconds=3600.0, p80_seconds=10800.0,
                           queue_rank=5, lane_name="gpu-xlarge")
        out = format_wait(est)
        self.assertIn("2.0h", out)

    def test_contains_rank_and_lane(self):
        est = WaitEstimate(median_seconds=60.0, p20_seconds=30.0, p80_seconds=120.0,
                           queue_rank=4, lane_name="gpu-large")
        out = format_wait(est)
        self.assertIn("rank=4", out)
        self.assertIn("gpu-large", out)


# ---------------------------------------------------------------------------
# ResidencyProfile validation
# ---------------------------------------------------------------------------

class TestResidencyProfile(unittest.TestCase):

    def test_valid_profile(self):
        p = ResidencyProfile(mean_pct=0.5, std_pct=0.1)
        mu, sigma = p.for_deadline(3600.0)
        self.assertAlmostEqual(mu,    1800.0)
        self.assertAlmostEqual(sigma,  360.0)

    def test_invalid_mean_pct(self):
        with self.assertRaises(ValueError):
            ResidencyProfile(mean_pct=1.5, std_pct=0.1)
        with self.assertRaises(ValueError):
            ResidencyProfile(mean_pct=0.0, std_pct=0.1)

    def test_invalid_std_pct(self):
        with self.assertRaises(ValueError):
            ResidencyProfile(mean_pct=0.5, std_pct=0.0)
        with self.assertRaises(ValueError):
            ResidencyProfile(mean_pct=0.5, std_pct=-0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ---------------------------------------------------------------------------
# WaitTimeCache tests
# ---------------------------------------------------------------------------

class TestWaitTimeCache(unittest.TestCase):

    def _make_cache(self, estimates: dict, interval: float = 60.0,
                    fail: bool = False) -> "WaitTimeCache":
        from lane_scheduler.estimation.wait_estimator import WaitTimeCache

        def snapshot_fn():
            if fail:
                raise RuntimeError("snapshot error")
            return estimates

        return WaitTimeCache(snapshot_fn=snapshot_fn, interval=interval)

    def _dummy_estimate(self, uid="pod-1", rank=1) -> WaitEstimate:
        return WaitEstimate(
            median_seconds = 30.0 * rank,
            p20_seconds    = 15.0 * rank,
            p80_seconds    = 60.0 * rank,
            queue_rank     = rank,
            lane_name      = "cpu",
        )

    def test_get_returns_none_before_first_snapshot(self):
        cache = self._make_cache({})
        self.assertIsNone(cache.get("any-uid"))

    def test_get_returns_estimate_after_manual_snapshot(self):
        est   = self._dummy_estimate("p1", rank=2)
        cache = self._make_cache({"p1": est})
        # Drive snapshot directly without starting background thread
        cache._run_snapshot()
        result = cache.get("p1")
        self.assertIsNotNone(result)
        self.assertEqual(result.queue_rank, 2)

    def test_get_returns_none_for_unknown_pod(self):
        cache = self._make_cache({"p1": self._dummy_estimate()})
        cache._run_snapshot()
        self.assertIsNone(cache.get("does-not-exist"))

    def test_snapshot_replaces_stale_entries(self):
        """Pods removed from queue should disappear from cache on next snapshot."""
        est = self._dummy_estimate("p1")
        cache = self._make_cache({"p1": est})
        cache._run_snapshot()
        self.assertIsNotNone(cache.get("p1"))

        # Next snapshot no longer includes p1 (it was dispatched)
        cache._snapshot_fn = lambda: {}
        cache._run_snapshot()
        self.assertIsNone(cache.get("p1"))

    def test_get_with_age_returns_age(self):
        est   = self._dummy_estimate()
        cache = self._make_cache({"p1": est})
        cache._run_snapshot()
        result, age = cache.get_with_age("p1")
        self.assertIsNotNone(result)
        self.assertGreaterEqual(age, 0.0)
        self.assertLess(age, 1.0)   # computed moments ago

    def test_failed_snapshot_preserves_existing_cache(self):
        """A snapshot error must not clear previously valid estimates."""
        est   = self._dummy_estimate("p1")
        cache = self._make_cache({"p1": est})
        cache._run_snapshot()
        self.assertIsNotNone(cache.get("p1"))

        # Now simulate a failure
        cache._snapshot_fn = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        cache._run_snapshot()

        # Original estimate still present
        self.assertIsNotNone(cache.get("p1"))

    def test_len_reflects_current_cache(self):
        ests  = {f"p{i}": self._dummy_estimate(f"p{i}", rank=i+1) for i in range(5)}
        cache = self._make_cache(ests)
        cache._run_snapshot()
        self.assertEqual(len(cache), 5)

    def test_last_duration_recorded(self):
        cache = self._make_cache({"p1": self._dummy_estimate()})
        cache._run_snapshot()
        self.assertGreaterEqual(cache.last_duration, 0.0)

    def test_background_thread_populates_cache(self):
        """Integration: start the thread and confirm cache populates."""
        est   = self._dummy_estimate("p1")
        cache = self._make_cache({"p1": est}, interval=0.05)
        cache.start()
        # Wait long enough for the initial delay (min(interval, 15) = 0.05s) + snapshot
        time.sleep(0.3)
        cache.stop()
        self.assertIsNotNone(cache.get("p1"))

    def test_stop_prevents_further_snapshots(self):
        call_count = [0]

        def counting_snapshot():
            call_count[0] += 1
            return {}

        from lane_scheduler.estimation.wait_estimator import WaitTimeCache
        cache = WaitTimeCache(snapshot_fn=counting_snapshot, interval=0.05)
        cache.start()
        time.sleep(0.3)
        cache.stop()
        count_at_stop = call_count[0]
        time.sleep(0.2)
        # No further snapshots after stop
        self.assertEqual(call_count[0], count_at_stop)
