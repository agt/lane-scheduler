"""
Unit tests for residency_stats.py
"""

import math
import unittest

from lane_scheduler.estimation.wait_estimator import ResidencyProfile
from lane_scheduler.estimation.residency_stats import (
    ResidencyStats, _EWMA, DEFAULT_PRIOR_WEIGHT, DEFAULT_EWMA_ALPHA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTERACTIVE = ResidencyProfile(mean_pct=0.4, std_pct=0.2)
BATCH       = ResidencyProfile(mean_pct=0.7, std_pct=0.15)


def _stats(prior_weight=10.0, ewma_alpha=DEFAULT_EWMA_ALPHA) -> ResidencyStats:
    return ResidencyStats(
        interactive_prior = INTERACTIVE,
        batch_prior       = BATCH,
        prior_weight      = prior_weight,
        ewma_alpha        = ewma_alpha,
    )


# ---------------------------------------------------------------------------
# EWMA accumulator
# ---------------------------------------------------------------------------

class TestEWMA(unittest.TestCase):

    def test_single_observation(self):
        e = _EWMA(alpha=0.1)
        e.update(0.5)
        self.assertEqual(e.n, 1)
        self.assertAlmostEqual(e.mean, 0.5)
        self.assertAlmostEqual(e.variance, 0.0)

    def test_mean_tracks_recent_shift(self):
        # Seed with 0.0, then switch to 1.0 — EWMA mean should rise.
        e = _EWMA(alpha=0.3)
        for _ in range(20):
            e.update(0.0)
        mean_before = e.mean
        for _ in range(10):
            e.update(1.0)
        self.assertGreater(e.mean, mean_before)
        self.assertLess(e.mean, 1.0)

    def test_constant_stream_converges_to_value(self):
        e = _EWMA(alpha=0.2)
        for _ in range(200):
            e.update(0.7)
        self.assertAlmostEqual(e.mean, 0.7, places=5)

    def test_variance_positive_with_variation(self):
        e = _EWMA(alpha=0.2)
        for v in (0.2, 0.8, 0.2, 0.8):
            e.update(v)
        self.assertGreater(e.variance, 0.0)

    def test_std_is_sqrt_variance(self):
        e = _EWMA(alpha=0.1)
        for v in (0.1, 0.5, 0.9):
            e.update(v)
        self.assertAlmostEqual(e.std, math.sqrt(e.variance), places=10)

    def test_monotone_count(self):
        e = _EWMA(alpha=0.1)
        for i in range(20):
            e.update(float(i) / 20)
            self.assertEqual(e.n, i + 1)

    def test_higher_alpha_adapts_faster(self):
        # Both start at 0.0, then see only 1.0 observations.
        e_slow = _EWMA(alpha=0.05)
        e_fast = _EWMA(alpha=0.5)
        for _ in range(5):
            e_slow.update(0.0)
            e_fast.update(0.0)
        for _ in range(5):
            e_slow.update(1.0)
            e_fast.update(1.0)
        self.assertGreater(e_fast.mean, e_slow.mean)


# ---------------------------------------------------------------------------
# Prior-only behaviour
# ---------------------------------------------------------------------------

class TestPriorOnly(unittest.TestCase):

    def test_no_observations_returns_prior_interactive(self):
        stats   = _stats()
        profile = stats.profile_for("CSE101", "cpu", batch=False)
        self.assertAlmostEqual(profile.mean_pct, INTERACTIVE.mean_pct)
        self.assertAlmostEqual(profile.std_pct,  INTERACTIVE.std_pct)

    def test_no_observations_returns_prior_batch(self):
        stats   = _stats()
        profile = stats.profile_for("CSE234", "gpu-large", batch=True)
        self.assertAlmostEqual(profile.mean_pct, BATCH.mean_pct)
        self.assertAlmostEqual(profile.std_pct,  BATCH.std_pct)

    def test_unknown_course_returns_prior(self):
        stats = _stats()
        stats.record("CSE101", "cpu", batch=False, residency_pct=0.6)
        # Different course — should still get pure prior
        profile = stats.profile_for("CSE999", "cpu", batch=False)
        self.assertAlmostEqual(profile.mean_pct, INTERACTIVE.mean_pct)

    def test_observation_count_zero_before_any_records(self):
        stats = _stats()
        self.assertEqual(stats.observation_count("CSE101", "cpu", False), 0)


# ---------------------------------------------------------------------------
# Shrinkage behaviour
# ---------------------------------------------------------------------------

class TestShrinkage(unittest.TestCase):

    def test_single_observation_stays_near_prior(self):
        stats = _stats(prior_weight=10.0)
        # One observation far from prior (1.0 vs prior mean 0.4)
        stats.record("CSE101", "cpu", batch=False, residency_pct=1.0)
        profile = stats.profile_for("CSE101", "cpu", batch=False)
        # Posterior mean should be between prior and observation, closer to prior
        self.assertGreater(profile.mean_pct, INTERACTIVE.mean_pct)
        self.assertLess(profile.mean_pct, 1.0)
        # With weight=10, n=1: posterior = (10*0.4 + 1*1.0)/11 ≈ 0.454
        expected = (10 * 0.4 + 1 * 1.0) / 11
        self.assertAlmostEqual(profile.mean_pct, expected, places=4)

    def test_many_observations_converge_to_course_mean(self):
        stats = _stats(prior_weight=10.0)
        course_mean = 0.9
        for _ in range(1000):
            stats.record("CSE234", "gpu-medium", batch=False,
                         residency_pct=course_mean)
        profile = stats.profile_for("CSE234", "gpu-medium", batch=False)
        # With 1000 obs vs weight=10, should be very close to course_mean
        self.assertAlmostEqual(profile.mean_pct, course_mean, places=2)

    def test_higher_prior_weight_slower_convergence(self):
        stats_low  = _stats(prior_weight=1.0)
        stats_high = _stats(prior_weight=100.0)
        for s in (stats_low, stats_high):
            for _ in range(10):
                s.record("CSE101", "cpu", batch=False, residency_pct=0.9)
        p_low  = stats_low.profile_for("CSE101",  "cpu", batch=False)
        p_high = stats_high.profile_for("CSE101", "cpu", batch=False)
        # Low prior weight → posterior mean closer to 0.9
        # High prior weight → posterior mean stays closer to 0.4
        self.assertGreater(p_low.mean_pct, p_high.mean_pct)

    def test_posterior_mean_always_between_prior_and_ewma(self):
        stats = _stats(prior_weight=5.0)
        for pct in (0.1, 0.3, 0.8, 1.0):
            stats.record("CSE150", "cpu", batch=False, residency_pct=pct)
        profile   = stats.profile_for("CSE150", "cpu", batch=False)
        acc       = stats._strata[list(stats._strata.keys())[0]]
        # Posterior is a weighted blend of prior_mean and EWMA mean.
        lo = min(INTERACTIVE.mean_pct, acc.mean)
        hi = max(INTERACTIVE.mean_pct, acc.mean)
        self.assertGreaterEqual(profile.mean_pct, lo - 1e-9)
        self.assertLessEqual(profile.mean_pct,    hi + 1e-9)

    def test_batch_and_interactive_tracked_independently(self):
        stats = _stats()
        for _ in range(20):
            stats.record("CSE234", "gpu-large", batch=False, residency_pct=0.3)
            stats.record("CSE234", "gpu-large", batch=True,  residency_pct=0.95)
        p_int   = stats.profile_for("CSE234", "gpu-large", batch=False)
        p_batch = stats.profile_for("CSE234", "gpu-large", batch=True)
        # Interactive should have lower mean than batch for this course
        self.assertLess(p_int.mean_pct, p_batch.mean_pct)

    def test_different_lanes_tracked_independently(self):
        stats = _stats()
        for _ in range(20):
            stats.record("CSE101", "cpu",        batch=False, residency_pct=0.2)
            stats.record("CSE101", "gpu-medium", batch=False, residency_pct=0.9)
        p_cpu = stats.profile_for("CSE101", "cpu",        batch=False)
        p_gpu = stats.profile_for("CSE101", "gpu-medium", batch=False)
        self.assertLess(p_cpu.mean_pct, p_gpu.mean_pct)


# ---------------------------------------------------------------------------
# Clamping and edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_residency_clamped_to_zero_one(self):
        stats = _stats()
        stats.record("C", "cpu", batch=False, residency_pct=1.5)   # over 1
        stats.record("C", "cpu", batch=False, residency_pct=-0.1)  # under 0
        # Both should be clamped; mean should be between 0 and 1
        profile = stats.profile_for("C", "cpu", batch=False)
        self.assertGreaterEqual(profile.mean_pct, 0.0)
        self.assertLessEqual(profile.mean_pct,    1.0)

    def test_posterior_mean_in_valid_range(self):
        stats = _stats()
        for _ in range(5):
            stats.record("C", "cpu", batch=False, residency_pct=0.99)
        p = stats.profile_for("C", "cpu", batch=False)
        self.assertGreater(p.mean_pct, 0.0)
        self.assertLess(p.mean_pct,    1.0)

    def test_posterior_std_positive(self):
        stats = _stats()
        # All identical observations → sample variance = 0, but prior variance
        # contributes, so posterior std should still be positive
        for _ in range(50):
            stats.record("C", "cpu", batch=False, residency_pct=0.5)
        p = stats.profile_for("C", "cpu", batch=False)
        self.assertGreater(p.std_pct, 0.0)

    def test_all_profiles_returns_only_observed_strata(self):
        stats = _stats()
        stats.record("CSE101", "cpu",     batch=False, residency_pct=0.4)
        stats.record("CSE234", "gpu-big", batch=True,  residency_pct=0.8)
        profiles = stats.all_profiles()
        self.assertEqual(len(profiles), 2)
        self.assertIn(("CSE101", "cpu",     False), profiles)
        self.assertIn(("CSE234", "gpu-big", True),  profiles)

    def test_observation_count_increments(self):
        stats = _stats()
        for i in range(7):
            stats.record("CSE101", "cpu", batch=False, residency_pct=0.5)
            self.assertEqual(
                stats.observation_count("CSE101", "cpu", False), i + 1
            )

    def test_thread_safety_concurrent_updates(self):
        """Multiple threads recording should not corrupt the accumulator."""
        import threading
        stats = _stats()
        errors = []

        def worker():
            try:
                for _ in range(100):
                    stats.record("CSE101", "cpu", batch=False,
                                 residency_pct=0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(
            stats.observation_count("CSE101", "cpu", False), 800
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
