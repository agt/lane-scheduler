"""
Tests for ResidencyStore and the seed()/dump() helpers on ResidencyStats.
"""

import math
import tempfile
import unittest
from pathlib import Path

from lane_scheduler.estimation.residency_store import ResidencyStore
from lane_scheduler.estimation.residency_stats import ResidencyStats, DEFAULT_EWMA_ALPHA
from lane_scheduler.estimation.wait_estimator import ResidencyProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTERACTIVE = ResidencyProfile(mean_pct=0.4, std_pct=0.2)
BATCH       = ResidencyProfile(mean_pct=0.7, std_pct=0.15)


def _stats(**kw) -> ResidencyStats:
    return ResidencyStats(
        interactive_prior=INTERACTIVE,
        batch_prior=BATCH,
        prior_weight=kw.pop("prior_weight", 10.0),
        ewma_alpha=kw.pop("ewma_alpha", DEFAULT_EWMA_ALPHA),
        **kw,
    )


def _store(tmp: Path) -> ResidencyStore:
    return ResidencyStore(tmp / "residency.db")


# ---------------------------------------------------------------------------
# ResidencyStore: basic I/O
# ---------------------------------------------------------------------------

class TestResidencyStoreIO(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_load_empty_db_returns_empty_list(self):
        store = _store(self.tmp)
        self.assertEqual(store.load(), [])

    def test_save_and_load_roundtrip(self):
        store = _store(self.tmp)
        rows = [
            ("CSE101", "cpu",        False, 0.42, 0.03, 15),
            ("CSE234", "gpu-medium", True,  0.75, 0.01, 8),
        ]
        store.save(rows)
        loaded = store.load()
        self.assertEqual(len(loaded), 2)
        by_key = {(r[0], r[1], r[2]): r for r in loaded}
        r0 = by_key[("CSE101", "cpu", False)]
        self.assertAlmostEqual(r0[3], 0.42)
        self.assertAlmostEqual(r0[4], 0.03)
        self.assertEqual(r0[5], 15)
        r1 = by_key[("CSE234", "gpu-medium", True)]
        self.assertAlmostEqual(r1[3], 0.75)
        self.assertEqual(r1[5], 8)

    def test_upsert_updates_existing_row(self):
        store = _store(self.tmp)
        store.save([("CSE101", "cpu", False, 0.40, 0.02, 5)])
        store.save([("CSE101", "cpu", False, 0.55, 0.04, 10)])
        rows = store.load()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][3], 0.55)
        self.assertEqual(rows[0][5], 10)

    def test_save_skips_n_zero_rows(self):
        store = _store(self.tmp)
        store.save([("CSE101", "cpu", False, 0.0, 0.0, 0)])
        self.assertEqual(store.load(), [])

    def test_multiple_saves_accumulate_distinct_strata(self):
        store = _store(self.tmp)
        store.save([("A", "cpu", False, 0.3, 0.01, 3)])
        store.save([("B", "gpu-large", True, 0.8, 0.02, 7)])
        self.assertEqual(len(store.load()), 2)

    def test_db_created_in_subdirectory(self):
        nested = self.tmp / "sub" / "dir"
        store = ResidencyStore(nested / "residency.db")
        store.save([("C", "cpu", False, 0.5, 0.01, 1)])
        self.assertEqual(len(store.load()), 1)

    def test_load_preserves_batch_flag_type(self):
        store = _store(self.tmp)
        store.save([
            ("X", "cpu", False, 0.3, 0.01, 2),
            ("X", "cpu", True,  0.7, 0.02, 4),
        ])
        rows = store.load()
        flags = {r[2] for r in rows}
        self.assertIn(True,  flags)
        self.assertIn(False, flags)
        for _, _, batch, _, _, _ in rows:
            self.assertIsInstance(batch, bool)


# ---------------------------------------------------------------------------
# ResidencyStats: seed() and dump()
# ---------------------------------------------------------------------------

class TestSeedAndDump(unittest.TestCase):

    def test_seed_sets_ewma_state(self):
        stats = _stats()
        stats.seed("CSE101", "cpu", False, ewma_mean=0.55, ewma_var=0.03, n=20)
        self.assertEqual(stats.observation_count("CSE101", "cpu", False), 20)

    def test_seed_affects_profile(self):
        stats = _stats(prior_weight=10.0)
        # Seed with a mean far from the prior (0.4 → 0.9)
        stats.seed("CSE101", "cpu", False, ewma_mean=0.9, ewma_var=0.01, n=100)
        profile = stats.profile_for("CSE101", "cpu", False)
        # With 100 obs vs prior_weight=10, posterior should be close to 0.9
        self.assertGreater(profile.mean_pct, 0.7)

    def test_seed_ignored_if_stratum_already_has_data(self):
        stats = _stats()
        stats.record("CSE101", "cpu", False, residency_pct=0.3)
        n_before = stats.observation_count("CSE101", "cpu", False)
        stats.seed("CSE101", "cpu", False, ewma_mean=0.9, ewma_var=0.0, n=999)
        # seed() should not overwrite live data
        self.assertEqual(stats.observation_count("CSE101", "cpu", False), n_before)

    def test_seed_n_zero_is_noop(self):
        stats = _stats()
        stats.seed("CSE101", "cpu", False, ewma_mean=0.9, ewma_var=0.0, n=0)
        self.assertEqual(stats.observation_count("CSE101", "cpu", False), 0)

    def test_dump_empty_when_no_observations(self):
        stats = _stats()
        self.assertEqual(stats.dump(), [])

    def test_dump_includes_seeded_strata(self):
        stats = _stats()
        stats.seed("CSE101", "cpu",        False, 0.42, 0.03, 15)
        stats.seed("CSE234", "gpu-medium", True,  0.75, 0.01, 8)
        rows = stats.dump()
        self.assertEqual(len(rows), 2)

    def test_dump_includes_live_observations(self):
        stats = _stats()
        for _ in range(5):
            stats.record("CSE101", "cpu", False, 0.5)
        rows = stats.dump()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "CSE101")
        self.assertEqual(rows[0][5], 5)  # n == 5

    def test_dump_row_fields(self):
        stats = _stats()
        stats.seed("CSE101", "cpu", False, ewma_mean=0.5, ewma_var=0.02, n=10)
        rows = stats.dump()
        course_id, lane_name, batch, mean, var, n = rows[0]
        self.assertEqual(course_id, "CSE101")
        self.assertEqual(lane_name, "cpu")
        self.assertIsInstance(batch, bool)
        self.assertAlmostEqual(mean, 0.5)
        self.assertAlmostEqual(var,  0.02)
        self.assertEqual(n, 10)


# ---------------------------------------------------------------------------
# Full round-trip: seed from store, record more, dump back
# ---------------------------------------------------------------------------

class TestRoundTrip(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_seed_dump_save_load_cycle(self):
        store = _store(self.tmp)

        # First run: accumulate 20 observations, dump to DB
        stats1 = _stats()
        for _ in range(20):
            stats1.record("CSE234", "gpu-large", False, residency_pct=0.65)
        store.save(stats1.dump())

        # Second run: seed from DB, check state is restored
        stats2 = _stats()
        for course_id, lane_name, batch, mean, var, n in store.load():
            stats2.seed(course_id, lane_name, batch, mean, var, n)

        self.assertEqual(stats2.observation_count("CSE234", "gpu-large", False), 20)
        p1 = stats1.profile_for("CSE234", "gpu-large", False)
        p2 = stats2.profile_for("CSE234", "gpu-large", False)
        self.assertAlmostEqual(p1.mean_pct, p2.mean_pct, places=6)
        self.assertAlmostEqual(p1.std_pct,  p2.std_pct,  places=6)

    def test_continued_learning_after_restore(self):
        store = _store(self.tmp)

        stats1 = _stats(prior_weight=10.0)
        for _ in range(50):
            stats1.record("CSE101", "cpu", False, 0.8)
        store.save(stats1.dump())

        stats2 = _stats(prior_weight=10.0)
        for course_id, lane_name, batch, mean, var, n in store.load():
            stats2.seed(course_id, lane_name, batch, mean, var, n)

        # Record 50 more in the restored stats; mean should still be near 0.8
        for _ in range(50):
            stats2.record("CSE101", "cpu", False, 0.8)

        profile = stats2.profile_for("CSE101", "cpu", False)
        self.assertGreater(profile.mean_pct, 0.7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
