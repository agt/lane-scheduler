"""
Unit tests for event_publisher.py
No Kubernetes cluster required — core_v1 is mocked.
"""

import time
import unittest
from unittest.mock import MagicMock, call, patch

from event_publisher import (
    EventPublisher,
    PodEventSchedule,
    _format_message,
    _EARLY_INTERVAL,
    _LATE_INTERVAL,
    _EARLY_COUNT,
    _BATCH_MAX_EMITS,
    class_rank,
)
from wait_estimator import WaitEstimate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _est(uid="uid-1", rank=1, lane="cpu") -> WaitEstimate:
    return WaitEstimate(
        median_seconds = 60.0 * rank,
        p20_seconds    = 30.0 * rank,
        p80_seconds    = 120.0 * rank,
        queue_rank     = rank,
        lane_name      = lane,
    )


def _pod(name="my-pod", namespace="jsmith", uid="uid-1", rv="1000") -> dict:
    return {"metadata": {
        "name": name, "namespace": namespace,
        "uid": uid, "resourceVersion": rv,
    }}


def _mock_job(uid, class_id, score=1.0):
    j = MagicMock()
    j.job_id   = uid
    j.class_id = class_id
    return j


def _make_publisher(raises=False) -> tuple[EventPublisher, MagicMock]:
    core_v1 = MagicMock()
    if raises:
        core_v1.create_namespaced_event.side_effect = Exception("API error")
    return EventPublisher(core_v1), core_v1


# ---------------------------------------------------------------------------
# PodEventSchedule
# ---------------------------------------------------------------------------

class TestPodEventSchedule(unittest.TestCase):

    def test_due_immediately_at_zero(self):
        s = PodEventSchedule("u", "p", "ns", "rv")
        s.next_emit_at = 0.0
        self.assertTrue(s.due(now=1.0))

    def test_not_due_before_next_emit_at(self):
        now = time.monotonic()
        s   = PodEventSchedule("u", "p", "ns", "rv")
        s.next_emit_at = now + 999.0
        self.assertFalse(s.due(now=now))

    def test_advance_early_phase(self):
        now = time.monotonic()
        s   = PodEventSchedule("u", "p", "ns", "rv")
        for i in range(1, _EARLY_COUNT + 1):
            s.advance(now)
            self.assertEqual(s.emit_count, i)
            self.assertAlmostEqual(s.next_emit_at, now + _EARLY_INTERVAL, places=1)

    def test_advance_transitions_to_late_phase(self):
        now = time.monotonic()
        s   = PodEventSchedule("u", "p", "ns", "rv")
        for _ in range(_EARLY_COUNT):
            s.advance(now)
        s.advance(now)   # this is the (_EARLY_COUNT + 1)th advance
        self.assertAlmostEqual(s.next_emit_at, now + _LATE_INTERVAL, places=1)

    def test_late_phase_stays_at_late_interval(self):
        now = time.monotonic()
        s   = PodEventSchedule("u", "p", "ns", "rv")
        for _ in range(_EARLY_COUNT + 3):
            s.advance(now)
        self.assertAlmostEqual(s.next_emit_at, now + _LATE_INTERVAL, places=1)


# ---------------------------------------------------------------------------
# class_rank helper
# ---------------------------------------------------------------------------

class TestClassRank(unittest.TestCase):

    def test_returns_rank_within_class(self):
        candidates = [
            (3.0, _mock_job("u1", "CSE101")),
            (2.0, _mock_job("u2", "CSE234")),
            (1.5, _mock_job("u3", "CSE101")),
            (1.0, _mock_job("u4", "CSE101")),
        ]
        self.assertEqual(class_rank("u1", "CSE101", candidates), 1)
        self.assertEqual(class_rank("u3", "CSE101", candidates), 2)
        self.assertEqual(class_rank("u4", "CSE101", candidates), 3)

    def test_ignores_other_classes(self):
        candidates = [
            (3.0, _mock_job("u1", "CSE101")),
            (2.0, _mock_job("u2", "CSE234")),
            (1.0, _mock_job("u3", "CSE101")),
        ]
        # u2 is rank 1 within CSE234
        self.assertEqual(class_rank("u2", "CSE234", candidates), 1)

    def test_returns_none_if_not_found(self):
        candidates = [(1.0, _mock_job("u1", "CSE101"))]
        self.assertIsNone(class_rank("missing", "CSE101", candidates))


# ---------------------------------------------------------------------------
# EventPublisher registration
# ---------------------------------------------------------------------------

class TestEventPublisherRegistration(unittest.TestCase):

    def test_register_creates_schedule(self):
        pub, _ = _make_publisher()
        pod     = _pod(uid="u1")
        pub.register("u1", pod)
        with pub._lock:
            self.assertIn("u1", pub._schedules)

    def test_register_idempotent(self):
        pub, _ = _make_publisher()
        pod     = _pod(uid="u1")
        pub.register("u1", pod)
        pub.register("u1", pod)   # second call ignored
        with pub._lock:
            self.assertEqual(len(pub._schedules), 1)

    def test_deregister_removes_schedule(self):
        pub, _ = _make_publisher()
        pub.register("u1", _pod(uid="u1"))
        pub.deregister("u1")
        with pub._lock:
            self.assertNotIn("u1", pub._schedules)

    def test_deregister_unknown_uid_is_safe(self):
        pub, _ = _make_publisher()
        pub.deregister("nonexistent")   # should not raise


# ---------------------------------------------------------------------------
# EventPublisher.publish_due
# ---------------------------------------------------------------------------

class TestPublishDue(unittest.TestCase):

    def _setup(self, raises=False):
        pub, core_v1 = _make_publisher(raises=raises)
        uid  = "uid-1"
        pod  = _pod(uid=uid)
        pub.register(uid, pod)
        est  = _est(uid=uid, rank=2)
        now  = time.monotonic()
        job  = _mock_job(uid, "CSE101")
        import scheduler as _sched
        _sched.initialise_lanes(["medium"])
        lane = _sched.Lane.CPU
        return pub, core_v1, uid, est, now, job, lane

    def test_event_created_when_due(self):
        pub, core_v1, uid, est, now, job, lane = self._setup()
        n = pub.publish_due(
            estimates        = {uid: est},
            lane_candidates  = {lane: [(2.0, job)]},
            pending_snapshot = {uid: _pod(uid=uid)},
            now              = now,
        )
        self.assertEqual(n, 1)
        core_v1.create_namespaced_event.assert_called_once()

    def test_no_event_when_not_due(self):
        pub, core_v1, uid, est, now, job, lane = self._setup()
        with pub._lock:
            pub._schedules[uid].next_emit_at = now + 9999.0
        n = pub.publish_due(
            estimates        = {uid: est},
            lane_candidates  = {lane: [(2.0, job)]},
            pending_snapshot = {uid: _pod(uid=uid)},
            now              = now,
        )
        self.assertEqual(n, 0)
        core_v1.create_namespaced_event.assert_not_called()

    def test_emit_count_incremented_after_success(self):
        pub, _, uid, est, now, job, lane = self._setup()
        pub.publish_due(
            estimates        = {uid: est},
            lane_candidates  = {lane: [(2.0, job)]},
            pending_snapshot = {uid: _pod(uid=uid)},
            now              = now,
        )
        with pub._lock:
            self.assertEqual(pub._schedules[uid].emit_count, 1)

    def test_api_error_does_not_increment_emit_count(self):
        pub, _, uid, est, now, job, lane = self._setup(raises=True)
        pub.publish_due(
            estimates        = {uid: est},
            lane_candidates  = {lane: [(2.0, job)]},
            pending_snapshot = {uid: _pod(uid=uid)},
            now              = now,
        )
        with pub._lock:
            self.assertEqual(pub._schedules[uid].emit_count, 0)

    def test_pod_not_in_estimates_deregisters(self):
        pub, core_v1, uid, est, now, job, lane = self._setup()
        pub.publish_due(
            estimates        = {},
            lane_candidates  = {lane: [(2.0, job)]},
            pending_snapshot = {uid: _pod(uid=uid)},
            now              = now,
        )
        with pub._lock:
            self.assertNotIn(uid, pub._schedules)
        core_v1.create_namespaced_event.assert_not_called()

    def test_message_contains_rank_and_wait(self):
        pub, core_v1, uid, est, now, job, lane = self._setup()
        pub.publish_due(
            estimates        = {uid: est},
            lane_candidates  = {lane: [(2.0, job)]},
            pending_snapshot = {uid: _pod(uid=uid)},
            now              = now,
        )
        args, kwargs = core_v1.create_namespaced_event.call_args
        body = kwargs.get("body") or args[1]
        msg  = body["message"]
        self.assertIn("#2", msg)
        self.assertIn("cpu", msg)
        self.assertIn("Estimated wait", msg)

    def test_event_name_unique_per_emit(self):
        pub, core_v1, uid, est, now, job, lane = self._setup()
        names = set()
        for i in range(3):
            pub.publish_due(
                estimates        = {uid: est},
                lane_candidates  = {lane: [(2.0, job)]},
                pending_snapshot = {uid: _pod(uid=uid)},
                now              = now + i * _EARLY_INTERVAL,
            )
        for c in core_v1.create_namespaced_event.call_args_list:
            args, kwargs = c
            body = kwargs.get("body") or args[1]
            names.add(body["metadata"]["name"])
        self.assertEqual(len(names), 3)

    def test_multiple_pods_all_published(self):
        pub, core_v1 = _make_publisher()
        import scheduler as _sched
        _sched.initialise_lanes(["medium"])
        lane = _sched.Lane.CPU
        now  = time.monotonic()
        uids = [f"uid-{i}" for i in range(4)]
        for uid in uids:
            pub.register(uid, _pod(uid=uid, name=f"pod-{uid}"))
        estimates  = {uid: _est(uid=uid, rank=i+1) for i, uid in enumerate(uids)}
        candidates = [(float(4 - i), _mock_job(uid, "CSE101"))
                      for i, uid in enumerate(uids)]
        pub.publish_due(
            estimates        = estimates,
            lane_candidates  = {lane: candidates},
            pending_snapshot = {uid: _pod(uid=uid) for uid in uids},
            now              = now,
        )
        self.assertEqual(core_v1.create_namespaced_event.call_count, 4)

# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------

class TestFormatMessage(unittest.TestCase):

    def test_includes_overall_rank(self):
        est = _est(rank=3, lane="gpu-medium")
        msg = _format_message(est, class_rank_val=None)
        self.assertIn("#3", msg)
        self.assertIn("gpu-medium", msg)

    def test_includes_class_rank_when_provided(self):
        est = _est(rank=5, lane="cpu")
        msg = _format_message(est, class_rank_val=2)
        self.assertIn("#2 within course", msg)

    def test_omits_class_rank_when_none(self):
        est = _est(rank=1, lane="cpu")
        msg = _format_message(est, class_rank_val=None)
        self.assertNotIn("within course", msg)

    def test_includes_wait_estimate(self):
        est = WaitEstimate(
            median_seconds=300.0, p20_seconds=120.0, p80_seconds=600.0,
            queue_rank=2, lane_name="gpu-small",
        )
        msg = _format_message(est, class_rank_val=1)
        self.assertIn("5.0m", msg)    # median
        self.assertIn("2.0m", msg)    # p20
        self.assertIn("10.0m", msg)   # p80

    def test_message_is_single_string(self):
        est = _est()
        msg = _format_message(est, class_rank_val=1)
        self.assertIsInstance(msg, str)
        self.assertGreater(len(msg), 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ---------------------------------------------------------------------------
# Batch emission schedule
# ---------------------------------------------------------------------------

class TestBatchSchedule(unittest.TestCase):

    def _batch_schedule(self, enqueued_at=0.0) -> PodEventSchedule:
        s = PodEventSchedule("u", "p", "ns", "rv", batch=True)
        s.enqueued_at  = enqueued_at
        s.next_emit_at = 0.0
        return s

    def test_first_emit_is_immediate(self):
        s = self._batch_schedule(enqueued_at=1000.0)
        self.assertTrue(s.due(now=1000.0))

    def test_advance_anchors_on_first_median(self):
        s   = self._batch_schedule(enqueued_at=1000.0)
        now = 1000.0
        s.advance(now, median_wait=3600.0)
        # emit_count=1 → next milestone at 25% of 3600 from enqueued_at
        self.assertAlmostEqual(s.next_emit_at, 1000.0 + 0.25 * 3600.0)

    def test_subsequent_advances_use_anchored_wait(self):
        s   = self._batch_schedule(enqueued_at=0.0)
        now = 0.0
        s.advance(now, median_wait=4000.0)   # emit 0→1, anchor=4000
        s.advance(now, median_wait=9999.0)   # emit 1→2, median ignored
        # 50% milestone should still use the original 4000s anchor
        self.assertAlmostEqual(s.next_emit_at, 0.0 + 0.50 * 4000.0)

    def test_milestones_are_25_50_75_percent(self):
        s   = self._batch_schedule(enqueued_at=0.0)
        now = 0.0
        anchor = 8000.0
        expected = [0.25 * anchor, 0.50 * anchor, 0.75 * anchor]
        for i, exp in enumerate(expected):
            s.advance(now, median_wait=anchor)
            self.assertAlmostEqual(s.next_emit_at, exp,
                                   msg=f"milestone {i+1} wrong")

    def test_exhausted_after_four_emits(self):
        s   = self._batch_schedule(enqueued_at=0.0)
        now = 0.0
        for _ in range(_BATCH_MAX_EMITS):
            s.advance(now, median_wait=3600.0)
        self.assertTrue(s.exhausted)
        self.assertFalse(s.due(now=now + 1e9))

    def test_past_milestone_fires_immediately(self):
        """If the milestone time is already past when computed, fire at once."""
        s   = self._batch_schedule(enqueued_at=0.0)
        # Pod has been waiting 3000s; median is only 100s → all milestones in past
        now = 3000.0
        s.advance(now, median_wait=100.0)   # 25% of 100 = 25s from enqueue = past
        self.assertLessEqual(s.next_emit_at, now)
        self.assertTrue(s.due(now=now))

    def test_interactive_schedule_unchanged(self):
        """Interactive pod advance behaviour must not regress."""
        s = PodEventSchedule("u", "p", "ns", "rv", batch=False)
        s.next_emit_at = 0.0
        now = 100.0
        for i in range(1, _EARLY_COUNT + 1):
            s.advance(now)
            self.assertEqual(s.emit_count, i)
            self.assertAlmostEqual(s.next_emit_at, now + _EARLY_INTERVAL)
        s.advance(now)
        self.assertAlmostEqual(s.next_emit_at, now + _LATE_INTERVAL)
        self.assertFalse(s.exhausted)


class TestBatchPublishDue(unittest.TestCase):
    """Integration: batch pod events emitted with correct count and timing."""

    def _batch_pod(self, uid="bu-1", name="batch-pod") -> dict:
        return {"metadata": {
            "name": name, "namespace": "jsmith",
            "uid": uid, "resourceVersion": "42",
            "labels": {"dsmlp/batch": "true"},
        }}

    def _run_publish(self, pub, uid, est, now, lane, job):
        return pub.publish_due(
            estimates        = {uid: est},
            lane_candidates  = {lane: [(1.0, job)]},
            pending_snapshot = {uid: self._batch_pod(uid=uid)},
            now              = now,
        )

    def test_batch_pod_emits_exactly_four_events(self):
        """One immediate + three milestones = four total."""
        core_v1 = MagicMock()
        pub     = EventPublisher(core_v1)
        import scheduler as _sched
        _sched.initialise_lanes(["medium"])
        lane    = _sched.Lane.CPU
        uid     = "bu-1"
        job     = _mock_job(uid, "CSE234")
        median  = 4000.0
        est     = WaitEstimate(median_seconds=median, p20_seconds=2000.0,
                               p80_seconds=6000.0, queue_rank=1, lane_name="cpu")

        pub.register(uid, self._batch_pod(uid=uid))
        enqueued_at = pub._schedules[uid].enqueued_at

        # Emit 0: immediate
        now = enqueued_at
        self._run_publish(pub, uid, est, now, lane, job)
        self.assertEqual(core_v1.create_namespaced_event.call_count, 1)

        # Emit 1: 25% milestone
        now = enqueued_at + 0.25 * median
        self._run_publish(pub, uid, est, now, lane, job)
        self.assertEqual(core_v1.create_namespaced_event.call_count, 2)

        # Emit 2: 50% milestone
        now = enqueued_at + 0.50 * median
        self._run_publish(pub, uid, est, now, lane, job)
        self.assertEqual(core_v1.create_namespaced_event.call_count, 3)

        # Emit 3: 75% milestone
        now = enqueued_at + 0.75 * median
        self._run_publish(pub, uid, est, now, lane, job)
        self.assertEqual(core_v1.create_namespaced_event.call_count, 4)

        # No further events even if much later
        now = enqueued_at + 10 * median
        self._run_publish(pub, uid, est, now, lane, job)
        self.assertEqual(core_v1.create_namespaced_event.call_count, 4)

    def test_batch_milestones_stable_despite_changing_estimate(self):
        """Milestone times must not shift when median_wait changes."""
        core_v1 = MagicMock()
        pub     = EventPublisher(core_v1)
        import scheduler as _sched
        _sched.initialise_lanes(["medium"])
        lane    = _sched.Lane.CPU
        uid     = "bu-2"
        job     = _mock_job(uid, "CSE234")

        pub.register(uid, self._batch_pod(uid=uid))
        enqueued_at = pub._schedules[uid].enqueued_at
        original_median = 4000.0

        # Emit 0 with original median
        self._run_publish(pub, uid,
            WaitEstimate(original_median, 2000.0, 6000.0, 1, "cpu"),
            enqueued_at, lane, job)

        # Record the anchored milestone time for emit 1
        with pub._lock:
            milestone_1 = pub._schedules[uid].next_emit_at

        # Emit 1 with a very different median — milestone should be unchanged
        self._run_publish(pub, uid,
            WaitEstimate(99999.0, 0.0, 99999.0, 1, "cpu"),
            milestone_1, lane, job)

        self.assertAlmostEqual(milestone_1,
                               enqueued_at + 0.25 * original_median, places=1)
