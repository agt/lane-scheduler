"""
Tests for lane_scheduler.web.snapshot.build_snapshot().

Uses a real Scheduler + SchedGroupRegistry + ResidencyStats/WaitTimeCache but
mocks the Kubernetes client so no cluster is needed.
"""

import time
import unittest
from unittest.mock import MagicMock

from lane_scheduler.core.scheduler import (
    SchedGroup, Job, SchedulerConfig, Scheduler,
    initialise_lanes, lane_for_gpu_class,
)
from lane_scheduler.core.sched_group_registry import SchedGroupRegistry
from lane_scheduler.core.node_capacity import NodeCapacityTracker
from lane_scheduler.estimation.wait_estimator import ResidencyProfile, WaitTimeCache
from lane_scheduler.estimation.residency_stats import ResidencyStats


def setUpModule():
    initialise_lanes(["small"])


def _small():
    return lane_for_gpu_class("small")


def _make_controller(cycle_interval=10.0):
    """Build a LaneSchedulerController with a mock K8s client."""
    from lane_scheduler.k8s.controller import LaneSchedulerController

    core_v1 = MagicMock()
    registry = SchedGroupRegistry()
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


def _register(ctrl, sched_group_id):
    """Register a scheduling group with the scheduler (registry always returns weight=1.0)."""
    group = SchedGroup(sched_group_id=sched_group_id, weight=1.0)
    ctrl.scheduler.register_group(group)
    return group


def _submit(ctrl, sched_group_id, lane, job_id="J1", username="s1", batch=False):
    job = Job(
        job_id=job_id,
        sched_group_id=sched_group_id,
        username=username,
        lane=lane,
        batch=batch,
        resource_units=1.0,
    )
    job.submit_time = time.monotonic() - 30.0
    ctrl.scheduler.submit(job)
    return job


class TestBuildSnapshotStructure(unittest.TestCase):

    def setUp(self):
        self.ctrl = _make_controller()

    def test_required_top_level_keys(self):
        from lane_scheduler.web.snapshot import build_snapshot
        snap = build_snapshot(self.ctrl)
        for key in ("generated_at", "system", "lanes", "sched_groups", "running_pods"):
            self.assertIn(key, snap)

    def test_running_pods_is_list(self):
        from lane_scheduler.web.snapshot import build_snapshot
        snap = build_snapshot(self.ctrl)
        self.assertIsInstance(snap["running_pods"], list)

    def test_system_keys(self):
        from lane_scheduler.web.snapshot import build_snapshot
        sys = build_snapshot(self.ctrl)["system"]
        for key in ("cycle_interval_s", "wait_cache_age_s",
                    "wait_cache_duration_s", "sched_group_count"):
            self.assertIn(key, sys)

    def test_cycle_interval_propagated(self):
        from lane_scheduler.web.snapshot import build_snapshot
        ctrl = _make_controller(cycle_interval=30.0)
        snap = build_snapshot(ctrl)
        self.assertEqual(snap["system"]["cycle_interval_s"], 30.0)

    def test_lanes_list_has_gpu_lanes(self):
        from lane_scheduler.web.snapshot import build_snapshot
        snap = build_snapshot(self.ctrl)
        names = {ln["name"] for ln in snap["lanes"]}
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

    def setUp(self):
        self.ctrl = _make_controller()

    def test_no_sched_groups_in_output(self):
        from lane_scheduler.web.snapshot import build_snapshot
        snap = build_snapshot(self.ctrl)
        self.assertEqual(snap["sched_groups"], [])

    def test_queued_count_zero(self):
        from lane_scheduler.web.snapshot import build_snapshot
        snap = build_snapshot(self.ctrl)
        for row in snap["lanes"]:
            self.assertEqual(row["queued_count"], 0)
            self.assertIsNone(row["drain_p80_s"])


class TestBuildSnapshotWithJobs(unittest.TestCase):

    def setUp(self):
        self.ctrl = _make_controller()
        _register(self.ctrl, "CSE101")
        _register(self.ctrl, "CSE250")

    def test_queued_group_appears(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE101", _small(), job_id="J1", username="s1")
        snap = build_snapshot(self.ctrl)
        group_ids = {g["sched_group_id"] for g in snap["sched_groups"]}
        self.assertIn("CSE101", group_ids)

    def test_queued_count_matches(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE101", _small(), job_id="J1", username="s1")
        _submit(self.ctrl, "CSE101", _small(), job_id="J2", username="s2")
        snap = build_snapshot(self.ctrl)
        cse101 = next(g for g in snap["sched_groups"] if g["sched_group_id"] == "CSE101")
        small_lane = next(l for l in cse101["lanes"] if l["lane_name"] == "gpu-small")
        self.assertEqual(small_lane["queued_count"], 2)

    def test_lane_queued_count_matches(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE101", _small(), job_id="J1", username="s1")
        _submit(self.ctrl, "CSE250", _small(), job_id="J2", username="s2")
        snap = build_snapshot(self.ctrl)
        small_row = next(r for r in snap["lanes"] if r["name"] == "gpu-small")
        self.assertEqual(small_row["queued_count"], 2)

    def test_group_weight_is_one(self):
        """SchedGroupRegistry always returns weight=1.0."""
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE250", _small(), job_id="J1", username="s1")
        snap = build_snapshot(self.ctrl)
        cse250 = next(g for g in snap["sched_groups"] if g["sched_group_id"] == "CSE250")
        self.assertAlmostEqual(cse250["weight"], 1.0, places=2)

    def test_tail_wait_absent_for_single_job(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE101", _small(), job_id="J1", username="s1")
        snap = build_snapshot(self.ctrl)
        cse101 = next(g for g in snap["sched_groups"] if g["sched_group_id"] == "CSE101")
        small_lane = next(l for l in cse101["lanes"] if l["lane_name"] == "gpu-small")
        self.assertEqual(small_lane["queued_count"], 1)
        self.assertIsNone(small_lane["tail_wait"])

    def test_tail_wait_present_for_multiple_jobs(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE101", _small(), job_id="J1", username="s1")
        _submit(self.ctrl, "CSE101", _small(), job_id="J2", username="s2")
        snap = build_snapshot(self.ctrl)
        cse101 = next(g for g in snap["sched_groups"] if g["sched_group_id"] == "CSE101")
        small_lane = next(l for l in cse101["lanes"] if l["lane_name"] == "gpu-small")
        self.assertEqual(small_lane["queued_count"], 2)
        self.assertIn("tail_wait", small_lane)

    def test_sorted_by_queued_descending(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE250", _small(), job_id="J1", username="s1")
        _submit(self.ctrl, "CSE101", _small(), job_id="J2", username="s2")
        _submit(self.ctrl, "CSE101", _small(), job_id="J3", username="s3")
        snap = build_snapshot(self.ctrl)
        queued = [
            sum(l["queued_count"] for l in g["lanes"])
            for g in snap["sched_groups"]
        ]
        self.assertEqual(queued, sorted(queued, reverse=True))


class TestRunningPodsDebugView(unittest.TestCase):

    def setUp(self):
        self.ctrl = _make_controller()

    def _inject_running(self, uid, username, lane, resource_units=1.0,
                        sched_group_id=None, batch=False, running_s=60):
        from lane_scheduler.estimation.wait_estimator import RunningPod
        rp = RunningPod(
            pod_uid=uid,
            start_time=time.monotonic() - running_s,
            active_deadline_seconds=3600,
            batch=batch,
            resource_units=resource_units,
        )
        with self.ctrl._running_lock:
            self.ctrl._kubernetes_running.setdefault(lane, {})[uid] = rp
            self.ctrl._kubernetes_running_user[uid] = username
        with self.ctrl._running_ctx_lock:
            self.ctrl._running_ctx[uid] = (
                sched_group_id or "__unlabelled__", lane, batch, 3600.0,
                time.monotonic()
            )

    def test_running_pod_appears_in_list(self):
        from lane_scheduler.web.snapshot import build_snapshot
        self._inject_running("uid-1", "jsmith", _small(), resource_units=2.0,
                             sched_group_id="CSE101")
        pods = build_snapshot(self.ctrl)["running_pods"]
        self.assertEqual(len(pods), 1)
        p = pods[0]
        self.assertEqual(p["uid"], "uid-1")
        self.assertEqual(p["username"], "jsmith")
        self.assertEqual(p["lane"], _small())
        self.assertAlmostEqual(p["resource_units"], 2.0)
        self.assertEqual(p["sched_group_id"], "CSE101")
        self.assertFalse(p["batch"])

    def test_unlabelled_pod_has_null_sched_group_id(self):
        from lane_scheduler.web.snapshot import build_snapshot
        self._inject_running("uid-2", "s7ko", _small(), sched_group_id=None)
        pods = build_snapshot(self.ctrl)["running_pods"]
        self.assertEqual(len(pods), 1)
        self.assertIsNone(pods[0]["sched_group_id"])

    def test_pod_row_has_required_keys(self):
        from lane_scheduler.web.snapshot import build_snapshot
        self._inject_running("uid-3", "ns1", _small())
        pods = build_snapshot(self.ctrl)["running_pods"]
        self.assertEqual(len(pods), 1)
        for key in ("uid", "username", "lane", "resource_units",
                    "sched_group_id", "batch", "running_s", "deadline_s"):
            self.assertIn(key, pods[0], f"missing key {key!r}")

    def test_sorted_by_lane_then_longest_running(self):
        from lane_scheduler.web.snapshot import build_snapshot
        self._inject_running("uid-a", "ns1", _small(), running_s=30)
        self._inject_running("uid-b", "ns2", _small(), running_s=120)
        pods = build_snapshot(self.ctrl)["running_pods"]
        self.assertEqual(len(pods), 2)
        self.assertGreaterEqual(pods[0]["running_s"], pods[1]["running_s"])

    def test_empty_when_no_running_pods(self):
        from lane_scheduler.web.snapshot import build_snapshot
        pods = build_snapshot(self.ctrl)["running_pods"]
        self.assertEqual(pods, [])


class TestNoPrivateIdentifiers(unittest.TestCase):

    def setUp(self):
        self.ctrl = _make_controller()
        _register(self.ctrl, "CSE101")

    def test_no_usernames_in_queued_output(self):
        from lane_scheduler.web.snapshot import build_snapshot
        _submit(self.ctrl, "CSE101", _small(), job_id="pod-uid-abc", username="alice")
        snap = build_snapshot(self.ctrl)
        # sched_groups output must not include usernames or job IDs
        dump = str(snap["sched_groups"])
        self.assertNotIn("alice", dump)
        self.assertNotIn("pod-uid-abc", dump)

    def test_unlabelled_group_excluded(self):
        from lane_scheduler.web.snapshot import build_snapshot
        from lane_scheduler.k8s.pod_translator import NO_SCHED_GROUP_LABEL
        ctrl = _make_controller()
        group = SchedGroup(sched_group_id=NO_SCHED_GROUP_LABEL, weight=1.0)
        ctrl.scheduler._groups[NO_SCHED_GROUP_LABEL] = group
        job = Job(job_id="J-unlabelled", sched_group_id=NO_SCHED_GROUP_LABEL,
                  username="s1", lane=_small())
        job.submit_time = time.monotonic()
        ctrl.scheduler._queues[_small()][NO_SCHED_GROUP_LABEL]["s1"].append(job)
        snap = build_snapshot(ctrl)
        group_ids = {g["sched_group_id"] for g in snap["sched_groups"]}
        self.assertNotIn(NO_SCHED_GROUP_LABEL, group_ids)


class TestWaitCacheAllEstimates(unittest.TestCase):

    def test_empty_cache(self):
        cache = WaitTimeCache(snapshot_fn=lambda: {})
        self.assertEqual(cache.all_estimates(), {})

    def test_populated_cache(self):
        from lane_scheduler.estimation.wait_estimator import WaitEstimate
        est = WaitEstimate(median_seconds=60.0, p20_seconds=30.0,
                           p80_seconds=120.0, queue_rank=1, lane_name="gpu-small")
        cache = WaitTimeCache(snapshot_fn=lambda: {"uid-1": est})
        from lane_scheduler.estimation.wait_estimator import CacheEntry
        cache._cache = {"uid-1": CacheEntry(estimate=est, computed_at=time.monotonic())}
        result = cache.all_estimates()
        self.assertIn("uid-1", result)
        self.assertIs(result["uid-1"], est)

    def test_returns_copy(self):
        from lane_scheduler.estimation.wait_estimator import WaitEstimate, CacheEntry
        est = WaitEstimate(median_seconds=60.0, p20_seconds=30.0,
                           p80_seconds=120.0, queue_rank=1, lane_name="gpu-small")
        cache = WaitTimeCache(snapshot_fn=lambda: {})
        cache._cache = {"uid-1": CacheEntry(estimate=est, computed_at=time.monotonic())}
        r1 = cache.all_estimates()
        cache._cache = {}
        r2 = cache.all_estimates()
        self.assertIn("uid-1", r1)
        self.assertEqual(r2, {})


class TestControllerDryRun(unittest.TestCase):

    def _make_dry_controller(self):
        from lane_scheduler.k8s.controller import LaneSchedulerController
        from lane_scheduler.estimation.wait_estimator import ResidencyProfile
        core_v1 = MagicMock()
        registry = SchedGroupRegistry()
        sched_config = SchedulerConfig()
        residency_profiles = {
            "interactive": ResidencyProfile(mean_pct=0.4, std_pct=0.2),
            "batch":       ResidencyProfile(mean_pct=0.7, std_pct=0.15),
        }
        return LaneSchedulerController(
            core_v1=core_v1,
            registry=registry,
            sched_config=sched_config,
            residency_profiles=residency_profiles,
            dry_run=True,
            web_port=0,
        ), core_v1

    def _make_pod_and_job(self, ctrl):
        _register(ctrl, "CSE101")
        job = _submit(ctrl, "CSE101", _small(), job_id="uid-dry", username="s1")
        pod = {
            "metadata": {
                "name": "dry-pod", "namespace": "ns",
                "uid": "uid-dry", "labels": {"dsmlp/course": "CSE101"},
            },
            "spec": {"tolerations": [], "containers": [],
                     "schedulingGates": [{"name": "lane-scheduler"}]},
            "status": {},
        }
        return pod, job

    def test_patch_not_called_in_dry_run(self):
        ctrl, core_v1 = self._make_dry_controller()
        pod, job = self._make_pod_and_job(ctrl)
        ctrl._admit_pod(pod, job)
        core_v1.patch_namespaced_pod.assert_not_called()

    def test_pod_still_marked_admitted_in_dry_run(self):
        ctrl, _ = self._make_dry_controller()
        pod, job = self._make_pod_and_job(ctrl)
        ctrl._admit_pod(pod, job)
        with ctrl._admitted_lock:
            self.assertIn("uid-dry", ctrl._admitted)

    def test_patch_called_when_not_dry_run(self):
        from lane_scheduler.k8s.controller import LaneSchedulerController
        from lane_scheduler.estimation.wait_estimator import ResidencyProfile
        core_v1 = MagicMock()
        registry = SchedGroupRegistry()
        ctrl = LaneSchedulerController(
            core_v1=core_v1,
            registry=registry,
            sched_config=SchedulerConfig(),
            residency_profiles={
                "interactive": ResidencyProfile(mean_pct=0.4, std_pct=0.2),
                "batch":       ResidencyProfile(mean_pct=0.7, std_pct=0.15),
            },
            dry_run=False,
            web_port=0,
        )
        _register(ctrl, "CSE101")
        job = _submit(ctrl, "CSE101", _small(), job_id="uid-live", username="s1")
        pod = {
            "metadata": {
                "name": "live-pod", "namespace": "ns",
                "uid": "uid-live", "labels": {"dsmlp/course": "CSE101"},
            },
            "spec": {"tolerations": [], "containers": [],
                     "schedulingGates": [{"name": "lane-scheduler"}]},
            "status": {},
        }
        ctrl._admit_pod(pod, job)
        core_v1.patch_namespaced_pod.assert_called_once()


class TestAdmitBackoff(unittest.TestCase):

    def _make_ctrl_with_failing_patch(self):
        from lane_scheduler.k8s.controller import LaneSchedulerController
        from lane_scheduler.estimation.wait_estimator import ResidencyProfile
        from kubernetes import client as k8s_client

        core_v1 = MagicMock()
        core_v1.patch_namespaced_pod.side_effect = (
            k8s_client.exceptions.ApiException(status=500, reason="Internal")
        )
        registry = SchedGroupRegistry()
        ctrl = LaneSchedulerController(
            core_v1=core_v1,
            registry=registry,
            sched_config=SchedulerConfig(),
            residency_profiles={
                "interactive": ResidencyProfile(mean_pct=0.4, std_pct=0.2),
                "batch":       ResidencyProfile(mean_pct=0.7, std_pct=0.15),
            },
            dry_run=False,
            web_port=0,
        )
        return ctrl, core_v1

    def _make_pod_job(self, ctrl, uid="uid-retry"):
        _register(ctrl, "CSE101")
        job = _submit(ctrl, "CSE101", _small(), job_id=uid, username="s1")
        pod = {
            "metadata": {
                "name": "retry-pod", "namespace": "ns",
                "uid": uid, "labels": {"dsmlp/course": "CSE101"},
            },
            "spec": {"tolerations": [], "containers": [],
                     "schedulingGates": [{"name": "lane-scheduler"}]},
            "status": {},
        }
        return pod, job

    def test_first_failure_records_backoff(self):
        ctrl, _ = self._make_ctrl_with_failing_patch()
        pod, job = self._make_pod_job(ctrl)
        ctrl._admit_pod(pod, job)
        with ctrl._admit_attempts_lock:
            self.assertIn("uid-retry", ctrl._admit_attempts)
            attempts, next_retry = ctrl._admit_attempts["uid-retry"]
        self.assertEqual(attempts, 1)
        self.assertGreater(next_retry, time.monotonic())
        with ctrl._pending_lock:
            self.assertIn("uid-retry", ctrl._pending)

    def test_gives_up_after_max_attempts(self):
        from lane_scheduler.k8s.controller import _MAX_ADMIT_ATTEMPTS
        ctrl, _ = self._make_ctrl_with_failing_patch()
        pod, job = self._make_pod_job(ctrl)
        for _ in range(_MAX_ADMIT_ATTEMPTS):
            ctrl._admit_pod(pod, job)
        with ctrl._admit_attempts_lock:
            self.assertNotIn("uid-retry", ctrl._admit_attempts)
        with ctrl._pending_lock:
            self.assertNotIn("uid-retry", ctrl._pending)
        self.assertIsNone(ctrl.scheduler.remove_job("uid-retry"))

    def test_404_clears_attempts(self):
        from kubernetes import client as k8s_client
        ctrl, core_v1 = self._make_ctrl_with_failing_patch()
        pod, job = self._make_pod_job(ctrl)
        ctrl._admit_pod(pod, job)
        core_v1.patch_namespaced_pod.side_effect = (
            k8s_client.exceptions.ApiException(status=404, reason="Not Found")
        )
        ctrl._admit_pod(pod, job)
        with ctrl._admit_attempts_lock:
            self.assertNotIn("uid-retry", ctrl._admit_attempts)


if __name__ == "__main__":
    unittest.main()
