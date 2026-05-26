"""
Integration-layer unit tests.
Tests sched_group_registry, pod_translator, and node_capacity
without requiring a live Kubernetes cluster.
"""

import unittest

from lane_scheduler.core.scheduler import (
    Lane, SchedGroup, Job, PriorityScorer, SchedulerConfig,
    initialise_lanes, lane_for_gpu_class,
)
from lane_scheduler.core.sched_group_registry import SchedGroupRegistry
from lane_scheduler.k8s.pod_translator import (
    admission_patch, needs_scheduling, pod_to_job, NO_SCHED_GROUP_LABEL,
)
from lane_scheduler.core.node_capacity import (
    NodeCapacityTracker, GPU_CLASS_LABEL_KEY,
    INHIBIT_TAINT_KEY, INHIBIT_TAINT_VALUE,
)
from lane_scheduler.k8s.pod_translator import SCHEDULING_GATE_NAME


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

def setUpModule():
    initialise_lanes(["xsmall", "small", "medium", "large", "xlarge"])


def _gpu(cls):
    return lane_for_gpu_class(cls)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pod(
    name="test-pod", namespace="jsmith", uid="uid-001",
    phase="Pending", node_name=None,
    labels=None, tolerations=None, containers=None,
    scheduling_gates=None,
):
    if containers is None:
        containers = [{"name": "c", "resources": {"requests": {"cpu": "2"}}}]
    if scheduling_gates is None and phase == "Pending" and node_name is None:
        scheduling_gates = [{"name": SCHEDULING_GATE_NAME}]
    return {
        "metadata": {"name": name, "namespace": namespace, "uid": uid,
                     "labels": labels or {}},
        "spec":     {"nodeName": node_name, "tolerations": tolerations or [],
                     "containers": containers,
                     "schedulingGates": scheduling_gates or []},
        "status":   {"phase": phase},
    }


def _gpu_class_taint(gpu_class):
    return {"key": GPU_CLASS_LABEL_KEY, "value": gpu_class, "effect": "NoSchedule"}


def _make_node(
    name="node-1", ready=True, unschedulable=False,
    cpu="32", gpu=None, gpu_class=None,
    has_inhibit_taint=True,
):
    taints = []
    labels = {}
    if gpu_class:
        labels[GPU_CLASS_LABEL_KEY] = gpu_class
        taints.append(_gpu_class_taint(gpu_class))
    allocatable = {"cpu": cpu}
    if gpu:
        allocatable["nvidia.com/gpu"] = gpu
    return {
        "metadata": {"name": name, "labels": labels},
        "spec":     {"unschedulable": unschedulable, "taints": taints},
        "status":   {"allocatable": allocatable,
                     "conditions": [{"type": "Ready",
                                     "status": "True" if ready else "False"}]},
    }


def _gpu_pod(gpu_class, batch=False, gpu_count="1",
             user=None, namespace="jsmith"):
    labels = {"dsmlp/course": "CSE234_SP26_A00", "gpu-class": gpu_class}
    if batch:
        labels["dsmlp/batch"] = "true"
    if user is not None:
        labels["dsmlp/user"] = user
    return _make_pod(
        namespace=namespace,
        labels=labels,
        containers=[{"name": "c", "resources": {"requests": {
            "cpu": "4", "nvidia.com/gpu": gpu_count,
        }}}],
    )


# ---------------------------------------------------------------------------
# SchedGroupRegistry
# ---------------------------------------------------------------------------

class TestSchedGroupRegistry(unittest.TestCase):

    def test_always_returns_weight_one(self):
        reg = SchedGroupRegistry()
        g = reg.get("CSE234_SP26_A00")
        self.assertAlmostEqual(g.weight, 1.0)

    def test_returns_scoped_sched_group(self):
        reg = SchedGroupRegistry()
        g = reg.get("CSE101_SP26_A00")
        self.assertIsInstance(g, SchedGroup)
        self.assertEqual(g.sched_group_id, "CSE101_SP26_A00")

    def test_unknown_id_returns_weight_one(self):
        reg = SchedGroupRegistry()
        g = reg.get("NONEXISTENT_GROUP")
        self.assertAlmostEqual(g.weight, 1.0)

    def test_different_ids_each_get_weight_one(self):
        reg = SchedGroupRegistry()
        ids = ["CSE101", "CSE234", "GRAD-501", "__unlabelled__"]
        for gid in ids:
            g = reg.get(gid)
            self.assertAlmostEqual(g.weight, 1.0, msg=f"failed for {gid!r}")


# ---------------------------------------------------------------------------
# needs_scheduling
# ---------------------------------------------------------------------------

class TestNeedsScheduling(unittest.TestCase):

    def test_pending_with_gate_needs_scheduling(self):
        pod = _make_pod(phase="Pending",
                        scheduling_gates=[{"name": SCHEDULING_GATE_NAME}])
        self.assertTrue(needs_scheduling(pod))

    def test_pending_without_gate_does_not(self):
        pod = _make_pod(phase="Pending", scheduling_gates=[])
        self.assertFalse(needs_scheduling(pod))

    def test_running_does_not(self):
        self.assertFalse(needs_scheduling(_make_pod(phase="Running")))

    def test_node_assigned_does_not(self):
        self.assertFalse(needs_scheduling(_make_pod(phase="Pending", node_name="n1")))

    def test_other_gate_only_does_not(self):
        pod = _make_pod(phase="Pending",
                        scheduling_gates=[{"name": "other-gate"}])
        self.assertFalse(needs_scheduling(pod))


# ---------------------------------------------------------------------------
# pod_to_job
# ---------------------------------------------------------------------------

class TestPodToJob(unittest.TestCase):

    def test_each_gpu_class_maps_to_distinct_lane(self):
        from lane_scheduler.core.scheduler import Lane
        lanes = {cls: pod_to_job(_gpu_pod(cls)).lane
                 for cls in ("xsmall", "small", "medium", "large", "xlarge")}
        self.assertEqual(len(set(lanes.values())), 5)
        for cls, lane in lanes.items():
            self.assertIn(lane, Lane)

    def test_gpu_class_matches_lane_for_gpu_class(self):
        for cls in ("xsmall", "small", "medium", "large", "xlarge"):
            self.assertEqual(pod_to_job(_gpu_pod(cls)).lane, _gpu(cls))

    def test_batch_flag_set(self):
        self.assertTrue(pod_to_job(_gpu_pod("large", batch=True)).batch)

    def test_batch_flag_unset_for_interactive(self):
        self.assertFalse(pod_to_job(_gpu_pod("large", batch=False)).batch)

    def test_gpu_count_is_resource_units(self):
        self.assertAlmostEqual(pod_to_job(_gpu_pod("large", gpu_count="3")).resource_units, 3.0)

    def test_no_course_label(self):
        pod = _gpu_pod("small")
        del pod["metadata"]["labels"]["dsmlp/course"]
        self.assertEqual(pod_to_job(pod).sched_group_id, NO_SCHED_GROUP_LABEL)

    def test_batch_label_case_insensitive(self):
        for val in ("true", "True", "TRUE"):
            pod = _gpu_pod("small")
            pod["metadata"]["labels"]["dsmlp/batch"] = val
            self.assertTrue(pod_to_job(pod).batch)

    def test_username_from_dsmlp_user_label(self):
        pod = _gpu_pod("small", user="alice")
        self.assertEqual(pod_to_job(pod).username, "alice")

    def test_username_falls_back_to_namespace(self):
        """When dsmlp/user label is absent, username falls back to namespace."""
        pod = _gpu_pod("small", namespace="jsmith-ns")
        self.assertEqual(pod_to_job(pod).username, "jsmith-ns")

    def test_username_label_overrides_namespace(self):
        pod = _gpu_pod("small", user="rgarcia", namespace="some-other-ns")
        self.assertEqual(pod_to_job(pod).username, "rgarcia")

    def test_unrecognised_gpu_class_raises(self):
        with self.assertRaises(ValueError):
            pod_to_job(_gpu_pod("supergpu"))


# ---------------------------------------------------------------------------
# admission_patch
# ---------------------------------------------------------------------------

class TestAdmissionPatch(unittest.TestCase):

    def _gated_gpu_pod(self, gpu_class):
        pod = _gpu_pod(gpu_class)
        pod["spec"]["schedulingGates"] = [{"name": SCHEDULING_GATE_NAME}]
        return pod

    def test_adds_node_selector(self):
        patch = admission_patch(self._gated_gpu_pod("medium"))
        self.assertEqual(
            patch["spec"]["nodeSelector"].get(GPU_CLASS_LABEL_KEY), "medium"
        )

    def test_preserves_existing_node_selector_entries(self):
        pod = self._gated_gpu_pod("small")
        pod["spec"]["nodeSelector"] = {"other-key": "other-val"}
        patch = admission_patch(pod)
        ns = patch["spec"]["nodeSelector"]
        self.assertEqual(ns.get("other-key"), "other-val")
        self.assertEqual(ns.get(GPU_CLASS_LABEL_KEY), "small")

    def test_adds_gpu_class_toleration(self):
        patch = admission_patch(self._gated_gpu_pod("medium"))
        tols = patch["spec"]["tolerations"]
        self.assertTrue(
            any(t.get("key") == GPU_CLASS_LABEL_KEY and t.get("value") == "medium"
                for t in tols),
        )

    def test_removes_scheduling_gate(self):
        patch = admission_patch(self._gated_gpu_pod("medium"))
        gate_patch = patch["spec"]["schedulingGates"]
        self.assertTrue(
            any(g.get("$patch") == "delete" and g.get("name") == SCHEDULING_GATE_NAME
                for g in gate_patch),
        )

    def test_preserves_existing_tolerations(self):
        pod = self._gated_gpu_pod("small")
        pod["spec"]["tolerations"] = [{"key": "other", "operator": "Exists"}]
        patch = admission_patch(pod)
        self.assertEqual(len(patch["spec"]["tolerations"]), 2)

    def test_idempotent_when_gate_absent(self):
        pod = _gpu_pod("medium")
        pod["spec"]["schedulingGates"] = []
        self.assertEqual(admission_patch(pod), {})


# ---------------------------------------------------------------------------
# NodeCapacityTracker
# ---------------------------------------------------------------------------

class TestNodeCapacityTracker(unittest.TestCase):

    def test_node_without_gpu_class_label_ignored(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(cpu="64"))
        for v in t.lane_capacity().values():
            self.assertAlmostEqual(v, 0.0)

    def test_each_gpu_class_tracked_independently(self):
        t = NodeCapacityTracker()
        for cls in ("xsmall", "small", "medium", "large", "xlarge"):
            t.upsert(_make_node(name=f"n-{cls}", gpu="4", gpu_class=cls))
        caps = t.lane_capacity()
        for cls in ("xsmall", "small", "medium", "large", "xlarge"):
            self.assertAlmostEqual(caps[_gpu(cls)], 4.0, msg=f"gpu-class={cls}")

    def test_same_class_aggregates(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(name="s1", gpu="4", gpu_class="small"))
        t.upsert(_make_node(name="s2", gpu="4", gpu_class="small"))
        self.assertAlmostEqual(t.lane_capacity()[_gpu("small")], 8.0)

    def test_unready_gpu_node_excluded(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(gpu="4", gpu_class="medium", ready=False))
        self.assertAlmostEqual(t.lane_capacity()[_gpu("medium")], 0.0)

    def test_node_identified_by_gpu_class_label(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(name="gn", gpu="4", gpu_class="medium",
                            has_inhibit_taint=False))
        self.assertAlmostEqual(t.lane_capacity()[_gpu("medium")], 4.0)

    def test_remove_gpu_node(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(name="gn", gpu="4", gpu_class="small"))
        t.remove("gn")
        self.assertAlmostEqual(t.lane_capacity()[_gpu("small")], 0.0)

    def test_gpu_node_capacity_is_gpu_count(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(name="gn", cpu="32", gpu="4", gpu_class="medium"))
        self.assertAlmostEqual(t.lane_capacity()[_gpu("medium")], 4.0)

    def test_node_with_unmanaged_gpu_class_excluded(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(name="unknown-gpu", cpu="32", gpu="4",
                            gpu_class="h200"))
        caps = t.lane_capacity()
        for cls in ("xsmall", "small", "medium", "large", "xlarge"):
            self.assertAlmostEqual(caps[_gpu(cls)], 0.0, msg=f"leaked into {cls}")

    def test_lane_for_node_known_node(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(name="gpu-node-1", gpu="4", gpu_class="large"))
        self.assertEqual(t.lane_for_node("gpu-node-1"), _gpu("large"))

    def test_lane_for_node_unknown_node(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(name="gpu-node-1", gpu="4", gpu_class="large"))
        self.assertIsNone(t.lane_for_node("not-a-node"))

    def test_lane_for_node_cpu_only_node_not_tracked(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(cpu="64"))
        self.assertIsNone(t.lane_for_node("default-node"))

    def test_lane_for_node_after_remove(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(name="gpu-node-1", gpu="4", gpu_class="medium"))
        t.remove("gpu-node-1")
        self.assertIsNone(t.lane_for_node("gpu-node-1"))


# ---------------------------------------------------------------------------
# Controller-level integration: pod DELETE clears the scheduler queue
# ---------------------------------------------------------------------------

class TestDeleteClearsSchedulerQueue(unittest.TestCase):

    def _build_controller(self):
        from lane_scheduler.k8s.controller import LaneSchedulerController
        from lane_scheduler.core.scheduler import SchedulerConfig
        from lane_scheduler.estimation.wait_estimator import ResidencyProfile

        class _StubCore:
            def create_namespaced_event(self, **kw): pass
            def patch_namespaced_pod(self, **kw):    pass

        registry = SchedGroupRegistry()
        return LaneSchedulerController(
            core_v1            = _StubCore(),
            registry           = registry,
            sched_config       = SchedulerConfig(),
            residency_profiles = {
                "interactive": ResidencyProfile(mean_pct=0.4, std_pct=0.2),
                "batch":       ResidencyProfile(mean_pct=0.7, std_pct=0.15),
            },
            dry_run            = True,
        )

    def test_delete_event_removes_job_from_scheduler(self):
        ctrl = self._build_controller()
        pod = _make_pod(
            uid="uid-DEL",
            labels={"dsmlp/course": "CSE101", "gpu-class": "medium"},
            phase="Pending",
        )
        ctrl._handle_pod_event({"type": "ADDED", "object": pod})
        self.assertGreater(
            sum(sum(cm.values())
                for cm in ctrl.scheduler.queue_depths().values()),
            0,
        )
        ctrl._handle_pod_event({"type": "DELETED", "object": pod})
        self.assertNotIn("uid-DEL", ctrl._pending)
        self.assertEqual(ctrl.scheduler.queue_depths(), {})


# ---------------------------------------------------------------------------
# Batch mode penalty (scoring)
# ---------------------------------------------------------------------------

class TestBatchModePenalty(unittest.TestCase):

    def _job(self, batch, submit_time=0.0):
        j = Job(job_id="J", sched_group_id="C", username="S",
                lane=_gpu("medium"), batch=batch)
        j.submit_time = submit_time
        return j

    def setUp(self):
        self.cfg    = SchedulerConfig()
        self.scorer = PriorityScorer(self.cfg)
        self.group  = SchedGroup(sched_group_id="C", weight=0.75)

    def test_interactive_beats_batch(self):
        si = self.scorer.score(self._job(False), self.group, 0.1, 100.0)
        sb = self.scorer.score(self._job(True),  self.group, 0.1, 100.0)
        self.assertGreater(si, sb)

    def test_batch_ages_slower(self):
        now = self.cfg.t_half_interactive * 2
        bi  = self.scorer.age_boost(self._job(False, submit_time=0.0), now)
        bb  = self.scorer.age_boost(self._job(True,  submit_time=0.0), now)
        self.assertGreater(bi, bb)

    def test_batch_eventually_drains(self):
        fresh_interactive = self._job(False, submit_time=1_000_000.0)
        old_batch         = self._job(True,  submit_time=0.0)
        now               = 1_000_000.0
        si = self.scorer.score(fresh_interactive, self.group, 0.1, now)
        sb = self.scorer.score(old_batch,         self.group, 0.1, now)
        self.assertGreater(sb, si)


if __name__ == "__main__":
    unittest.main(verbosity=2)
