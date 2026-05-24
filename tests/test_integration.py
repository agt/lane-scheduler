"""
Integration-layer unit tests.
Tests course_registry, pod_translator, and node_capacity
without requiring a live Kubernetes cluster.
"""

import csv
import tempfile
import unittest
from pathlib import Path

from lane_scheduler.core.scheduler import (
    Lane, initialise_lanes, lane_for_gpu_class,
)
from lane_scheduler.core.course_registry import CourseRegistry
from lane_scheduler.k8s.pod_translator import (
    admission_patch, needs_scheduling, pod_to_job, NO_COURSE_LABEL,
)
from lane_scheduler.core.node_capacity import (
    NodeCapacityTracker, GPU_CLASS_LABEL_KEY,
    INHIBIT_TAINT_KEY, INHIBIT_TAINT_VALUE,
)


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

def setUpModule():
    initialise_lanes(["xsmall", "small", "medium", "large", "xlarge"])


def _cpu():
    from lane_scheduler.core.scheduler import CPU_LANE; return CPU_LANE

def _gpu(cls):
    return lane_for_gpu_class(cls)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(rows):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    writer = csv.DictWriter(tmp, fieldnames=["course_id", "weight"])
    writer.writeheader()
    writer.writerows(rows)
    tmp.close()
    return Path(tmp.name)


def _make_pod(
    name="test-pod", namespace="jsmith", uid="uid-001",
    phase="Pending", node_name=None,
    labels=None, tolerations=None, containers=None,
):
    if containers is None:
        containers = [{"name": "c", "resources": {"requests": {"cpu": "2"}}}]
    return {
        "metadata": {"name": name, "namespace": namespace, "uid": uid,
                     "labels": labels or {}},
        "spec":     {"nodeName": node_name, "tolerations": tolerations or [],
                     "containers": containers},
        "status":   {"phase": phase},
    }


def _inhibit_taint():
    return {"key": INHIBIT_TAINT_KEY, "value": INHIBIT_TAINT_VALUE,
            "effect": "NoSchedule"}


def _make_node(
    name="node-1", ready=True, unschedulable=False,
    cpu="32", gpu=None, gpu_class=None, has_inhibit_taint=True,
):
    taints = []
    if has_inhibit_taint:
        taints.append(_inhibit_taint())
    labels = {}
    if gpu_class:
        labels[GPU_CLASS_LABEL_KEY] = gpu_class
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


def _gpu_pod(gpu_class, batch=False, gpu_count="1"):
    labels = {"dsmlp/course": "CSE234_SP26_A00", "gpu-class": gpu_class}
    if batch:
        labels["dsmlp/batch"] = "true"
    return _make_pod(
        labels=labels,
        containers=[{"name": "c", "resources": {"requests": {
            "cpu": "4", "nvidia.com/gpu": gpu_count,
        }}}],
    )


# ---------------------------------------------------------------------------
# CourseRegistry
# ---------------------------------------------------------------------------

class TestCourseRegistry(unittest.TestCase):

    def test_csv_load(self):
        path = _write_csv([
            {"course_id": "CSE101_SP26_A00", "weight": 0.270},
            {"course_id": "CSE234_SP26_A00", "weight": 0.775},
        ])
        reg = CourseRegistry()
        self.assertEqual(reg.load_csv(path), 2)
        c = reg.get("CSE101_SP26_A00")
        self.assertAlmostEqual(c.class_weight, 0.270)

    def test_weight_values(self):
        path = _write_csv([
            {"course_id": "A", "weight": 0.07},
            {"course_id": "B", "weight": 0.27},
            {"course_id": "C", "weight": 0.77},
        ])
        reg = CourseRegistry()
        reg.load_csv(path)
        self.assertAlmostEqual(reg.get("A").class_weight, 0.07)
        self.assertAlmostEqual(reg.get("B").class_weight, 0.27)
        self.assertAlmostEqual(reg.get("C").class_weight, 0.77)

    def test_fallback_on_unknown_course(self):
        reg = CourseRegistry()
        c   = reg.get("UNKNOWN_SP26_A00")
        self.assertAlmostEqual(c.class_weight, 1.0)

    def test_fallback_cached(self):
        reg = CourseRegistry()
        self.assertIs(reg.get("CSE101_SP26_A00"), reg.get("CSE101_SP26_A00"))

    def test_missing_column_raises(self):
        tmp = _write_csv([{"course_id": "X", "weight": 1.0}])
        tmp.write_text("course_id\nX\n")
        with self.assertRaises(ValueError):
            CourseRegistry().load_csv(tmp)

    def test_bad_weight_skipped(self):
        path = _write_csv([
            {"course_id": "CSE101_SP26_A00", "weight": "bad"},
            {"course_id": "CSE234_SP26_A00", "weight": 0.77},
        ])
        self.assertEqual(CourseRegistry().load_csv(path), 1)

    def test_nonpositive_weight_skipped(self):
        path = _write_csv([
            {"course_id": "CSE101_SP26_A00", "weight": -1.0},
            {"course_id": "CSE234_SP26_A00", "weight": 0.77},
        ])
        self.assertEqual(CourseRegistry().load_csv(path), 1)


# ---------------------------------------------------------------------------
# needs_scheduling
# ---------------------------------------------------------------------------

class TestNeedsScheduling(unittest.TestCase):

    def test_pending_needs_scheduling(self):
        self.assertTrue(needs_scheduling(_make_pod(phase="Pending")))

    def test_running_does_not(self):
        self.assertFalse(needs_scheduling(_make_pod(phase="Running")))

    def test_already_admitted_does_not(self):
        pod = _make_pod(phase="Pending", tolerations=[{
            "key": INHIBIT_TAINT_KEY, "value": INHIBIT_TAINT_VALUE, "effect": "NoSchedule",
        }])
        self.assertFalse(needs_scheduling(pod))

    def test_node_assigned_does_not(self):
        self.assertFalse(needs_scheduling(_make_pod(phase="Pending", node_name="n1")))


# ---------------------------------------------------------------------------
# pod_to_job
# ---------------------------------------------------------------------------

class TestPodToJob(unittest.TestCase):

    def test_cpu_pod(self):
        pod = _make_pod(
            labels={"dsmlp/course": "CSE101_SP26_A00"},
            containers=[{"name": "c", "resources": {"requests": {"cpu": "4"}}}],
        )
        job = pod_to_job(pod, submit_time=0.0)
        self.assertEqual(job.lane, _cpu())
        self.assertAlmostEqual(job.resource_units, 4.0)
        self.assertFalse(job.batch)

    def test_each_gpu_class_maps_to_distinct_lane(self):
        from lane_scheduler.core.scheduler import GPU_LANES
        lanes = {cls: pod_to_job(_gpu_pod(cls)).lane
                 for cls in ("xsmall", "small", "medium", "large", "xlarge")}
        self.assertEqual(len(set(lanes.values())), 5)
        for cls, lane in lanes.items():
            self.assertIn(lane, GPU_LANES)

    def test_gpu_class_matches_lane_for_gpu_class(self):
        for cls in ("xsmall", "small", "medium", "large", "xlarge"):
            self.assertEqual(pod_to_job(_gpu_pod(cls)).lane, _gpu(cls))

    def test_batch_flag_set(self):
        self.assertTrue(pod_to_job(_gpu_pod("large", batch=True)).batch)

    def test_batch_flag_unset_for_interactive(self):
        self.assertFalse(pod_to_job(_gpu_pod("large", batch=False)).batch)

    def test_batch_on_cpu_pod(self):
        pod = _make_pod(labels={"dsmlp/batch": "true"})
        job = pod_to_job(pod)
        self.assertEqual(job.lane, _cpu())
        self.assertTrue(job.batch)

    def test_gpu_count_is_resource_units(self):
        self.assertAlmostEqual(pod_to_job(_gpu_pod("large", gpu_count="3")).resource_units, 3.0)

    def test_no_course_label(self):
        self.assertEqual(pod_to_job(_make_pod(labels={})).class_id, NO_COURSE_LABEL)

    def test_millicore_cpu_floor(self):
        pod = _make_pod(
            containers=[{"name": "c", "resources": {"requests": {"cpu": "500m"}}}]
        )
        self.assertAlmostEqual(pod_to_job(pod).resource_units, 1.0)

    def test_batch_label_case_insensitive(self):
        for val in ("true", "True", "TRUE"):
            pod = _gpu_pod("small")
            pod["metadata"]["labels"]["dsmlp/batch"] = val
            self.assertTrue(pod_to_job(pod).batch)

    def test_student_id_is_namespace(self):
        self.assertEqual(pod_to_job(_make_pod(namespace="alice")).student_id, "alice")

    def test_unrecognised_gpu_class_falls_back_to_cpu(self):
        job = pod_to_job(_gpu_pod("supergpu"))
        self.assertEqual(job.lane, _cpu())


# ---------------------------------------------------------------------------
# admission_patch
# ---------------------------------------------------------------------------

class TestAdmissionPatch(unittest.TestCase):

    def test_adds_toleration(self):
        patch = admission_patch(_make_pod())
        keys  = [t["key"] for t in patch["spec"]["tolerations"]]
        self.assertIn(INHIBIT_TAINT_KEY, keys)

    def test_preserves_existing_tolerations(self):
        pod   = _make_pod(tolerations=[{"key": "other", "operator": "Exists"}])
        patch = admission_patch(pod)
        self.assertEqual(len(patch["spec"]["tolerations"]), 2)

    def test_idempotent(self):
        pod = _make_pod(tolerations=[{
            "key": INHIBIT_TAINT_KEY, "value": INHIBIT_TAINT_VALUE, "effect": "NoSchedule",
        }])
        self.assertEqual(admission_patch(pod), {})


# ---------------------------------------------------------------------------
# NodeCapacityTracker
# ---------------------------------------------------------------------------

class TestNodeCapacityTracker(unittest.TestCase):

    def test_cpu_node(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(cpu="64"))
        self.assertAlmostEqual(t.lane_capacity()[_cpu()], 64.0)

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

    def test_unready_excluded(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(cpu="64", ready=False))
        self.assertAlmostEqual(t.lane_capacity()[_cpu()], 0.0)

    def test_no_inhibit_taint_excluded(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(cpu="64", has_inhibit_taint=False))
        self.assertAlmostEqual(t.lane_capacity()[_cpu()], 0.0)

    def test_remove_node(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(name="cpu-1", cpu="32"))
        t.remove("cpu-1")
        self.assertAlmostEqual(t.lane_capacity()[_cpu()], 0.0)

    def test_gpu_node_not_counted_in_cpu_lane(self):
        t = NodeCapacityTracker()
        t.upsert(_make_node(name="gn", cpu="32", gpu="4", gpu_class="medium"))
        self.assertAlmostEqual(t.lane_capacity()[_cpu()], 0.0)
        self.assertAlmostEqual(t.lane_capacity()[_gpu("medium")], 4.0)

    def test_node_with_unmanaged_gpu_class_excluded(self):
        """A node labelled with a gpu-class the controller doesn't manage
        must be excluded from the pool entirely — not misclassified as CPU."""
        t = NodeCapacityTracker()
        t.upsert(_make_node(name="unknown-gpu", cpu="32", gpu="4",
                            gpu_class="h200"))
        caps = t.lane_capacity()
        self.assertAlmostEqual(caps[_cpu()], 0.0)
        # No GPU lane should pick up its GPUs either.
        for cls in ("xsmall", "small", "medium", "large", "xlarge"):
            self.assertAlmostEqual(caps[_gpu(cls)], 0.0, msg=f"leaked into {cls}")


# ---------------------------------------------------------------------------
# Controller-level integration: pod DELETE clears the scheduler queue
# ---------------------------------------------------------------------------

class TestDeleteClearsSchedulerQueue(unittest.TestCase):
    """A DELETED pod event must clear the Job from scheduler._queues too,
    not just from controller _pending."""

    def _build_controller(self):
        # Build a minimal controller without K8s API.  We exercise only
        # _enqueue / _handle_pod_event / _dequeue paths.
        from lane_scheduler.k8s.controller import LaneSchedulerController
        from lane_scheduler.core.scheduler import SchedulerConfig
        from lane_scheduler.estimation.wait_estimator import ResidencyProfile
        from lane_scheduler.core.course_registry import CourseRegistry

        # Stub core_v1 — never called by these tests
        class _StubCore:
            def create_namespaced_event(self, **kw): pass
            def patch_namespaced_pod(self, **kw):    pass

        registry = CourseRegistry()
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
            uid="uid-DEL", labels={"dsmlp/course": "CSE101"},
            phase="Pending",
        )
        ctrl._handle_pod_event({"type": "ADDED", "object": pod})
        # Pod is enqueued in scheduler
        self.assertGreater(
            sum(sum(cm.values())
                for cm in ctrl.scheduler.queue_depths().values()),
            0,
        )
        # DELETED should clear both controller _pending AND scheduler queue
        ctrl._handle_pod_event({"type": "DELETED", "object": pod})
        self.assertNotIn("uid-DEL", ctrl._pending)
        self.assertEqual(ctrl.scheduler.queue_depths(), {})


# ---------------------------------------------------------------------------
# Batch mode penalty (scoring)
# ---------------------------------------------------------------------------

class TestBatchModePenalty(unittest.TestCase):

    def _job(self, batch, submit_time=0.0):
        from lane_scheduler.core.scheduler import Job
        j = Job(job_id="J", class_id="C", student_id="S",
                lane=_gpu("medium"), batch=batch)
        j.submit_time = submit_time
        return j

    def setUp(self):
        from lane_scheduler.core.scheduler import PriorityScorer, SchedulerConfig
        self.cfg    = SchedulerConfig()
        self.scorer = PriorityScorer(self.cfg)
        from lane_scheduler.core.scheduler import CourseClass
        self.course = CourseClass("C", class_weight=0.75)

    def test_interactive_beats_batch(self):
        si = self.scorer.score(self._job(False), self.course, 0.1, 100.0)
        sb = self.scorer.score(self._job(True),  self.course, 0.1, 100.0)
        self.assertGreater(si, sb)

    def test_batch_ages_slower(self):
        now = self.cfg.t_half_interactive * 2
        bi  = self.scorer.age_boost(self._job(False, submit_time=0.0), now)
        bb  = self.scorer.age_boost(self._job(True,  submit_time=0.0), now)
        self.assertGreater(bi, bb)

    def test_batch_eventually_drains(self):
        """Long-waiting batch job should outscore a brand-new interactive job."""
        fresh_interactive = self._job(False, submit_time=1_000_000.0)
        old_batch         = self._job(True,  submit_time=0.0)
        now               = 1_000_000.0
        si = self.scorer.score(fresh_interactive, self.course, 0.1, now)
        sb = self.scorer.score(old_batch,         self.course, 0.1, now)
        self.assertGreater(sb, si)


if __name__ == "__main__":
    unittest.main(verbosity=2)
