"""
Integration-layer unit tests.
Tests course_registry, pod_translator, and node_capacity
without requiring a live Kubernetes cluster.
"""

import csv
import tempfile
import unittest
from pathlib import Path

from scheduler import (
    Lane, Tier, initialise_lanes, lane_for_gpu_class,
)
from course_registry import CourseRegistry, _infer_tier
from pod_translator import (
    admission_patch, needs_scheduling, pod_to_job,
    INHIBIT_TAINT_KEY, INHIBIT_TAINT_VALUE, NO_COURSE_LABEL,
)
from node_capacity import NodeCapacityTracker, GPU_CLASS_TAINT_KEY


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

def setUpModule():
    initialise_lanes(["xsmall", "small", "medium", "large", "xlarge"])


def _cpu():
    from scheduler import Lane as _L; return _L.CPU

def _gpu(cls):
    return lane_for_gpu_class(cls)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(rows):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    writer = csv.DictWriter(tmp, fieldnames=["course_id", "level", "seats"])
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


def _gpu_class_taint(gpu_class):
    return {"key": GPU_CLASS_TAINT_KEY, "value": gpu_class, "effect": "NoSchedule"}


def _make_node(
    name="node-1", ready=True, unschedulable=False,
    cpu="32", gpu=None, gpu_class=None, has_inhibit_taint=True,
):
    taints = []
    if has_inhibit_taint:
        taints.append(_inhibit_taint())
    if gpu_class:
        taints.append(_gpu_class_taint(gpu_class))
    allocatable = {"cpu": cpu}
    if gpu:
        allocatable["nvidia.com/gpu"] = gpu
    return {
        "metadata": {"name": name, "labels": {}},
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

class TestCourseRegistryInference(unittest.TestCase):

    def test_lower_division(self):
        self.assertEqual(_infer_tier("CSE8_SP26_A00"),    Tier.INTRO)
        self.assertEqual(_infer_tier("MATH20C_FA25_B01"), Tier.INTRO)

    def test_upper_division(self):
        self.assertEqual(_infer_tier("CSE101_SP26_A00"), Tier.UPPER_DIV)
        self.assertEqual(_infer_tier("CSE150_SP26_A00"), Tier.UPPER_DIV)

    def test_graduate(self):
        self.assertEqual(_infer_tier("CSE234_SP26_A00"), Tier.GRAD)

    def test_csv_load(self):
        path = _write_csv([
            {"course_id": "CSE101_SP26_A00", "level": "upper",    "seats": 55},
            {"course_id": "CSE234_SP26_A00", "level": "graduate", "seats": 18},
        ])
        reg = CourseRegistry()
        self.assertEqual(reg.load_csv(path), 2)
        c = reg.get("CSE101_SP26_A00")
        self.assertEqual(c.tier, Tier.UPPER_DIV)
        self.assertEqual(c.enrollment, 55)

    def test_fallback_on_unknown_course(self):
        reg = CourseRegistry()
        c   = reg.get("CSE234_SP26_A00")
        self.assertEqual(c.tier, Tier.GRAD)
        self.assertEqual(c.enrollment, 50)

    def test_fallback_cached(self):
        reg = CourseRegistry()
        self.assertIs(reg.get("CSE101_SP26_A00"), reg.get("CSE101_SP26_A00"))

    def test_missing_column_raises(self):
        tmp = _write_csv([{"course_id": "X", "level": "grad", "seats": 10}])
        tmp.write_text("course_id,level\nX,grad\n")
        with self.assertRaises(ValueError):
            CourseRegistry().load_csv(tmp)

    def test_bad_seat_count_skipped(self):
        path = _write_csv([
            {"course_id": "CSE101_SP26_A00", "level": "upper",    "seats": "bad"},
            {"course_id": "CSE234_SP26_A00", "level": "graduate", "seats": "18"},
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
        from scheduler import GPU_LANES
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

    def test_unrecognised_gpu_class_returns_fallback_lane(self):
        from scheduler import GPU_LANES
        job = pod_to_job(_gpu_pod("supergpu"))
        self.assertIn(job.lane, GPU_LANES)


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


# ---------------------------------------------------------------------------
# Batch mode penalty (scoring)
# ---------------------------------------------------------------------------

class TestBatchModePenalty(unittest.TestCase):

    def _job(self, batch, submit_time=0.0):
        from scheduler import Job
        j = Job(job_id="J", class_id="C", student_id="S",
                lane=_gpu("medium"), batch=batch)
        j.submit_time = submit_time
        return j

    def setUp(self):
        from scheduler import PriorityScorer, SchedulerConfig
        self.cfg    = SchedulerConfig()
        self.scorer = PriorityScorer(self.cfg)
        from scheduler import CourseClass
        self.course = CourseClass("C", Tier.GRAD, 16)

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
