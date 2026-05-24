"""
Tests for the capacity gate in LaneSchedulerController._run_cycle().

The gate prevents admitting a pod when:
    lane_capacity - running_units - admitted_but_not_running_units < job.resource_units

Uses a minimal controller stub backed by real Scheduler / NodeCapacityTracker
instances and a MagicMock Kubernetes client.
"""

import csv
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from lane_scheduler.core.scheduler import (
    Lane, SchedulerConfig,
    initialise_lanes, lane_for_gpu_class,
)
from lane_scheduler.core.node_capacity import NodeCapacityTracker, GPU_CLASS_LABEL_KEY
from lane_scheduler.core.course_registry import CourseRegistry
from lane_scheduler.estimation.wait_estimator import WaitTimeCache, RunningPod, ResidencyProfile
from lane_scheduler.k8s.controller import LaneSchedulerController
from lane_scheduler.k8s.pod_translator import INHIBIT_TAINT_KEY, INHIBIT_TAINT_VALUE


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

def setUpModule():
    lane_enum = initialise_lanes(["xsmall", "small", "medium", "large", "xlarge"])
    # Several modules import Lane at module load time (before setUpModule runs).
    # Their bindings are stale (None). Update them so runtime code sees the enum.
    import lane_scheduler.k8s.controller as _ctrl
    import lane_scheduler.core.node_capacity as _nc
    import lane_scheduler.k8s.pod_translator as _pt
    _ctrl.Lane = lane_enum
    _nc.Lane   = lane_enum
    _pt.Lane   = lane_enum


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


def _make_pod(uid="uid-001", gpu_class="small", gpu_count="1",
              course="CSE234_SP26_A00", namespace="ns", name=None):
    if name is None:
        name = f"pod-{uid}"
    labels = {"dsmlp/course": course, GPU_CLASS_LABEL_KEY: gpu_class}
    return {
        "metadata": {"name": name, "namespace": namespace, "uid": uid,
                     "labels": labels},
        "spec": {
            "nodeName": None,
            "tolerations": [],
            "containers": [{"name": "c", "resources": {"requests": {
                "cpu": "2", "nvidia.com/gpu": gpu_count,
            }}}],
        },
        "status": {"phase": "Pending"},
    }


def _make_running_pod(uid, lane, resource_units=1.0):
    return RunningPod(
        pod_uid=uid,
        start_time=time.monotonic() - 60,
        active_deadline_seconds=3600,
        batch=False,
        resource_units=resource_units,
    )


def _make_node(name="node-1", gpu_class="small", gpu_count="4", ready=True):
    return {
        "metadata": {"name": name, "labels": {GPU_CLASS_LABEL_KEY: gpu_class}},
        "spec": {
            "unschedulable": False,
            "taints": [{"key": INHIBIT_TAINT_KEY, "value": INHIBIT_TAINT_VALUE,
                        "effect": "NoSchedule"},
                       {"key": GPU_CLASS_LABEL_KEY, "value": gpu_class,
                        "effect": "NoSchedule"}],
        },
        "status": {
            "allocatable": {"cpu": "32", "nvidia.com/gpu": gpu_count},
            "conditions": [{"type": "Ready",
                            "status": "True" if ready else "False"}],
        },
    }


def _build_controller(csv_path, gpu_class="small", gpu_count=4):
    """Return a LaneSchedulerController with mocked k8s clients."""
    core_v1 = MagicMock()
    registry = CourseRegistry()
    registry.load_csv(csv_path)
    residency_profiles = {
        "interactive": ResidencyProfile(mean_pct=0.4, std_pct=0.2),
        "batch":       ResidencyProfile(mean_pct=0.7, std_pct=0.15),
    }
    ctrl = LaneSchedulerController(
        core_v1=core_v1,
        registry=registry,
        sched_config=SchedulerConfig(),
        residency_profiles=residency_profiles,
        dry_run=False,
    )
    ctrl.node_tracker.upsert(_make_node(gpu_class=gpu_class, gpu_count=str(gpu_count)))
    # Sync lane capacity into the scheduler
    ctrl.scheduler.set_lane_capacity(ctrl.node_tracker.lane_capacity())
    return ctrl, core_v1


def _enqueue_pod(ctrl, pod):
    """Directly inject a pod into the controller's pending queue and scheduler."""
    from lane_scheduler.k8s.pod_translator import pod_to_job
    import time as _time
    course_id = (pod.get("metadata", {}).get("labels") or {}).get(
        "dsmlp/course", "UNKNOWN"
    )
    if not ctrl.scheduler.has_class(course_id):
        course = ctrl.registry.get(course_id)
        ctrl.scheduler.register_class(course)
    job = pod_to_job(pod, submit_time=_time.monotonic())
    uid = pod["metadata"]["uid"]
    with ctrl._pending_lock:
        ctrl._pending[uid] = pod
    ctrl.scheduler.submit(job)
    return job


def _queue_depth(sched, lane):
    """Total jobs waiting in a lane across all courses."""
    from lane_scheduler.core.scheduler import LANE_NAMES
    lane_name = LANE_NAMES.get(lane, str(lane))
    return sum(sched.queue_depths().get(lane_name, {}).values())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCapacityGate(unittest.TestCase):

    def setUp(self):
        self.csv_path = _write_csv([
            {"course_id": "CSE234_SP26_A00", "level": "grad",  "seats": "20"},
            {"course_id": "CSE190_SP26_A00", "level": "upper", "seats": "30"},
            {"course_id": "CSE100_SP26_A00", "level": "intro", "seats": "40"},
        ])

    def tearDown(self):
        self.csv_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Gate blocks when lane is full (running pods fill capacity)
    # ------------------------------------------------------------------

    def test_gate_blocks_when_running_fills_capacity(self):
        ctrl, core_v1 = _build_controller(self.csv_path, gpu_class="small", gpu_count=1)

        # One GPU in the lane, one already Running
        lane = _gpu("small")
        with ctrl._running_lock:
            ctrl._running.setdefault(lane, {})["running-uid"] = _make_running_pod(
                "running-uid", lane, resource_units=1.0
            )

        pod = _make_pod(uid="new-uid", gpu_class="small", gpu_count="1")
        _enqueue_pod(ctrl, pod)

        ctrl._run_cycle()

        core_v1.patch_namespaced_pod.assert_not_called()
        # Job must be re-submitted so the next cycle can retry
        self.assertEqual(_queue_depth(ctrl.scheduler, lane), 1)

    # ------------------------------------------------------------------
    # Gate blocks when admitted-but-not-running fills capacity
    # ------------------------------------------------------------------

    def test_gate_blocks_when_admitted_not_running_fills_capacity(self):
        ctrl, core_v1 = _build_controller(self.csv_path, gpu_class="small", gpu_count=2)

        lane = _gpu("small")
        # 2 GPUs in lane; 2 already admitted-but-not-running
        with ctrl._admitted_resources_lock:
            ctrl._admitted_resources["admitted-1"] = (lane, 1.0)
            ctrl._admitted_resources["admitted-2"] = (lane, 1.0)

        pod = _make_pod(uid="new-uid", gpu_class="small", gpu_count="1")
        _enqueue_pod(ctrl, pod)

        ctrl._run_cycle()

        core_v1.patch_namespaced_pod.assert_not_called()

    # ------------------------------------------------------------------
    # Gate allows admission when free capacity exists
    # ------------------------------------------------------------------

    def test_gate_allows_when_capacity_available(self):
        ctrl, core_v1 = _build_controller(self.csv_path, gpu_class="small", gpu_count=4)

        lane = _gpu("small")
        # 4 GPUs; 1 running, 1 admitted — 2 free; pod requests 1
        with ctrl._running_lock:
            ctrl._running.setdefault(lane, {})["r1"] = _make_running_pod(
                "r1", lane, resource_units=1.0
            )
        with ctrl._admitted_resources_lock:
            ctrl._admitted_resources["a1"] = (lane, 1.0)

        pod = _make_pod(uid="new-uid", gpu_class="small", gpu_count="1")
        _enqueue_pod(ctrl, pod)

        ctrl._run_cycle()

        core_v1.patch_namespaced_pod.assert_called_once()

    # ------------------------------------------------------------------
    # Admitted entry is recorded on successful patch
    # ------------------------------------------------------------------

    def test_admitted_resources_recorded_on_success(self):
        ctrl, core_v1 = _build_controller(self.csv_path, gpu_class="small", gpu_count=4)

        pod = _make_pod(uid="p1", gpu_class="small", gpu_count="2")
        _enqueue_pod(ctrl, pod)

        ctrl._run_cycle()

        core_v1.patch_namespaced_pod.assert_called_once()
        with ctrl._admitted_resources_lock:
            entry = ctrl._admitted_resources.get("p1")
        self.assertIsNotNone(entry)
        lane, units = entry
        self.assertEqual(lane, _gpu("small"))
        self.assertAlmostEqual(units, 2.0)

    # ------------------------------------------------------------------
    # Admitted entry removed when pod transitions to Running (_dequeue)
    # ------------------------------------------------------------------

    def test_admitted_resources_removed_on_running(self):
        ctrl, _ = _build_controller(self.csv_path, gpu_class="small", gpu_count=4)

        lane = _gpu("small")
        with ctrl._admitted_resources_lock:
            ctrl._admitted_resources["p1"] = (lane, 1.0)
        with ctrl._admitted_lock:
            ctrl._admitted.add("p1")

        # Simulate pod watch: MODIFIED event with phase=Running
        running_pod = {
            "metadata": {"uid": "p1", "name": "pod-p1", "namespace": "ns",
                         "labels": {"dsmlp/course": "CSE234_SP26_A00",
                                    GPU_CLASS_LABEL_KEY: "small"}},
            "spec": {"nodeName": "node-1", "tolerations": [],
                     "containers": [{"name": "c", "resources": {
                         "requests": {"cpu": "2", "nvidia.com/gpu": "1"}
                     }}],
                     "activeDeadlineSeconds": 3600},
            "status": {"phase": "Running",
                       "startTime": "2026-01-01T00:00:00Z"},
        }
        ctrl._handle_pod_event({"type": "MODIFIED", "object": running_pod})

        with ctrl._admitted_resources_lock:
            self.assertNotIn("p1", ctrl._admitted_resources)

    # ------------------------------------------------------------------
    # Admitted entry removed when pod is deleted
    # ------------------------------------------------------------------

    def test_admitted_resources_removed_on_deleted(self):
        ctrl, _ = _build_controller(self.csv_path, gpu_class="small", gpu_count=4)

        lane = _gpu("small")
        with ctrl._admitted_resources_lock:
            ctrl._admitted_resources["p1"] = (lane, 1.0)

        ctrl._dequeue("p1")

        with ctrl._admitted_resources_lock:
            self.assertNotIn("p1", ctrl._admitted_resources)

    # ------------------------------------------------------------------
    # Admitted entry released on patch failure (ApiException)
    # ------------------------------------------------------------------

    def test_admitted_resources_released_on_patch_failure(self):
        from kubernetes import client as k8s_client

        ctrl, core_v1 = _build_controller(self.csv_path, gpu_class="small", gpu_count=4)
        core_v1.patch_namespaced_pod.side_effect = k8s_client.exceptions.ApiException(
            status=500
        )

        pod = _make_pod(uid="p1", gpu_class="small", gpu_count="1")
        _enqueue_pod(ctrl, pod)

        ctrl._run_cycle()

        with ctrl._admitted_resources_lock:
            self.assertNotIn("p1", ctrl._admitted_resources)

    # ------------------------------------------------------------------
    # Admitted entry released on 404 (pod vanished)
    # ------------------------------------------------------------------

    def test_admitted_resources_released_on_404(self):
        from kubernetes import client as k8s_client

        ctrl, core_v1 = _build_controller(self.csv_path, gpu_class="small", gpu_count=4)
        core_v1.patch_namespaced_pod.side_effect = k8s_client.exceptions.ApiException(
            status=404
        )

        pod = _make_pod(uid="p1", gpu_class="small", gpu_count="1")
        _enqueue_pod(ctrl, pod)

        ctrl._run_cycle()

        with ctrl._admitted_resources_lock:
            self.assertNotIn("p1", ctrl._admitted_resources)

    # ------------------------------------------------------------------
    # Multiple jobs in one cycle — sequential capacity accounting
    # ------------------------------------------------------------------

    def test_sequential_accounting_within_cycle(self):
        """Two 1-GPU pods should both be admitted in a 2-GPU lane; third deferred.

        Uses three different students (namespaces) so the scheduler picks one
        candidate per student per class, yielding up to 3 candidates in one
        cycle and letting the capacity gate trim to 2.
        """
        ctrl, core_v1 = _build_controller(self.csv_path, gpu_class="small", gpu_count=2)

        courses = ["CSE234_SP26_A00", "CSE190_SP26_A00", "CSE100_SP26_A00"]
        for i, course in enumerate(courses):
            pod = _make_pod(uid=f"p{i}", gpu_class="small", gpu_count="1",
                            course=course, namespace=f"student{i}",
                            name=f"pod-p{i}")
            _enqueue_pod(ctrl, pod)

        ctrl._run_cycle()

        # Exactly 2 patches (capacity = 2)
        self.assertEqual(core_v1.patch_namespaced_pod.call_count, 2)

        # Third job re-queued
        lane = _gpu("small")
        self.assertEqual(_queue_depth(ctrl.scheduler, lane), 1)

    # ------------------------------------------------------------------
    # Deferred job is re-submitted to the scheduler
    # ------------------------------------------------------------------

    def test_deferred_job_resubmitted_to_scheduler(self):
        ctrl, core_v1 = _build_controller(self.csv_path, gpu_class="small", gpu_count=1)

        lane = _gpu("small")
        with ctrl._running_lock:
            ctrl._running.setdefault(lane, {})["r1"] = _make_running_pod(
                "r1", lane, resource_units=1.0
            )

        pod = _make_pod(uid="p1", gpu_class="small", gpu_count="1")
        _enqueue_pod(ctrl, pod)

        ctrl._run_cycle()

        core_v1.patch_namespaced_pod.assert_not_called()
        self.assertEqual(_queue_depth(ctrl.scheduler, lane), 1)


if __name__ == "__main__":
    unittest.main()
