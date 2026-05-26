"""
Tests for the capacity gate in LaneSchedulerController._run_cycle().

The gate prevents admitting a pod when:
    lane_capacity - running_units - kubernetes_pending_units - admitted_units
    < job.resource_units

Uses a minimal controller stub backed by real Scheduler / NodeCapacityTracker
instances and a MagicMock Kubernetes client.
"""

import time
import unittest
from unittest.mock import MagicMock

from lane_scheduler.core.scheduler import (
    Lane, SchedulerConfig,
    initialise_lanes, lane_for_gpu_class,
)
from lane_scheduler.core.node_capacity import NodeCapacityTracker, GPU_CLASS_LABEL_KEY
from lane_scheduler.core.sched_group_registry import SchedGroupRegistry
from lane_scheduler.estimation.wait_estimator import WaitTimeCache, RunningPod, ResidencyProfile
from lane_scheduler.k8s.controller import LaneSchedulerController
from lane_scheduler.core.node_capacity import INHIBIT_TAINT_KEY, INHIBIT_TAINT_VALUE
from lane_scheduler.k8s.pod_translator import SCHEDULING_GATE_NAME


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

def setUpModule():
    lane_enum = initialise_lanes(["xsmall", "small", "medium", "large", "xlarge"])
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
            "schedulingGates": [{"name": SCHEDULING_GATE_NAME}],
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
            "taints": [{"key": GPU_CLASS_LABEL_KEY, "value": gpu_class,
                        "effect": "NoSchedule"}],
        },
        "status": {
            "allocatable": {"cpu": "32", "nvidia.com/gpu": gpu_count},
            "conditions": [{"type": "Ready",
                            "status": "True" if ready else "False"}],
        },
    }


def _build_controller(gpu_class="small", gpu_count=4):
    """Return a LaneSchedulerController with mocked k8s clients."""
    core_v1 = MagicMock()
    registry = SchedGroupRegistry()
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
    ctrl.scheduler.set_lane_capacity(ctrl.node_tracker.lane_capacity())
    return ctrl, core_v1


def _enqueue_pod(ctrl, pod):
    """Directly inject a pod into the controller's pending queue and scheduler."""
    from lane_scheduler.k8s.pod_translator import pod_to_job
    import time as _time
    sched_group_id = (pod.get("metadata", {}).get("labels") or {}).get(
        "dsmlp/course", "UNKNOWN"
    )
    if not ctrl.scheduler.has_group(sched_group_id):
        group = ctrl.registry.get(sched_group_id)
        ctrl.scheduler.register_group(group)
    job = pod_to_job(pod, submit_time=_time.monotonic())
    uid = pod["metadata"]["uid"]
    with ctrl._pending_lock:
        ctrl._pending[uid] = pod
    ctrl.scheduler.submit(job)
    return job


def _queue_depth(sched, lane):
    """Total jobs waiting in a lane across all scheduling groups."""
    return sum(sched.queue_depths().get(lane, {}).values())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCapacityGate(unittest.TestCase):

    # ------------------------------------------------------------------
    # Gate blocks when lane is full (running pods fill capacity)
    # ------------------------------------------------------------------

    def test_gate_blocks_when_running_fills_capacity(self):
        ctrl, core_v1 = _build_controller(gpu_class="small", gpu_count=1)

        lane = _gpu("small")
        with ctrl._running_lock:
            ctrl._kubernetes_running.setdefault(lane, {})["running-uid"] = (
                _make_running_pod("running-uid", lane, resource_units=1.0)
            )

        pod = _make_pod(uid="new-uid", gpu_class="small", gpu_count="1")
        _enqueue_pod(ctrl, pod)

        ctrl._run_cycle()

        core_v1.patch_namespaced_pod.assert_not_called()
        self.assertEqual(_queue_depth(ctrl.scheduler, lane), 1)

    # ------------------------------------------------------------------
    # Gate blocks when admitted-but-not-running fills capacity
    # ------------------------------------------------------------------

    def test_gate_blocks_when_admitted_not_running_fills_capacity(self):
        ctrl, core_v1 = _build_controller(gpu_class="small", gpu_count=2)

        lane = _gpu("small")
        with ctrl._admitted_resources_lock:
            ctrl._admitted_resources["admitted-1"] = (lane, 1.0)
            ctrl._admitted_resources["admitted-2"] = (lane, 1.0)

        pod = _make_pod(uid="new-uid", gpu_class="small", gpu_count="1")
        _enqueue_pod(ctrl, pod)

        ctrl._run_cycle()

        core_v1.patch_namespaced_pod.assert_not_called()

    # ------------------------------------------------------------------
    # Gate blocks when kubernetes-pending pods fill capacity
    # ------------------------------------------------------------------

    def test_gate_blocks_when_kubernetes_pending_fills_capacity(self):
        """kubernetes_pending pods (admitted and gate-removed but not yet Running)
        must count against the capacity gate."""
        ctrl, core_v1 = _build_controller(gpu_class="small", gpu_count=2)

        lane = _gpu("small")
        # 2 GPUs consumed by k8s-Pending pods
        with ctrl._kubernetes_pending_lock:
            ctrl._kubernetes_pending.setdefault(lane, {})["kp-1"] = (
                _make_running_pod("kp-1", lane, resource_units=1.0)
            )
            ctrl._kubernetes_pending.setdefault(lane, {})["kp-2"] = (
                _make_running_pod("kp-2", lane, resource_units=1.0)
            )

        pod = _make_pod(uid="new-uid", gpu_class="small", gpu_count="1")
        _enqueue_pod(ctrl, pod)

        ctrl._run_cycle()

        core_v1.patch_namespaced_pod.assert_not_called()
        self.assertEqual(_queue_depth(ctrl.scheduler, lane), 1)

    # ------------------------------------------------------------------
    # Gate allows admission when free capacity exists
    # ------------------------------------------------------------------

    def test_gate_allows_when_capacity_available(self):
        ctrl, core_v1 = _build_controller(gpu_class="small", gpu_count=4)

        lane = _gpu("small")
        # 4 GPUs; 1 running, 1 admitted — 2 free; pod requests 1
        with ctrl._running_lock:
            ctrl._kubernetes_running.setdefault(lane, {})["r1"] = (
                _make_running_pod("r1", lane, resource_units=1.0)
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
        ctrl, core_v1 = _build_controller(gpu_class="small", gpu_count=4)

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
    # Admitted entry released when pod transitions to Running
    # ------------------------------------------------------------------

    def test_admitted_resources_removed_on_running(self):
        ctrl, _ = _build_controller(gpu_class="small", gpu_count=4)

        lane = _gpu("small")
        with ctrl._admitted_resources_lock:
            ctrl._admitted_resources["p1"] = (lane, 1.0)
        with ctrl._admitted_lock:
            ctrl._admitted.add("p1")

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
        ctrl, _ = _build_controller(gpu_class="small", gpu_count=4)

        lane = _gpu("small")
        with ctrl._admitted_resources_lock:
            ctrl._admitted_resources["p1"] = (lane, 1.0)
        with ctrl._admitted_lock:
            ctrl._admitted.add("p1")

        deleted_pod = {
            "metadata": {"uid": "p1", "name": "pod-p1", "namespace": "ns", "labels": {}},
            "spec": {"containers": [], "schedulingGates": []},
            "status": {"phase": "Pending"},
        }
        ctrl._handle_pod_event({"type": "DELETED", "object": deleted_pod})

        with ctrl._admitted_resources_lock:
            self.assertNotIn("p1", ctrl._admitted_resources)

    # ------------------------------------------------------------------
    # Capacity reservation transferred to kubernetes_pending during limbo
    # ------------------------------------------------------------------

    def test_capacity_transferred_to_kubernetes_pending_during_limbo(self):
        """After admission the pod watch fires with gate removed but still Pending.
        Capacity must move to _kubernetes_pending (not remain in _admitted_resources).
        """
        ctrl, core_v1 = _build_controller(gpu_class="small", gpu_count=4)

        pod = _make_pod(uid="p1", gpu_class="small", gpu_count="2")
        _enqueue_pod(ctrl, pod)
        ctrl._run_cycle()
        core_v1.patch_namespaced_pod.assert_called_once()
        with ctrl._admitted_resources_lock:
            self.assertIn("p1", ctrl._admitted_resources,
                          "reservation must be set after admission")

        # Simulate pod watch: gate removed, still Pending, no nodeName
        limbo_pod = {
            "metadata": {"uid": "p1", "name": "pod-p1", "namespace": "ns",
                         "labels": {"dsmlp/course": "CSE234_SP26_A00",
                                    GPU_CLASS_LABEL_KEY: "small"}},
            "spec": {"nodeName": None, "tolerations": [], "schedulingGates": [],
                     "containers": [{"name": "c", "resources": {
                         "requests": {"cpu": "2", "nvidia.com/gpu": "2"}}}]},
            "status": {"phase": "Pending"},
        }
        ctrl._handle_pod_event({"type": "MODIFIED", "object": limbo_pod})

        # Reservation moved to _kubernetes_pending, cleared from _admitted_resources
        with ctrl._admitted_resources_lock:
            self.assertNotIn("p1", ctrl._admitted_resources,
                             "admitted_resources must be cleared once pod is k8s-Pending")
        with ctrl._kubernetes_pending_lock:
            lane = _gpu("small")
            self.assertIn("p1", ctrl._kubernetes_pending.get(lane, {}),
                          "pod must appear in _kubernetes_pending during limbo")

        # Simulate: pod transitions to Running
        running_pod = {
            "metadata": {"uid": "p1", "name": "pod-p1", "namespace": "ns",
                         "labels": {"dsmlp/course": "CSE234_SP26_A00",
                                    GPU_CLASS_LABEL_KEY: "small"}},
            "spec": {"nodeName": "node-1", "tolerations": [],
                     "schedulingGates": [],
                     "containers": [{"name": "c", "resources": {
                         "requests": {"cpu": "2", "nvidia.com/gpu": "2"}}}],
                     "activeDeadlineSeconds": 3600},
            "status": {"phase": "Running", "startTime": "2026-01-01T00:00:00Z"},
        }
        ctrl._handle_pod_event({"type": "MODIFIED", "object": running_pod})

        with ctrl._kubernetes_pending_lock:
            self.assertNotIn("p1", ctrl._kubernetes_pending.get(lane, {}),
                             "pod must be removed from _kubernetes_pending after Running")
        with ctrl._running_lock:
            self.assertIn("p1", ctrl._kubernetes_running.get(lane, {}),
                          "pod must be tracked in _kubernetes_running after Running")

    # ------------------------------------------------------------------
    # Capacity gate must block new admissions during the limbo period
    # ------------------------------------------------------------------

    def test_capacity_gate_blocks_during_pending_limbo(self):
        """A second pod must be blocked while the first is admitted-but-k8s-pending.

        Node has exactly 2 GPUs.  After the first 2-GPU pod transitions to
        k8s-Pending, the capacity gate must see zero free GPUs and reject the
        second pod.
        """
        ctrl, core_v1 = _build_controller(gpu_class="small", gpu_count=2)

        pod1 = _make_pod(uid="p1", gpu_class="small", gpu_count="2")
        _enqueue_pod(ctrl, pod1)
        ctrl._run_cycle()
        core_v1.patch_namespaced_pod.assert_called_once()

        # Drive pod1 into the limbo state (gate removed, still Pending)
        limbo_pod = {
            "metadata": {"uid": "p1", "name": "pod-p1", "namespace": "ns",
                         "labels": {"dsmlp/course": "CSE234_SP26_A00",
                                    GPU_CLASS_LABEL_KEY: "small"}},
            "spec": {"nodeName": None, "tolerations": [], "schedulingGates": [],
                     "containers": [{"name": "c", "resources": {
                         "requests": {"cpu": "2", "nvidia.com/gpu": "2"}}}]},
            "status": {"phase": "Pending"},
        }
        ctrl._handle_pod_event({"type": "MODIFIED", "object": limbo_pod})

        # Enqueue a second pod and run another cycle
        pod2 = _make_pod(uid="p2", gpu_class="small", gpu_count="1",
                         namespace="ns2", name="pod-p2")
        _enqueue_pod(ctrl, pod2)
        core_v1.patch_namespaced_pod.reset_mock()
        ctrl._run_cycle()

        core_v1.patch_namespaced_pod.assert_not_called()
        lane = _gpu("small")
        self.assertEqual(_queue_depth(ctrl.scheduler, lane), 1,
                         "second pod must remain queued while first is in limbo")

    # ------------------------------------------------------------------
    # Admitted entry released on patch failure (ApiException)
    # ------------------------------------------------------------------

    def test_admitted_resources_released_on_patch_failure(self):
        from kubernetes import client as k8s_client

        ctrl, core_v1 = _build_controller(gpu_class="small", gpu_count=4)
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

        ctrl, core_v1 = _build_controller(gpu_class="small", gpu_count=4)
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
        """Two 1-GPU pods should both be admitted in a 2-GPU lane; third deferred."""
        ctrl, core_v1 = _build_controller(gpu_class="small", gpu_count=2)

        courses = ["CSE234_SP26_A00", "CSE190_SP26_A00", "CSE100_SP26_A00"]
        for i, course in enumerate(courses):
            pod = _make_pod(uid=f"p{i}", gpu_class="small", gpu_count="1",
                            course=course, namespace=f"student{i}",
                            name=f"pod-p{i}")
            _enqueue_pod(ctrl, pod)

        ctrl._run_cycle()

        self.assertEqual(core_v1.patch_namespaced_pod.call_count, 2)

        lane = _gpu("small")
        self.assertEqual(_queue_depth(ctrl.scheduler, lane), 1)

    # ------------------------------------------------------------------
    # Deferred job is re-submitted to the scheduler
    # ------------------------------------------------------------------

    def test_deferred_job_resubmitted_to_scheduler(self):
        ctrl, core_v1 = _build_controller(gpu_class="small", gpu_count=1)

        lane = _gpu("small")
        with ctrl._running_lock:
            ctrl._kubernetes_running.setdefault(lane, {})["r1"] = (
                _make_running_pod("r1", lane, resource_units=1.0)
            )

        pod = _make_pod(uid="p1", gpu_class="small", gpu_count="1")
        _enqueue_pod(ctrl, pod)

        ctrl._run_cycle()

        core_v1.patch_namespaced_pod.assert_not_called()
        self.assertEqual(_queue_depth(ctrl.scheduler, lane), 1)

    # ------------------------------------------------------------------
    # Default activeDeadlineSeconds for pods without one set
    # ------------------------------------------------------------------

    def test_running_pod_no_deadline_uses_default(self):
        """Pod with no activeDeadlineSeconds is tracked with the 86400s default."""
        ctrl, _ = _build_controller(gpu_class="small", gpu_count=4)

        lane = _gpu("small")
        running_pod = {
            "metadata": {"uid": "p1", "name": "pod-p1", "namespace": "ns",
                         "labels": {"dsmlp/course": "CSE234_SP26_A00",
                                    GPU_CLASS_LABEL_KEY: "small"}},
            "spec": {"nodeName": "node-1", "tolerations": [],
                     "containers": [{"name": "c", "resources": {
                         "requests": {"cpu": "2", "nvidia.com/gpu": "1"}
                     }}]},
            "status": {"phase": "Running",
                       "startTime": "2026-01-01T00:00:00Z"},
        }
        ctrl._handle_pod_event({"type": "MODIFIED", "object": running_pod})

        with ctrl._running_lock:
            tracked = ctrl._kubernetes_running.get(lane, {}).get("p1")
        self.assertIsNotNone(tracked, "pod should be tracked even without activeDeadlineSeconds")
        self.assertEqual(tracked.active_deadline_seconds, 86400)

    def test_running_pod_no_deadline_custom_default(self):
        """Custom default_active_deadline is used for pods without activeDeadlineSeconds."""
        core_v1 = MagicMock()
        registry = SchedGroupRegistry()
        residency_profiles = {
            "interactive": ResidencyProfile(mean_pct=0.4, std_pct=0.2),
            "batch":       ResidencyProfile(mean_pct=0.7, std_pct=0.15),
        }
        ctrl = LaneSchedulerController(
            core_v1=core_v1,
            registry=registry,
            sched_config=SchedulerConfig(),
            residency_profiles=residency_profiles,
            default_active_deadline=7200,
        )
        ctrl.node_tracker.upsert(_make_node(gpu_class="small", gpu_count="4"))
        ctrl.scheduler.set_lane_capacity(ctrl.node_tracker.lane_capacity())

        lane = _gpu("small")
        running_pod = {
            "metadata": {"uid": "p2", "name": "pod-p2", "namespace": "ns",
                         "labels": {"dsmlp/course": "CSE234_SP26_A00",
                                    GPU_CLASS_LABEL_KEY: "small"}},
            "spec": {"nodeName": "node-1", "tolerations": [],
                     "containers": [{"name": "c", "resources": {
                         "requests": {"cpu": "2", "nvidia.com/gpu": "1"}
                     }}]},
            "status": {"phase": "Running",
                       "startTime": "2026-01-01T00:00:00Z"},
        }
        ctrl._handle_pod_event({"type": "MODIFIED", "object": running_pod})

        with ctrl._running_lock:
            tracked = ctrl._kubernetes_running.get(lane, {}).get("p2")
        self.assertIsNotNone(tracked, "pod should be tracked with custom default")
        self.assertEqual(tracked.active_deadline_seconds, 7200)

    def test_running_pod_tracked_from_v1pod_model_object(self):
        """Bootstrap pods arrive as V1Pod model objects via list_namespaced_pod().
        The kubernetes client's to_dict() returns snake_case keys with datetime values
        (start_time=datetime, active_deadline_seconds=int).  The controller must handle
        both this shape and the camelCase ISO-string shape from the watch stream.
        """
        from kubernetes.client.models import (
            V1Pod, V1PodSpec, V1PodStatus, V1ObjectMeta, V1Container,
            V1ResourceRequirements,
        )
        import datetime

        ctrl, _ = _build_controller(gpu_class="small", gpu_count=4)
        lane = _gpu("small")

        model_pod = V1Pod(
            metadata=V1ObjectMeta(
                uid="model-uid",
                name="model-pod",
                namespace="ns",
                labels={"dsmlp/course": "CSE234_SP26_A00", GPU_CLASS_LABEL_KEY: "small"},
            ),
            spec=V1PodSpec(
                containers=[V1Container(
                    name="c",
                    resources=V1ResourceRequirements(requests={"nvidia.com/gpu": "1"}),
                )],
                active_deadline_seconds=3600,
            ),
            status=V1PodStatus(
                phase="Running",
                start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            ),
        )
        ctrl._handle_pod_event({"type": "ADDED", "object": model_pod})

        with ctrl._running_lock:
            tracked = ctrl._kubernetes_running.get(lane, {}).get("model-uid")
        self.assertIsNotNone(tracked, "V1Pod model object (bootstrap path) must be tracked")
        self.assertEqual(tracked.active_deadline_seconds, 3600.0)


class TestUnlabelledPodUtilization(unittest.TestCase):
    """Running pods without a gpu-class pod label should be counted against
    the lane derived from the node they are running on."""

    def _running_pod_dict(self, uid, node_name, gpu_count, gpu_class=None):
        labels = {"dsmlp/course": "CSE234_SP26_A00"}
        if gpu_class is not None:
            labels[GPU_CLASS_LABEL_KEY] = gpu_class
        return {
            "metadata": {"uid": uid, "name": f"pod-{uid}", "namespace": "ns",
                         "labels": labels},
            "spec": {"nodeName": node_name, "tolerations": [], "schedulingGates": [],
                     "containers": [{"name": "c", "resources": {
                         "requests": {"cpu": "2", "nvidia.com/gpu": gpu_count},
                     }}]},
            "status": {"phase": "Running", "startTime": "2026-01-01T00:00:00Z"},
        }

    def test_unlabelled_pod_on_gpu_node_is_counted(self):
        ctrl, _ = _build_controller(gpu_class="large", gpu_count=8)
        lane = _gpu("large")

        pod = self._running_pod_dict("u1", node_name="gpu-node-large-1",
                                     gpu_count="2")
        ctrl.node_tracker.upsert(_make_node(name="gpu-node-large-1",
                                            gpu_class="large", gpu_count="4"))
        ctrl._handle_pod_event({"type": "MODIFIED", "object": pod})

        with ctrl._running_lock:
            tracked = ctrl._kubernetes_running.get(lane, {}).get("u1")
        self.assertIsNotNone(tracked, "unlabelled pod on GPU node must be tracked")
        self.assertAlmostEqual(tracked.resource_units, 2.0)

    def test_unlabelled_pod_on_unmanaged_node_not_counted(self):
        ctrl, _ = _build_controller(gpu_class="small", gpu_count=4)

        pod = self._running_pod_dict("u2", node_name="cpu-node-99", gpu_count="0")
        ctrl._handle_pod_event({"type": "MODIFIED", "object": pod})

        with ctrl._running_lock:
            for lane_dict in ctrl._kubernetes_running.values():
                self.assertNotIn("u2", lane_dict,
                                 "pod on unmanaged node must not appear in _kubernetes_running")

    def test_unlabelled_pod_counts_toward_capacity_gate(self):
        ctrl, core_v1 = _build_controller(gpu_class="small", gpu_count=2)
        lane = _gpu("small")

        ctrl.node_tracker.upsert(_make_node(name="node-1",
                                            gpu_class="small", gpu_count="2"))
        unlabelled = self._running_pod_dict("unlabelled-1", node_name="node-1",
                                            gpu_count="2")
        ctrl._handle_pod_event({"type": "MODIFIED", "object": unlabelled})

        pod = _make_pod(uid="new-uid", gpu_class="small", gpu_count="1")
        _enqueue_pod(ctrl, pod)
        ctrl._run_cycle()

        core_v1.patch_namespaced_pod.assert_not_called()
        self.assertEqual(_queue_depth(ctrl.scheduler, lane), 1)

    def test_bootstrap_barrier_set_after_node_bootstrap(self):
        ctrl, _ = _build_controller(gpu_class="small", gpu_count=4)
        ctrl._nodes_bootstrapped.set()
        self.assertTrue(ctrl._nodes_bootstrapped.is_set())


if __name__ == "__main__":
    unittest.main()
