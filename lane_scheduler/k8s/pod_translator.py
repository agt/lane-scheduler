"""
Pod Translator
--------------
Converts raw Kubernetes pod dicts (as returned by the Python kubernetes client)
into scheduler domain objects, and produces the patch payload needed to admit
a pod by removing its scheduling gate and adding the matching GPU-class
NoSchedule toleration.

Admission gate model
~~~~~~~~~~~~~~~~~~~~
    An external mutating admission controller injects a scheduling gate
    ("lane-scheduler") and the gpu-class label onto every pod of interest
    before it reaches our controller.  The pod is held in the SchedulingGated /
    Pending state until we:

        a) add a nodeSelector entry (gpu-class=<class>) so the default
           scheduler targets only the correct GPU nodes,
        b) add the gpu-class=<class>:NoSchedule toleration so the pod can land
           on tainted GPU nodes, and
        c) remove the "lane-scheduler" schedulingGate entry.

    After both operations the default Kubernetes scheduler places the pod on a
    matching node.

Label / annotation contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dsmlp/course        : course identifier, e.g. "CSE234_SP26_A00"
                          Required.  Pod is held under NO_COURSE_LABEL if absent.
    gpu-class           : one of {xsmall, small, medium, large, xlarge}
                          Injected by the mutating admission controller.
                          Pods without this label are rejected with an error Event.
    dsmlp/batch         : "true" (case-insensitive) → batch mode scoring penalty
                          Absent or any other value → interactive priority.
                          Override key with LANE_BATCH_LABEL env var.

Resource → lane mapping
~~~~~~~~~~~~~~~~~~~~~~~
    gpu-class=xsmall                → "gpu-xsmall"
    gpu-class=small                 → "gpu-small"
    gpu-class=medium                → "gpu-medium"
    gpu-class=large                 → "gpu-large"
    gpu-class=xlarge                → "gpu-xlarge"
    No gpu-class label / unknown    → rejected; error Event emitted; ValueError
                                      raised by pod_to_job()

Resource units
~~~~~~~~~~~~~~
    GPU lanes : total nvidia.com/gpu count requested, floor 1.0
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from lane_scheduler.core.scheduler import Job, lane_for_gpu_class
from lane_scheduler.core.node_capacity import GPU_CLASS_LABEL_KEY as _GPU_TAINT_KEY

logger = logging.getLogger(__name__)

# Labels we read
LABEL_COURSE     = os.environ.get("LANE_COURSE_LABEL",     "dsmlp/course")
LABEL_BATCH      = os.environ.get("LANE_BATCH_LABEL",      "dsmlp/batch")
# Label key on pods that identifies the requested GPU class / lane.
# Override with LANE_POD_GPU_CLASS_LABEL if your cluster uses a different key.
LABEL_GPU_CLASS  = os.environ.get("LANE_GPU_CLASS_LABEL", "gpu-class")

# Name of the scheduling gate injected by the mutating admission controller.
SCHEDULING_GATE_NAME = os.environ.get("LANE_SCHEDULING_GATE_NAME", "lane-scheduler")

# Fallback course label when pod carries none
_NO_COURSE_LABEL = "__unlabelled__"

# Resource request key for GPUs
_GPU_RESOURCE = "nvidia.com/gpu"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _labels(pod: dict) -> dict[str, str]:
    return pod.get("metadata", {}).get("labels", {}) or {}


def _is_batch(pod: dict) -> bool:
    return _labels(pod).get(LABEL_BATCH, "").strip().lower() == "true"


def _gpu_lane(pod: dict) -> Optional[str]:
    """Return the GPU lane string from the gpu-class label, or None if absent."""
    gpu_class = _labels(pod).get(LABEL_GPU_CLASS, "").strip()
    if not gpu_class:
        return None
    return lane_for_gpu_class(gpu_class)


def _parse_gpu_count(value: str) -> float:
    return float(value.strip())


def _resource_units(pod: dict, gpu_lane: Optional[str]) -> float:
    """
    Returns a resource scalar for utilization accounting.
        GPU lanes : nvidia.com/gpu count, floor 1.0
        Unknown lane (gpu_lane=None): returns 1.0
    """
    if gpu_lane is None:
        return 1.0
    containers = pod.get("spec", {}).get("containers", []) or []
    total_gpu  = 0.0
    for container in containers:
        requests = (container.get("resources", {}) or {}).get("requests", {}) or {}
        if _GPU_RESOURCE in requests:
            try:
                total_gpu += _parse_gpu_count(requests[_GPU_RESOURCE])
            except ValueError:
                logger.debug("Unparseable GPU request %r in pod %s",
                             requests[_GPU_RESOURCE], _pod_name(pod))
    return max(total_gpu, 1.0)


def _pod_name(pod: dict) -> str:
    meta = pod.get("metadata", {}) or {}
    return f"{meta.get('namespace', '?')}/{meta.get('name', '?')}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pod_to_job(pod: dict, submit_time: Optional[float] = None) -> Job:
    """
    Convert a Kubernetes pod dict to a scheduler Job.

        student_id  = pod namespace          (one namespace per student)
        class_id    = dsmlp/course label
        job_id      = pod UID
        lane        = gpu-class label → GPU lane string
        batch       = dsmlp/batch == "true"
        resource_units = nvidia.com/gpu count requested, floor 1.0

    Raises ValueError if the pod has no recognised gpu-class label.
    """
    meta      = pod.get("metadata", {}) or {}
    namespace = meta.get("namespace", "unknown")
    uid       = meta.get("uid", _pod_name(pod))
    labels    = _labels(pod)

    course_id = labels.get(LABEL_COURSE, _NO_COURSE_LABEL).strip() or _NO_COURSE_LABEL
    batch     = _is_batch(pod)
    gpu_lane  = _gpu_lane(pod)
    if gpu_lane is None:
        gpu_class = labels.get(LABEL_GPU_CLASS, "").strip()
        raise ValueError(
            f"Pod {_pod_name(pod)} has no recognised gpu-class label "
            f"(got {gpu_class!r}); cannot assign to a lane"
        )
    lane  = gpu_lane
    units = _resource_units(pod, gpu_lane)

    job = Job(
        job_id         = uid,
        class_id       = course_id,
        student_id     = namespace,
        lane           = lane,
        batch          = batch,
        resource_units = units,
    )
    job.submit_time = submit_time if submit_time is not None else time.monotonic()

    logger.debug(
        "Translated pod %s → job %s [course=%s lane=%s batch=%s units=%.2f]",
        _pod_name(pod), uid, course_id, lane, batch, units,
    )
    return job


def needs_scheduling(pod: dict) -> bool:
    """
    True if the pod is Pending, not yet node-assigned, and still carries
    the lane-scheduler scheduling gate (i.e. has not been admitted yet).
    """
    phase = (pod.get("status", {}) or {}).get("phase", "")
    if phase != "Pending":
        return False

    if (pod.get("spec", {}) or {}).get("nodeName"):
        return False

    gates = (pod.get("spec", {}) or {}).get("schedulingGates", []) or []
    return any(g.get("name") == SCHEDULING_GATE_NAME for g in gates)


def admission_patch(pod: dict) -> dict:
    """
    Build a strategic-merge patch that admits the pod for Kubernetes scheduling:

        a) Adds a nodeSelector entry (gpu-class=<class>) so the default
           scheduler targets only nodes in the correct GPU lane.
        b) Adds the gpu-class=<class>:NoSchedule toleration so the pod can
           land on the tainted GPU node.
        c) Removes the "lane-scheduler" schedulingGate entry so the default
           Kubernetes scheduler picks up the pod.

    Returns {} if the scheduling gate is already absent (idempotent).

    Apply via:
        core_v1.patch_namespaced_pod(name, namespace, body=admission_patch(pod))
    """
    gates = (pod.get("spec", {}) or {}).get("schedulingGates", []) or []
    if not any(g.get("name") == SCHEDULING_GATE_NAME for g in gates):
        return {}   # gate already removed — nothing to do

    gpu_class = _labels(pod).get(LABEL_GPU_CLASS, "").strip()
    patch: dict = {"spec": {}}

    # a) nodeSelector: direct the default scheduler to the right GPU lane.
    existing_node_selector = dict((pod.get("spec", {}) or {}).get("nodeSelector", {}) or {})
    if gpu_class and existing_node_selector.get(_GPU_TAINT_KEY) != gpu_class:
        patch["spec"]["nodeSelector"] = {**existing_node_selector, _GPU_TAINT_KEY: gpu_class}

    # b) GPU-class NoSchedule toleration
    existing_tolerations = list((pod.get("spec", {}) or {}).get("tolerations", []) or [])
    already_tolerated = any(
        t.get("key") == _GPU_TAINT_KEY and t.get("value") == gpu_class
        for t in existing_tolerations
    )
    if gpu_class and not already_tolerated:
        patch["spec"]["tolerations"] = existing_tolerations + [{
            "key":      _GPU_TAINT_KEY,
            "operator": "Equal",
            "value":    gpu_class,
            "effect":   "NoSchedule",
        }]

    # c) Remove the lane-scheduler scheduling gate.
    # Use the $patch:delete directive so other gates (if any) are preserved.
    patch["spec"]["schedulingGates"] = [
        {"$patch": "delete", "name": SCHEDULING_GATE_NAME}
    ]

    return patch


NO_COURSE_LABEL = _NO_COURSE_LABEL
