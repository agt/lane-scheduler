"""
Pod Translator
--------------
Converts raw Kubernetes pod dicts (as returned by the Python kubernetes client)
into scheduler domain objects, and produces the patch payload needed to admit
a pod by adding the inhibitory-taint toleration.

Label / annotation contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dsmlp/course        : course identifier, e.g. "CSE234_SP26_A00"
                          Required.  Pod is held under NO_COURSE_LABEL if absent.
    gpu-class           : one of {xsmall, small, medium, large, xlarge}
                          Present on GPU pods; injected by the existing mutating
                          admission controller alongside nodeSelector+toleration.
                          Absent → CPU lane.
    dsmlp/batch         : "true" (case-insensitive) → batch mode scoring penalty
                          Absent or any other value → interactive priority.

Resource → Lane mapping
~~~~~~~~~~~~~~~~~~~~~~~
    No gpu-class label              → Lane.CPU
    gpu-class=xsmall                → Lane.GPU_XSMALL
    gpu-class=small                 → Lane.GPU_SMALL
    gpu-class=medium                → Lane.GPU_MEDIUM
    gpu-class=large                 → Lane.GPU_LARGE
    gpu-class=xlarge                → Lane.GPU_XLARGE
    gpu-class=<unrecognised>        → Lane.GPU_SMALL  (logged as warning)

Resource units
~~~~~~~~~~~~~~
    CPU lane  : total CPU cores requested (millicores normalised), floor 1.0
    GPU lanes : total nvidia.com/gpu count requested, floor 1.0
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from lane_scheduler.core.scheduler import Job, Lane, lane_for_gpu_class
from lane_scheduler.core.node_capacity import (
    INHIBIT_TAINT_KEY, INHIBIT_TAINT_VALUE, INHIBIT_TAINT_EFFECT,
)

logger = logging.getLogger(__name__)

# Labels we read
LABEL_COURSE     = os.environ.get("LANE_COURSE_LABEL", "dsmlp/course")
LABEL_BATCH      = "dsmlp/batch"
# Label key on pods that identifies the requested GPU class / lane.
# Override with LANE_POD_GPU_CLASS_LABEL if your cluster uses a different key.
LABEL_GPU_CLASS  = os.environ.get("LANE_POD_GPU_CLASS_LABEL", "gpu-class")

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


def _gpu_lane(pod: dict) -> Optional[object]:
    """Return the GPU Lane member from the gpu-class label, or None if absent."""
    gpu_class = _labels(pod).get(LABEL_GPU_CLASS, "").strip()
    if not gpu_class:
        return None
    return lane_for_gpu_class(gpu_class)


def _cpu_lane() -> object:
    """Return Lane.CPU — resolved lazily so the dynamic enum is always current."""
    from lane_scheduler.core.scheduler import Lane as _Lane
    return _Lane.CPU


def _parse_cpu_millicores(value: str) -> float:
    value = value.strip()
    if value.endswith("m"):
        return float(value[:-1])
    return float(value) * 1000.0


def _parse_gpu_count(value: str) -> float:
    return float(value.strip())


def _resource_units(pod: dict, gpu_lane: Optional[Lane]) -> float:
    """
    Returns a resource scalar for utilization accounting.
        GPU lanes : nvidia.com/gpu count, floor 1.0
        CPU lane  : CPU cores, floor 1.0
    """
    containers = pod.get("spec", {}).get("containers", []) or []
    total_cpu_mc = 0.0
    total_gpu    = 0.0

    for container in containers:
        requests = (container.get("resources", {}) or {}).get("requests", {}) or {}
        if "cpu" in requests:
            try:
                total_cpu_mc += _parse_cpu_millicores(requests["cpu"])
            except ValueError:
                logger.debug("Unparseable CPU request %r in pod %s",
                             requests["cpu"], _pod_name(pod))
        if _GPU_RESOURCE in requests:
            try:
                total_gpu += _parse_gpu_count(requests[_GPU_RESOURCE])
            except ValueError:
                logger.debug("Unparseable GPU request %r in pod %s",
                             requests[_GPU_RESOURCE], _pod_name(pod))

    if gpu_lane is not None:
        return max(total_gpu, 1.0)
    return max(total_cpu_mc / 1000.0, 1.0)


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
        lane        = gpu-class label → GPU lane, or CPU if absent
        batch       = dsmlp/batch == "true"
        resource_units = GPU count (GPU lanes) or CPU cores (CPU lane)
    """
    meta      = pod.get("metadata", {}) or {}
    namespace = meta.get("namespace", "unknown")
    uid       = meta.get("uid", _pod_name(pod))
    labels    = _labels(pod)

    course_id = labels.get(LABEL_COURSE, _NO_COURSE_LABEL).strip() or _NO_COURSE_LABEL
    batch     = _is_batch(pod)
    gpu_lane  = _gpu_lane(pod)
    lane      = gpu_lane if gpu_lane is not None else _cpu_lane()
    units     = _resource_units(pod, gpu_lane)

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
        _pod_name(pod), uid, course_id, lane.name, batch, units,
    )
    return job


def needs_scheduling(pod: dict) -> bool:
    """
    True if the pod is Pending, not yet node-assigned, and hasn't already
    received our scheduling-gate toleration.
    """
    phase = (pod.get("status", {}) or {}).get("phase", "")
    if phase != "Pending":
        return False

    if (pod.get("spec", {}) or {}).get("nodeName"):
        return False

    tolerations = (pod.get("spec", {}) or {}).get("tolerations", []) or []
    for t in tolerations:
        if (t.get("key") == INHIBIT_TAINT_KEY
                and t.get("value") == INHIBIT_TAINT_VALUE):
            return False

    return True


def admission_patch(pod: dict) -> dict:
    """
    Build a strategic-merge patch body that adds the scheduling-gate toleration,
    allowing the default Kubernetes scheduler to place the pod on a tainted node.

    Returns {} if the toleration is already present (idempotent).

    Apply via:
        core_v1.patch_namespaced_pod(name, namespace, body=admission_patch(pod))
    """
    existing = list((pod.get("spec", {}) or {}).get("tolerations", []) or [])
    for t in existing:
        if t.get("key") == INHIBIT_TAINT_KEY:
            return {}   # already patched

    return {
        "spec": {
            "tolerations": existing + [{
                "key":      INHIBIT_TAINT_KEY,
                "operator": "Equal",
                "value":    INHIBIT_TAINT_VALUE,
                "effect":   INHIBIT_TAINT_EFFECT,
            }]
        }
    }


NO_COURSE_LABEL = _NO_COURSE_LABEL
