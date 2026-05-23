"""
Node Capacity Tracker
---------------------
Maintains a live view of schedulable cluster capacity per resource lane by
watching node ADDED/MODIFIED/DELETED events.

Lane capacity units match pod_translator:
    CPU lane           : total allocatable CPU cores
    GPU_XSMALL … XLARGE: total allocatable GPU count per gpu-class

Node → Lane classification
~~~~~~~~~~~~~~~~~~~~~~~~~~
    Nodes are classified by a node label whose key is GPU_CLASS_LABEL_KEY
    (default "gpu-class", overridable via LANE_NODE_GPU_CLASS_LABEL), e.g.:
        gpu-class=xlarge  →  Lane.GPU_XLARGE

    A node with no gpu-class label but with the inhibitory scheduling-gate
    taint is treated as a CPU node.

    Nodes without the inhibitory scheduling-gate taint are ignored entirely —
    they are not part of our managed pool.

    Unready or unschedulable nodes are excluded from capacity totals.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

from lane_scheduler.core.scheduler import Lane, GPU_LANES, lane_for_gpu_class

logger = logging.getLogger(__name__)

# Taint applied to all cluster nodes to inhibit default scheduling
INHIBIT_TAINT_KEY    = "dsmlp/scheduling-gate"
INHIBIT_TAINT_VALUE  = "controller"
INHIBIT_TAINT_EFFECT = "NoSchedule"

# Label key on nodes that identifies the GPU class / lane.
# Override with LANE_NODE_GPU_CLASS_LABEL if your cluster uses a different key.
GPU_CLASS_LABEL_KEY = os.environ.get("LANE_NODE_GPU_CLASS_LABEL", "gpu-class")

_GPU_RESOURCE = "nvidia.com/gpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_name(node: dict) -> str:
    return (node.get("metadata", {}) or {}).get("name", "<unknown>")


def _is_ready(node: dict) -> bool:
    for c in (node.get("status", {}) or {}).get("conditions", []) or []:
        if c.get("type") == "Ready":
            return c.get("status") == "True"
    return False


def _is_schedulable(node: dict) -> bool:
    return not (node.get("spec", {}) or {}).get("unschedulable", False)


def _taints(node: dict) -> list[dict]:
    return (node.get("spec", {}) or {}).get("taints", []) or []


def _has_inhibit_taint(node: dict) -> bool:
    for t in _taints(node):
        if t.get("key") == INHIBIT_TAINT_KEY and t.get("value") == INHIBIT_TAINT_VALUE:
            return True
    return False


def _gpu_class_lane(node: dict) -> Optional[object]:
    """Return the GPU Lane member from the node's gpu-class label, or None."""
    labels = (node.get("metadata", {}) or {}).get("labels", {}) or {}
    gpu_class = labels.get(GPU_CLASS_LABEL_KEY, "").strip()
    if gpu_class:
        return lane_for_gpu_class(gpu_class)
    return None


def _parse_cpu_cores(value: str) -> float:
    value = value.strip()
    if value.endswith("m"):
        return float(value[:-1]) / 1000.0
    return float(value)


def _parse_gpu_count(value: str) -> float:
    return float(value.strip())


# ---------------------------------------------------------------------------
# NodeInfo
# ---------------------------------------------------------------------------

@dataclass
class NodeInfo:
    name:        str
    lane:        Lane
    cpu_cores:   float
    gpu_count:   float
    ready:       bool
    schedulable: bool

    @property
    def active(self) -> bool:
        return self.ready and self.schedulable

    @property
    def capacity(self) -> float:
        from lane_scheduler.core.scheduler import GPU_LANES as _GPU_LANES
        return self.gpu_count if self.lane in _GPU_LANES else self.cpu_cores


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class NodeCapacityTracker:
    """
    Thread-safe node capacity tracker.

        tracker = NodeCapacityTracker()
        tracker.upsert(node_dict)    # ADDED / MODIFIED
        tracker.remove(node_name)    # DELETED
        caps = tracker.lane_capacity()
    """

    def __init__(self) -> None:
        self._lock  = threading.RLock()
        self._nodes: dict[str, NodeInfo] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def upsert(self, node: dict) -> None:
        name = _node_name(node)

        if not _has_inhibit_taint(node):
            logger.debug("Node %s: no scheduling-gate taint — not in managed pool", name)
            with self._lock:
                self._nodes.pop(name, None)
            return

        allocatable = (node.get("status", {}) or {}).get("allocatable", {}) or {}

        cpu_cores = 0.0
        if "cpu" in allocatable:
            try:
                cpu_cores = _parse_cpu_cores(allocatable["cpu"])
            except ValueError:
                logger.warning("Unparseable CPU allocatable on node %s: %r",
                               name, allocatable["cpu"])

        gpu_count = 0.0
        if _GPU_RESOURCE in allocatable:
            try:
                gpu_count = _parse_gpu_count(allocatable[_GPU_RESOURCE])
            except ValueError:
                logger.warning("Unparseable GPU allocatable on node %s: %r",
                               name, allocatable[_GPU_RESOURCE])

        gpu_lane = _gpu_class_lane(node)
        if gpu_lane is not None and gpu_count > 0:
            lane = gpu_lane
        elif gpu_lane is not None and gpu_count == 0:
            logger.warning("Node %s has gpu-class label but zero GPU allocatable", name)
            from lane_scheduler.core.scheduler import Lane as _Lane
            lane = _Lane.CPU
        else:
            from lane_scheduler.core.scheduler import Lane as _Lane
            lane = _Lane.CPU

        info = NodeInfo(
            name        = name,
            lane        = lane,
            cpu_cores   = cpu_cores,
            gpu_count   = gpu_count,
            ready       = _is_ready(node),
            schedulable = _is_schedulable(node),
        )
        with self._lock:
            self._nodes[name] = info

        logger.info(
            "Node %s upserted [lane=%s cpu=%.1f gpu=%.0f ready=%s schedulable=%s]",
            name, lane.name, cpu_cores, gpu_count, info.ready, info.schedulable,
        )

    def remove(self, node_name: str) -> None:
        with self._lock:
            removed = self._nodes.pop(node_name, None)
        if removed:
            logger.info("Node %s removed from capacity tracker", node_name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def lane_capacity(self) -> dict:
        """Aggregate allocatable capacity per lane across all active nodes."""
        from lane_scheduler.core.scheduler import Lane as _Lane
        totals: dict = {lane: 0.0 for lane in _Lane}
        with self._lock:
            for info in self._nodes.values():
                if info.active:
                    totals[info.lane] += info.capacity
        return totals

    def node_count(self) -> dict:
        from lane_scheduler.core.scheduler import Lane as _Lane
        counts: dict = {lane: 0 for lane in _Lane}
        with self._lock:
            for info in self._nodes.values():
                if info.active:
                    counts[info.lane] += 1
        return counts

    def summary(self) -> str:
        caps   = self.lane_capacity()
        counts = self.node_count()
        parts  = [
            f"{lane.name}: {counts[lane]} nodes / {caps[lane]:.1f} units"
            for lane in Lane
        ]
        return " | ".join(parts)
