"""
Node Capacity Tracker
---------------------
Maintains a live view of schedulable cluster capacity per resource lane by
watching node ADDED/MODIFIED/DELETED events.

Lane capacity units match pod_translator:
    GPU_XSMALL … XLARGE: total allocatable GPU count per gpu-class

Node → Lane classification
~~~~~~~~~~~~~~~~~~~~~~~~~~
    A node is part of the managed GPU pool if it carries a gpu-class label
    whose key is GPU_CLASS_LABEL_KEY (default "gpu-class", overridable via
    LANE_GPU_CLASS_LABEL), e.g.:
        gpu-class=xlarge  →  Lane.GPU_XLARGE

    Nodes without the gpu-class label are not part of the managed pool and
    are ignored by this tracker.

    Nodes labelled with an unrecognised gpu-class (one not discovered at
    controller startup) are also excluded.

    Unready or unschedulable nodes are excluded from capacity totals.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

from lane_scheduler.core.scheduler import Lane, lane_for_gpu_class

logger = logging.getLogger(__name__)

# Taint applied to all cluster nodes to inhibit default scheduling
INHIBIT_TAINT_KEY    = os.environ.get("LANE_INHIBIT_TAINT_KEY",   "dsmlp/scheduling-gate")
INHIBIT_TAINT_VALUE  = os.environ.get("LANE_INHIBIT_TAINT_VALUE", "controller")
INHIBIT_TAINT_EFFECT = "NoSchedule"

# Label key on both nodes and pods that identifies the GPU class / lane.
# Override with LANE_GPU_CLASS_LABEL if your cluster uses a different key.
GPU_CLASS_LABEL_KEY = os.environ.get("LANE_GPU_CLASS_LABEL", "gpu-class")

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


def _gpu_class_lane(node: dict) -> tuple[Optional[str], str]:
    """
    Return (lane, raw_label) for the node's gpu-class label.

    lane is None if either the label is absent OR present-but-unknown to this
    controller.  Callers should use raw_label to distinguish those two cases.
    """
    labels = (node.get("metadata", {}) or {}).get("labels", {}) or {}
    gpu_class = labels.get(GPU_CLASS_LABEL_KEY, "").strip()
    if gpu_class:
        return lane_for_gpu_class(gpu_class, strict=True), gpu_class
    return None, ""


def _parse_gpu_count(value: str) -> float:
    return float(value.strip())


# ---------------------------------------------------------------------------
# NodeInfo
# ---------------------------------------------------------------------------

@dataclass
class NodeInfo:
    name:        str
    lane:        str
    gpu_count:   float
    ready:       bool
    schedulable: bool

    @property
    def active(self) -> bool:
        return self.ready and self.schedulable

    @property
    def capacity(self) -> float:
        return self.gpu_count


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

        gpu_lane, raw_gpu_class = _gpu_class_lane(node)

        # Only manage nodes with a gpu-class label.
        if not raw_gpu_class:
            logger.debug("Node %s: no gpu-class label — not in managed GPU pool", name)
            with self._lock:
                self._nodes.pop(name, None)
            return

        # Unknown gpu-class → exclude to avoid mis-routing pods.
        if gpu_lane is None:
            logger.warning(
                "Node %s has unmanaged gpu-class label %r — excluding from pool. "
                "Restart controller after the class is added to recognise it.",
                name, raw_gpu_class,
            )
            with self._lock:
                self._nodes.pop(name, None)
            return

        allocatable = (node.get("status", {}) or {}).get("allocatable", {}) or {}

        gpu_count = 0.0
        if _GPU_RESOURCE in allocatable:
            try:
                gpu_count = _parse_gpu_count(allocatable[_GPU_RESOURCE])
            except ValueError:
                logger.warning("Unparseable GPU allocatable on node %s: %r",
                               name, allocatable[_GPU_RESOURCE])

        if gpu_count == 0:
            logger.warning("Node %s has gpu-class label %r but zero GPU allocatable",
                           name, raw_gpu_class)

        info = NodeInfo(
            name        = name,
            lane        = gpu_lane,
            gpu_count   = gpu_count,
            ready       = _is_ready(node),
            schedulable = _is_schedulable(node),
        )
        with self._lock:
            self._nodes[name] = info

        logger.info(
            "Node %s upserted [lane=%s gpu=%.0f ready=%s schedulable=%s]",
            name, gpu_lane, gpu_count, info.ready, info.schedulable,
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
        totals: dict = {lane: 0.0 for lane in _Lane} if _Lane else {}
        with self._lock:
            for info in self._nodes.values():
                if info.active:
                    totals.setdefault(info.lane, 0.0)
                    totals[info.lane] += info.capacity
        return totals

    def lane_for_node(self, node_name: str) -> Optional[str]:
        """Return the GPU lane for a named node, or None if not in the managed pool."""
        with self._lock:
            info = self._nodes.get(node_name)
            return info.lane if info is not None else None

    def node_count(self) -> dict:
        from lane_scheduler.core.scheduler import Lane as _Lane
        counts: dict = {lane: 0 for lane in _Lane} if _Lane else {}
        with self._lock:
            for info in self._nodes.values():
                if info.active:
                    counts.setdefault(info.lane, 0)
                    counts[info.lane] += 1
        return counts

    def summary(self) -> str:
        from lane_scheduler.core.scheduler import Lane as _Lane
        caps   = self.lane_capacity()
        counts = self.node_count()
        parts  = [
            f"{lane}: {counts.get(lane, 0)} nodes / {caps.get(lane, 0.0):.1f} units"
            for lane in sorted(_Lane or [])
        ]
        return " | ".join(parts)
