"""
Scheduling Group Registry
-------------------------
Returns a SchedGroup for any scheduling group identifier.

v2 stub: all scheduling groups are treated equally (weight=1.0).
Per-group weight configuration via a registrar CSV is planned for a
future release.
"""

from __future__ import annotations

from lane_scheduler.core.scheduler import SchedGroup


class SchedGroupRegistry:
    """
    Stateless registry that returns a default SchedGroup for any identifier.

    get() never raises and never returns None.  All groups receive weight=1.0,
    producing equal fair-share competition among scheduling groups.
    """

    def get(self, sched_group_id: str) -> SchedGroup:
        return SchedGroup(sched_group_id=sched_group_id, weight=1.0)
