"""
Course Registry
---------------
Loads course metadata (tier + enrollment) from a CSV file exported by the
registrar.  Pods that reference an unknown course default to tier 1 and
200 seats.

Expected CSV columns (order-independent, header required):
    course_id, tier, seats
    e.g.:
        CSE234_SP26_A00,3,18
        CSE101_SP26_A00,1,210
        CSE150_SP26_A00,2,55

    tier must be 1 (intro/lower), 2 (upper-division), or 3 (graduate).

Reload at any time by calling CourseRegistry.load_csv(); the registry is
replaced atomically so the controller never sees a partially-loaded state.
"""

from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path

from lane_scheduler.core.scheduler import CourseClass, Tier

logger = logging.getLogger(__name__)

_FALLBACK_TIER       = Tier.INTRO
_FALLBACK_ENROLLMENT = 200


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class CourseRegistry:
    """
    Thread-safe store of CourseClass objects.

    Usage:
        registry = CourseRegistry()
        registry.load_csv(Path("/etc/lane-scheduler/courses.csv"))
        course = registry.get("CSE234_SP26_A00")   # always returns something
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._courses: dict[str, CourseClass] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_csv(self, path: Path) -> int:
        """
        (Re)load course data from *path*.  Returns the number of courses loaded.
        Raises FileNotFoundError or ValueError on bad input; the existing registry
        is left intact on any failure.
        """
        new_courses: dict[str, CourseClass] = {}

        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            required = {"course_id", "tier", "seats"}
            if not reader.fieldnames:
                raise ValueError(f"CSV {path} appears to be empty")
            missing = required - {f.strip().lower() for f in reader.fieldnames}
            if missing:
                raise ValueError(f"CSV {path} missing columns: {missing}")

            for row in reader:
                course_id = row["course_id"].strip()
                if not course_id:
                    continue
                try:
                    seats = int(row["seats"].strip())
                except ValueError:
                    logger.warning("Bad seat count for %s (%r) — skipping",
                                   course_id, row["seats"])
                    continue
                try:
                    tier_int = int(row["tier"].strip())
                    tier = Tier(tier_int)
                except (ValueError, KeyError):
                    logger.warning("Bad tier %r for %s — skipping",
                                   row["tier"], course_id)
                    continue
                new_courses[course_id] = CourseClass(
                    class_id=course_id, tier=tier, enrollment=seats
                )

        with self._lock:
            self._courses = new_courses

        logger.info("Loaded %d courses from %s", len(new_courses), path)
        return len(new_courses)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, course_id: str) -> CourseClass:
        """
        Return the CourseClass for *course_id*.  If unknown, return a synthetic
        entry with tier=1 and enrollment=200, and log a warning.  Never raises.
        """
        with self._lock:
            course = self._courses.get(course_id)

        if course is not None:
            return course

        logger.warning(
            "Unknown course %r — defaulting to tier=%d enrollment=%d",
            course_id, _FALLBACK_TIER.value, _FALLBACK_ENROLLMENT,
        )
        synthetic = CourseClass(
            class_id   = course_id,
            tier       = _FALLBACK_TIER,
            enrollment = _FALLBACK_ENROLLMENT,
        )

        # Cache the synthetic entry so repeated lookups don't keep warning
        with self._lock:
            self._courses.setdefault(course_id, synthetic)

        return synthetic

    def all_courses(self) -> list[CourseClass]:
        with self._lock:
            return list(self._courses.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._courses)
