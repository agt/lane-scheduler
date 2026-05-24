"""
Course Registry
---------------
Loads course metadata (scheduling weight) from a CSV file exported by the
registrar.  Pods that reference an unknown course default to weight 1.0.

Expected CSV columns (order-independent, header required):
    course_id, weight
    e.g.:
        CSE234_SP26_A00,0.775
        CSE101_SP26_A00,0.071
        CSE150_SP26_A00,0.270

    weight must be a positive float.  It is used directly as W in the
    priority formula P = W × Mode × Age / U.

Reload at any time by calling CourseRegistry.load_csv(); the registry is
replaced atomically so the controller never sees a partially-loaded state.
"""

from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path

from lane_scheduler.core.scheduler import CourseClass

logger = logging.getLogger(__name__)

_FALLBACK_WEIGHT = 1.0


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
            required = {"course_id", "weight"}
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
                    weight = float(row["weight"].strip())
                    if weight <= 0:
                        raise ValueError("weight must be positive")
                except ValueError:
                    logger.warning("Bad weight %r for %s — skipping",
                                   row["weight"], course_id)
                    continue
                new_courses[course_id] = CourseClass(
                    class_id=course_id, class_weight=weight
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
        entry with weight=1.0 and log a warning.  Never raises.
        """
        with self._lock:
            course = self._courses.get(course_id)

        if course is not None:
            return course

        logger.warning(
            "Unknown course %r — defaulting to weight=%.3f",
            course_id, _FALLBACK_WEIGHT,
        )
        synthetic = CourseClass(
            class_id     = course_id,
            class_weight = _FALLBACK_WEIGHT,
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
