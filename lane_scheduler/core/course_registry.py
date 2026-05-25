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

The course_id field accepts shell glob patterns (fnmatch syntax: *, ?, [...]).
For example, "RITS_*" matches all courses whose ID starts with "RITS_".
Exact entries always take precedence over patterns; among patterns, the first
match in CSV order wins.  Glob-resolved entries are cached so repeated lookups
are O(1).

Reload at any time by calling CourseRegistry.load_csv(); the registry is
replaced atomically so the controller never sees a partially-loaded state.
"""

from __future__ import annotations

import csv
import fnmatch
import logging
import threading
from pathlib import Path

from lane_scheduler.core.scheduler import CourseClass

logger = logging.getLogger(__name__)

_FALLBACK_WEIGHT = 1.0


def _is_glob(s: str) -> bool:
    return any(c in s for c in ("*", "?", "["))


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
        self._exact: dict[str, CourseClass] = {}
        self._patterns: list[tuple[str, CourseClass]] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_csv(self, path: Path) -> int:
        """
        (Re)load course data from *path*.  Returns the number of courses loaded.
        Raises FileNotFoundError or ValueError on bad input; the existing registry
        is left intact on any failure.
        """
        new_exact: dict[str, CourseClass] = {}
        new_patterns: list[tuple[str, CourseClass]] = []

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
                entry = CourseClass(class_id=course_id, class_weight=weight)
                if _is_glob(course_id):
                    new_patterns.append((course_id, entry))
                else:
                    new_exact[course_id] = entry

        with self._lock:
            self._exact    = new_exact
            self._patterns = new_patterns

        total = len(new_exact) + len(new_patterns)
        logger.info("Loaded %d courses from %s", total, path)
        return total

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, course_id: str) -> CourseClass:
        """
        Return the CourseClass for *course_id*.  If unknown, return a synthetic
        entry with weight=1.0 and log a warning.  Never raises.

        Resolution order:
          1. Exact match in the CSV.
          2. First glob pattern in CSV order that matches (fnmatch semantics).
          3. Synthetic fallback with weight=1.0.
        Glob-resolved entries are cached so subsequent lookups are O(1).
        """
        with self._lock:
            course = self._exact.get(course_id)
            if course is not None:
                return course
            for pat, template in self._patterns:
                if fnmatch.fnmatch(course_id, pat):
                    resolved = CourseClass(
                        class_id=course_id,
                        class_weight=template.class_weight,
                    )
                    self._exact.setdefault(course_id, resolved)
                    return resolved

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
            self._exact.setdefault(course_id, synthetic)

        return synthetic

    def all_courses(self) -> list[CourseClass]:
        with self._lock:
            return list(self._exact.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._exact)
