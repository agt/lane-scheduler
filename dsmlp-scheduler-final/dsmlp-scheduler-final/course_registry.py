"""
Course Registry
---------------
Loads course metadata (tier + enrollment) from a CSV file exported by the
registrar.  Pods that reference an unknown course get inferred metadata based
on the numeric portion of the course code.

Expected CSV columns (order-independent, header required):
    course_id, level, seats
    e.g.:
        CSE234_SP26_A00,graduate,18
        CSE101_SP26_A00,lower,210
        CSE150_SP26_A00,upper,55

Reload at any time by calling CourseRegistry.load_csv(); the registry is
replaced atomically so the controller never sees a partially-loaded state.
"""

from __future__ import annotations

import csv
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scheduler import CourseClass, Tier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback enrollment assumptions when course data is unavailable
# ---------------------------------------------------------------------------
FALLBACK_ENROLLMENT: dict[Tier, int] = {
    Tier.INTRO:     200,
    Tier.UPPER_DIV: 200,
    Tier.GRAD:       50,
}

# Maps registrar-supplied level strings to our Tier enum
_LEVEL_MAP: dict[str, Tier] = {
    "lower":        Tier.INTRO,
    "lower_div":    Tier.INTRO,
    "lower division": Tier.INTRO,
    "intro":        Tier.INTRO,
    "undergraduate": Tier.INTRO,   # ambiguous — treated as intro
    "upper":        Tier.UPPER_DIV,
    "upper_div":    Tier.UPPER_DIV,
    "upper division": Tier.UPPER_DIV,
    "graduate":     Tier.GRAD,
    "grad":         Tier.GRAD,
    "phd":          Tier.GRAD,
}

# Regex to extract the leading numeric portion from a course code segment
# e.g. "CSE234_SP26_A00" → 234,  "MATH20C_FA25_B01" → 20
_COURSE_NUM_RE = re.compile(r'[A-Za-z]+(\d+)', re.IGNORECASE)


def _infer_tier(course_id: str) -> Tier:
    """Derive tier from the numeric portion of the course code."""
    m = _COURSE_NUM_RE.search(course_id)
    if not m:
        logger.warning("Cannot infer tier for course %r — defaulting to INTRO", course_id)
        return Tier.INTRO
    number = int(m.group(1))
    if number < 100:
        return Tier.INTRO
    elif number < 200:
        return Tier.UPPER_DIV
    else:
        return Tier.GRAD


def _parse_level(raw: str, course_id: str) -> Tier:
    key = raw.strip().lower()
    tier = _LEVEL_MAP.get(key)
    if tier is None:
        logger.warning("Unrecognised level %r for course %s — inferring from code",
                       raw, course_id)
        tier = _infer_tier(course_id)
    return tier


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class CourseRegistry:
    """
    Thread-safe store of CourseClass objects.

    Usage:
        registry = CourseRegistry()
        registry.load_csv(Path("/etc/dsmlp/courses.csv"))
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
        Raises FileNotFoundError or csv.Error on bad input; the existing registry
        is left intact on any failure.
        """
        new_courses: dict[str, CourseClass] = {}

        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            required = {"course_id", "level", "seats"}
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
                    logger.warning("Bad seat count for %s (%r) — skipping", course_id, row["seats"])
                    continue
                tier = _parse_level(row["level"], course_id)
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
        Return the CourseClass for *course_id*.  If unknown, synthesise one
        from the course code and log a warning.  Never raises.
        """
        with self._lock:
            course = self._courses.get(course_id)

        if course is not None:
            return course

        tier = _infer_tier(course_id)
        enrollment = FALLBACK_ENROLLMENT[tier]
        logger.warning(
            "Unknown course %r — inferred tier=%s enrollment=%d",
            course_id, tier.name, enrollment,
        )
        synthetic = CourseClass(class_id=course_id, tier=tier, enrollment=enrollment)

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
