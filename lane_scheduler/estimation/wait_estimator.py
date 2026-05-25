"""
Wait Time Estimator
-------------------
Estimates how long a queued pod will wait before being dispatched,
given:
  - The distribution of pod residency times (as a fraction of activeDeadlineSeconds)
  - The set of currently running pods in the same lane (with ages and deadlines)
  - The pod's position in the scored queue

Math
~~~~
Residency X ~ Normal(μ, σ) where μ = mean_pct × D, σ = std_pct × D
and D = activeDeadlineSeconds for a given pod.

For a running pod that has already survived `age` seconds, remaining
lifetime R is drawn from a left-truncated normal.  The probability it
finishes within the next t seconds:

    P(R ≤ t | X > age) = [Φ((age+t - μ)/σ) - Φ((age - μ)/σ)]
                          ─────────────────────────────────────
                                    1 - Φ((age - μ)/σ)

where Φ is the standard normal CDF, approximated via math.erf.

Summing finish-probabilities across all running pods in a lane gives
E[slots freed within t].  We invert this function numerically to find
the t at which the expected freed slots equals the pod's queue rank,
giving the median wait estimate.  P20/P80 bounds are derived from the
variance of the slot-count estimate.

All arithmetic uses the stdlib only (math module).
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ResidencyProfile:
    """
    Population-level residency distribution for one workload mode.

    mean_pct : mean residency as a fraction of activeDeadlineSeconds  (0 < x < 1)
    std_pct  : std dev as the same fraction                           (x > 0)
    """
    mean_pct: float
    std_pct:  float

    def __post_init__(self) -> None:
        if not (0 < self.mean_pct < 1):
            raise ValueError(f"mean_pct must be in (0, 1), got {self.mean_pct}")
        if self.std_pct <= 0:
            raise ValueError(f"std_pct must be > 0, got {self.std_pct}")

    def for_deadline(self, deadline_seconds: float) -> tuple[float, float]:
        """Return (μ, σ) in seconds for a pod with the given deadline."""
        return (self.mean_pct * deadline_seconds,
                self.std_pct  * deadline_seconds)


@dataclass
class RunningPod:
    """
    A pod currently occupying a resource slot in a lane.

    pod_uid                : Kubernetes pod UID
    start_time             : wall-clock time the pod started (time.monotonic())
    active_deadline_seconds: pod's activeDeadlineSeconds (from spec)
    batch                  : True if dsmlp/batch=true
    resource_units         : units consumed (same scale as Job.resource_units)
    """
    pod_uid:                 str
    start_time:              float
    active_deadline_seconds: float
    batch:                   bool
    resource_units:          float = 1.0

    def age(self, now: Optional[float] = None) -> float:
        t = now if now is not None else time.monotonic()
        return max(0.0, t - self.start_time)

    def remaining_max(self, now: Optional[float] = None) -> float:
        """Hard upper bound on remaining life (deadline - age)."""
        return max(0.0, self.active_deadline_seconds - self.age(now))


@dataclass
class WaitEstimate:
    """
    Estimated wait time for a queued pod, with uncertainty bounds.

    All values are in seconds.  p20 is optimistic (things clear faster),
    p80 is pessimistic (things clear slower).
    """
    median_seconds: float
    p20_seconds:    float   # 20th percentile — optimistic
    p80_seconds:    float   # 80th percentile — pessimistic
    queue_rank:     int     # pod's position in the scored queue (1 = next up)
    lane_name:      str


# ---------------------------------------------------------------------------
# Truncated-normal helpers (stdlib only)
# ---------------------------------------------------------------------------

def _phi(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _finish_prob(t: float, mu: float, sigma: float, age: float) -> float:
    """
    P(pod finishes within next t seconds | it has survived `age` seconds).

    If the pod has already exceeded its expected lifetime (age > mu + 4σ)
    we treat finish as near-certain within any positive t.
    If sigma is effectively zero, use a deterministic threshold.
    """
    if sigma < 1e-9:
        return 1.0 if age + t >= mu else 0.0

    surv_at_age = 1.0 - _phi((age - mu) / sigma)
    if surv_at_age < 1e-12:
        return 1.0   # pod is almost certainly already done / at deadline

    prob_finish_by_age_plus_t = _phi((age + t - mu) / sigma)
    prob_finish_by_age        = _phi((age       - mu) / sigma)
    return (prob_finish_by_age_plus_t - prob_finish_by_age) / surv_at_age


def _expected_slots(
    t: float,
    running: list[RunningPod],
    profiles: dict[str, ResidencyProfile],
    now: float,
) -> tuple[float, float]:
    """
    Returns (mean, variance) of the number of resource slots expected to
    free up within the next t seconds across all running pods.

    Each pod contributes an independent Bernoulli(p) to the slot count,
    weighted by resource_units.  (We approximate weighted slots as
    independent; covariance terms are small and require joint distributions.)
    """
    mean = 0.0
    var  = 0.0
    for pod in running:
        age      = pod.age(now)
        rem      = pod.remaining_max(now)
        if rem <= 0:
            # Deadline already reached — slot is freed
            p = 1.0
        elif t >= rem:
            # t extends to or past the hard deadline — pod must finish by then
            p = 1.0
        else:
            profile  = profiles["batch" if pod.batch else "interactive"]
            mu, sigma = profile.for_deadline(pod.active_deadline_seconds)
            p = _finish_prob(t, mu, sigma, age)
        w     = pod.resource_units
        mean += p * w
        var  += p * (1.0 - p) * w * w
    return mean, var


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

# Normal quantiles for p20 and p80
_Z_P20 = -0.8416   # Φ⁻¹(0.20)
_Z_P80 =  0.8416   # Φ⁻¹(0.80)

# Search bounds and resolution
_T_MAX_DEFAULT = 24 * 3600.0   # 24 hours — hard ceiling
_BISECT_ITERS  = 48            # bisection iterations (~24h / 2^48 ≈ sub-millisecond)


def _bisect_wait(
    target_slots: float,
    running: list[RunningPod],
    profiles: dict[str, ResidencyProfile],
    now: float,
    z_adjust: float = 0.0,
    t_max: float = _T_MAX_DEFAULT,
) -> float:
    """
    Find t such that E[slots freed by t] + z_adjust × std(slots) ≈ target_slots.

    z_adjust = 0   → median estimate
    z_adjust = -0.8416 → p20 (optimistic: slots free sooner)
    z_adjust = +0.8416 → p80 (pessimistic)

    Uses bisection on a monotone function.
    """
    def f(t: float) -> float:
        mean, var = _expected_slots(t, running, profiles, now)
        std = math.sqrt(var) if var > 0 else 0.0
        return mean + z_adjust * std - target_slots

    # Check if enough slots ever free up within t_max
    f_max = f(t_max)
    if f_max < 0:
        # Even by the deadline ceiling we don't expect enough slots
        logger.debug(
            "_bisect_wait saturated at t_max=%.0fs "
            "(target_slots=%.2f z_adjust=%.4f)",
            t_max, target_slots, z_adjust,
        )
        return t_max

    # Check if target already met at t=0 (queue rank <= already-freeing pods)
    if f(0.0) >= 0:
        return 0.0

    lo, hi = 0.0, t_max
    for _ in range(_BISECT_ITERS):
        mid = (lo + hi) / 2.0
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def estimate_wait(
    queue_rank:       int,
    lane_name:        str,
    running:          list[RunningPod],
    profiles:         dict[str, ResidencyProfile],
    required_units:   float = 1.0,
    now:              Optional[float] = None,
    t_max:            float = _T_MAX_DEFAULT,
) -> WaitEstimate:
    """
    Estimate how long a pod at position `queue_rank` (1 = next to dispatch)
    will wait before enough slots free up to accommodate it.

    Parameters
    ----------
    queue_rank      : 1-based position in the scored lane queue
    lane_name       : human-readable lane name (for the returned struct)
    running         : currently running pods in this lane
    profiles        : {"interactive": ResidencyProfile, "batch": ResidencyProfile}
    required_units  : resource_units the queued pod needs (default 1)
    now             : current time (time.monotonic()); defaults to now
    t_max           : search ceiling in seconds (default 24 h)

    Returns
    -------
    WaitEstimate with median, p20, and p80 in seconds.
    """
    if now is None:
        now = time.monotonic()

    # Target: number of resource-unit-slots that must free up before this pod.
    # A pod at rank 1 needs nothing ahead of it to clear — it's next.
    # A pod at rank N needs (N-1) × required_units of running pods to finish first.
    target = float(queue_rank - 1) * required_units

    median = _bisect_wait(target, running, profiles, now, z_adjust=0.0,     t_max=t_max)
    p20    = _bisect_wait(target, running, profiles, now, z_adjust=_Z_P20,  t_max=t_max)
    p80    = _bisect_wait(target, running, profiles, now, z_adjust=_Z_P80,  t_max=t_max)

    # p20 should be ≤ median ≤ p80; clamp in case of numerical noise
    p20 = min(p20, median)
    p80 = max(p80, median)

    return WaitEstimate(
        median_seconds = median,
        p20_seconds    = p20,
        p80_seconds    = p80,
        queue_rank     = queue_rank,
        lane_name      = lane_name,
    )


# ---------------------------------------------------------------------------
# Background cache
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    estimate:     WaitEstimate
    computed_at:  float          # time.monotonic() when computed


class WaitTimeCache:
    """
    Computes WaitEstimates for all queued pods on a background thread at a
    fixed cadence, and serves cached results to callers with no blocking.

    Usage
    -----
        cache = WaitTimeCache(
            snapshot_fn  = controller.build_snapshot,
            interval     = 60.0,
        )
        cache.start()
        ...
        est = cache.get("pod-uid-abc")   # None until first snapshot completes
        cache.stop()

    snapshot_fn
    -----------
    A zero-argument callable that returns:
        dict[str, WaitEstimate]   — {pod_uid: WaitEstimate}

    The controller builds this by iterating its pending queues and calling
    estimate_wait() for each pod.  The callable is invoked on the background
    thread, so it must be thread-safe (read-only access to shared state is fine
    if protected by locks).

    Stale entries
    -------------
    Entries for pods that have left the queue are removed each snapshot cycle,
    so the cache never serves estimates for already-dispatched pods.
    """

    def __init__(
        self,
        snapshot_fn: "Callable[[], dict[str, WaitEstimate]]",
        interval:    float = 60.0,
    ) -> None:
        import threading
        self._snapshot_fn  = snapshot_fn
        self._interval     = interval
        self._cache:  dict[str, CacheEntry] = {}
        self._lock    = threading.RLock()
        self._stop    = threading.Event()
        self._thread  = threading.Thread(
            target = self._loop,
            name   = "wait-cache",
            daemon = True,
        )
        self._last_duration: float = 0.0   # seconds the last snapshot took

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._thread.start()
        logger.info("WaitTimeCache started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Public read API — never blocks
    # ------------------------------------------------------------------

    def get(self, pod_uid: str) -> Optional[WaitEstimate]:
        """Return the most recent WaitEstimate for pod_uid, or None."""
        with self._lock:
            entry = self._cache.get(pod_uid)
        return entry.estimate if entry is not None else None

    def get_with_age(self, pod_uid: str) -> Optional[tuple[WaitEstimate, float]]:
        """Return (estimate, age_seconds) or None."""
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(pod_uid)
        if entry is None:
            return None
        return entry.estimate, now - entry.computed_at

    def snapshot_age(self) -> Optional[float]:
        """Seconds since the last snapshot completed, or None if never run."""
        with self._lock:
            if not self._cache:
                return None
            oldest = max(e.computed_at for e in self._cache.values())
        return time.monotonic() - oldest

    @property
    def last_duration(self) -> float:
        """Wall-clock seconds the most recent snapshot computation took."""
        return self._last_duration

    def all_estimates(self) -> dict[str, "WaitEstimate"]:
        """Return a snapshot copy of all current estimates (pod_uid → WaitEstimate)."""
        with self._lock:
            return {uid: e.estimate for uid, e in self._cache.items()}

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        # Slight initial delay so the controller's watches can populate state
        self._stop.wait(min(self._interval, 15.0))

        while not self._stop.is_set():
            self._run_snapshot()
            self._stop.wait(self._interval)

    def _run_snapshot(self) -> None:
        t0 = time.monotonic()
        try:
            estimates = self._snapshot_fn()
        except Exception as exc:
            logger.error("WaitTimeCache snapshot failed: %s", exc, exc_info=True)
            return

        elapsed = time.monotonic() - t0
        self._last_duration = elapsed

        ts = time.monotonic()
        new_cache: dict[str, CacheEntry] = {
            uid: CacheEntry(estimate=est, computed_at=ts)
            for uid, est in estimates.items()
        }

        with self._lock:
            self._cache = new_cache

        logger.info(
            "WaitTimeCache: %d estimates refreshed in %.1fs",
            len(new_cache), elapsed,
        )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_wait(est: WaitEstimate) -> str:
    """Human-readable one-liner for logs, kubectl describe annotations, etc."""
    def _fmt(s: float) -> str:
        if s < 60:
            return f"{s:.0f}s"
        if s < 3600:
            return f"{s/60:.1f}m"
        return f"{s/3600:.1f}h"

    if est.median_seconds == 0.0:
        return f"[{est.lane_name}] rank={est.queue_rank} → dispatch imminent"

    return (
        f"[{est.lane_name}] rank={est.queue_rank} "
        f"→ est. wait {_fmt(est.median_seconds)} "
        f"(optimistic {_fmt(est.p20_seconds)}, "
        f"pessimistic {_fmt(est.p80_seconds)})"
    )
