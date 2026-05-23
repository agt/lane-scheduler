"""
Residency Statistics
--------------------
Maintains per-(course, lane, batch-mode) residency distributions, updated
online as pods complete.  Each stratum starts from a cluster-wide prior and
converges toward course-specific observations via Bayesian shrinkage.

Math
~~~~
We track a running mean and variance using an exponentially weighted moving
average (EWMA), so that recent pod residencies carry more weight than older
ones.  Given smoothing factor α ∈ (0, 1):

    mean_new = (1 − α) × mean_old + α × x
             = mean_old + α × (x − mean_old)

    var_new  = (1 − α) × (var_old + α × (x − mean_old)²)

Higher α means faster adaptation to recent data (α = 1 reduces to always
using the latest observation; α → 0 gives equal weight to all history).
The actual observation count n is still tracked so Bayesian shrinkage can
be applied correctly.

The posterior mean and variance blend the prior with observed EWMA data:

    n_eff  = prior_weight + n_obs
    mean   = (prior_weight × prior_mean  + n_obs × ewma_mean)  / n_eff
    var    = (prior_weight × prior_var   + n_obs × ewma_var
              + prior_weight × n_obs × (prior_mean - ewma_mean)² / n_eff)
             / n_eff

prior_weight acts as a pseudo-count: with prior_weight=10, the prior counts
as 10 observations.  The course estimate stays close to the cluster prior
until ~10 real observations have accumulated, then shifts toward course data.

Residency observations
~~~~~~~~~~~~~~~~~~~~~~
    residency_pct = (finish_time - start_time) / active_deadline_seconds

Pods killed at their deadline are treated as 100% residency (residency_pct=1.0).
Only pods with active_deadline_seconds > 0 contribute observations.

Thread safety
~~~~~~~~~~~~~
All public methods are protected by a single RLock.  Updates and reads are
both short-lived and non-blocking.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Optional

from lane_scheduler.estimation.wait_estimator import ResidencyProfile

logger = logging.getLogger(__name__)

# Default prior pseudo-count: trust the prior as strongly as this many real
# observations.  Raise to make courses converge more slowly; lower to make
# them adapt faster.
DEFAULT_PRIOR_WEIGHT = 10.0

# Default EWMA smoothing factor.  Each new observation contributes α of the
# update; lower values retain history longer.  Raise to react faster to
# shifts in course behaviour.
DEFAULT_EWMA_ALPHA = 0.1


# ---------------------------------------------------------------------------
# EWMA accumulator
# ---------------------------------------------------------------------------

@dataclass
class _EWMA:
    """
    Exponentially weighted mean and variance estimator.

    alpha: smoothing factor in (0, 1).  Higher = faster adaptation to recent
           observations (α=0.1 means ~10 % weight on the newest point).
    """
    alpha: float
    n:     int   = 0
    mean:  float = 0.0
    var:   float = 0.0  # exponentially weighted variance

    def update(self, x: float) -> None:
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            diff      = x - self.mean
            self.mean += self.alpha * diff
            self.var   = (1.0 - self.alpha) * (self.var + self.alpha * diff ** 2)

    @property
    def variance(self) -> float:
        """Exponentially weighted variance; 0 until at least two observations."""
        return self.var if self.n >= 2 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


# ---------------------------------------------------------------------------
# Stratum key
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _StratumKey:
    course_id: str
    lane_name: str   # human-readable, e.g. "gpu-medium"
    batch:     bool


# ---------------------------------------------------------------------------
# ResidencyStats
# ---------------------------------------------------------------------------

class ResidencyStats:
    """
    Per-course residency distribution tracker with Bayesian shrinkage
    toward cluster-wide priors.

    Usage
    -----
        stats = ResidencyStats(
            interactive_prior = ResidencyProfile(mean_pct=0.4, std_pct=0.2),
            batch_prior       = ResidencyProfile(mean_pct=0.7, std_pct=0.15),
            prior_weight      = 10.0,
            ewma_alpha        = 0.1,
        )

        # On pod completion (from controller._handle_pod_event):
        stats.record(
            course_id    = "CSE234_SP26_A00",
            lane_name    = "gpu-medium",
            batch        = False,
            residency_pct = 0.63,
        )

        # In _build_wait_snapshot, per queued pod:
        profile = stats.profile_for(
            course_id = "CSE234_SP26_A00",
            lane_name = "gpu-medium",
            batch     = False,
        )
        # → ResidencyProfile blending prior + course observations
    """

    def __init__(
        self,
        interactive_prior: ResidencyProfile,
        batch_prior:       ResidencyProfile,
        prior_weight:      float = DEFAULT_PRIOR_WEIGHT,
        ewma_alpha:        float = DEFAULT_EWMA_ALPHA,
    ) -> None:
        self._interactive_prior = interactive_prior
        self._batch_prior       = batch_prior
        self._prior_weight      = prior_weight
        self._ewma_alpha        = ewma_alpha
        self._strata: dict[_StratumKey, _EWMA] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Recording completions
    # ------------------------------------------------------------------

    def record(
        self,
        course_id:     str,
        lane_name:     str,
        batch:         bool,
        residency_pct: float,
    ) -> None:
        """
        Record one pod completion.

        residency_pct should be in [0, 1], where 1.0 means the pod ran
        to its full deadline.  Values outside [0, 1] are clamped.
        """
        pct = max(0.0, min(1.0, residency_pct))
        key = _StratumKey(course_id=course_id, lane_name=lane_name, batch=batch)
        with self._lock:
            if key not in self._strata:
                self._strata[key] = _EWMA(alpha=self._ewma_alpha)
            self._strata[key].update(pct)

        logger.debug(
            "Residency recorded [course=%s lane=%s batch=%s]: "
            "pct=%.3f  n=%d  mean=%.3f  std=%.3f",
            course_id, lane_name, batch, pct,
            self._strata[key].n,
            self._strata[key].mean,
            self._strata[key].std,
        )

    # ------------------------------------------------------------------
    # Profile retrieval
    # ------------------------------------------------------------------

    def profile_for(
        self,
        course_id: str,
        lane_name: str,
        batch:     bool,
    ) -> ResidencyProfile:
        """
        Return a ResidencyProfile blending the cluster prior with any
        course-specific observations collected so far.

        With no observations the prior is returned unchanged.
        With many observations the profile converges to the course mean/std.
        """
        prior = self._batch_prior if batch else self._interactive_prior
        key   = _StratumKey(course_id=course_id, lane_name=lane_name, batch=batch)

        with self._lock:
            acc = self._strata.get(key)

        if acc is None or acc.n == 0:
            return prior

        return self._posterior(prior, acc)

    def observation_count(self, course_id: str, lane_name: str, batch: bool) -> int:
        """Return the number of observations recorded for this stratum."""
        key = _StratumKey(course_id=course_id, lane_name=lane_name, batch=batch)
        with self._lock:
            acc = self._strata.get(key)
        return acc.n if acc else 0

    def all_profiles(self) -> dict[tuple, ResidencyProfile]:
        """
        Return current posterior profiles for all strata that have at least
        one observation.  Key is (course_id, lane_name, batch).
        Useful for monitoring and logging.
        """
        result = {}
        with self._lock:
            items = list(self._strata.items())
        for key, acc in items:
            if acc.n > 0:
                prior = self._batch_prior if key.batch else self._interactive_prior
                result[(key.course_id, key.lane_name, key.batch)] = \
                    self._posterior(prior, acc)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _posterior(self, prior: ResidencyProfile, acc: _EWMA) -> ResidencyProfile:
        """
        Compute the Bayesian posterior ResidencyProfile.

        posterior_mean = (w × prior_mean + n × sample_mean) / (w + n)

        posterior_var  = (w × prior_var + n × sample_var
                          + w × n × (prior_mean - sample_mean)² / (w + n))
                         / (w + n)

        The second term in posterior_var is the between-group variance
        contribution — it grows when the course mean drifts far from the prior.
        """
        w   = self._prior_weight
        n   = float(acc.n)
        n_eff = w + n

        prior_mean = prior.mean_pct
        prior_var  = prior.std_pct ** 2
        samp_mean  = acc.mean
        samp_var   = acc.variance

        post_mean = (w * prior_mean + n * samp_mean) / n_eff
        post_var  = (
            w * prior_var
            + n * samp_var
            + w * n * (prior_mean - samp_mean) ** 2 / n_eff
        ) / n_eff

        # Clamp to valid ResidencyProfile ranges
        post_mean = max(0.01, min(0.99, post_mean))
        post_std  = max(0.01, math.sqrt(post_var))

        return ResidencyProfile(mean_pct=post_mean, std_pct=post_std)
