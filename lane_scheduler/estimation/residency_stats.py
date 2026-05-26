"""
Residency Statistics
--------------------
Maintains per-(sched_group, lane, batch-mode) residency distributions, updated
online as pods complete.  Each stratum starts from a cluster-wide prior and
converges toward scheduling-group-specific observations via Bayesian shrinkage.

Math
~~~~
We track a running mean and variance using an exponentially weighted moving
average (EWMA), so that recent pod residencies carry more weight than older
ones.  Given smoothing factor α ∈ (0, 1):

    mean_new = (1 − α) × mean_old + α × x
             = mean_old + α × (x − mean_old)

    var_new  = (1 − α) × (var_old + α × (x − mean_old)²)

The posterior mean and variance blend the prior with observed EWMA data:

    n_eff  = prior_weight + n_obs
    mean   = (prior_weight × prior_mean  + n_obs × ewma_mean)  / n_eff
    var    = (prior_weight × prior_var   + n_obs × ewma_var
              + prior_weight × n_obs × (prior_mean - ewma_mean)² / n_eff)
             / n_eff

prior_weight acts as a pseudo-count: with prior_weight=10, the prior counts
as 10 observations.  The group estimate stays close to the cluster prior
until ~10 real observations have accumulated, then shifts toward group data.

Residency observations
~~~~~~~~~~~~~~~~~~~~~~
    residency_pct = (finish_time - start_time) / active_deadline_seconds

Thread safety
~~~~~~~~~~~~~
All public methods are protected by a single RLock.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Optional

from lane_scheduler.estimation.wait_estimator import ResidencyProfile

logger = logging.getLogger(__name__)

DEFAULT_PRIOR_WEIGHT = 10.0
DEFAULT_EWMA_ALPHA   = 0.1


# ---------------------------------------------------------------------------
# EWMA accumulator
# ---------------------------------------------------------------------------

@dataclass
class _EWMA:
    alpha: float
    n:     int   = 0
    mean:  float = 0.0
    var:   float = 0.0

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
        return self.var if self.n >= 2 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


# ---------------------------------------------------------------------------
# Stratum key
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _StratumKey:
    sched_group_id: str
    lane_name:      str
    batch:          bool


# ---------------------------------------------------------------------------
# ResidencyStats
# ---------------------------------------------------------------------------

class ResidencyStats:
    """
    Per-scheduling-group residency distribution tracker with Bayesian shrinkage
    toward cluster-wide priors.

    Usage
    -----
        stats = ResidencyStats(
            interactive_prior = ResidencyProfile(mean_pct=0.4, std_pct=0.2),
            batch_prior       = ResidencyProfile(mean_pct=0.7, std_pct=0.15),
        )

        # On pod completion:
        stats.record(
            sched_group_id = "CSE234_SP26_A00",
            lane_name      = "gpu-medium",
            batch          = False,
            residency_pct  = 0.63,
        )

        # Per queued pod in the wait snapshot:
        profile = stats.profile_for(
            sched_group_id = "CSE234_SP26_A00",
            lane_name      = "gpu-medium",
            batch          = False,
        )
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
        sched_group_id: str,
        lane_name:      str,
        batch:          bool,
        residency_pct:  float,
    ) -> None:
        """
        Record one pod completion.

        residency_pct should be in [0, 1], where 1.0 means the pod ran
        to its full deadline.  Values outside [0, 1] are clamped.
        """
        pct = max(0.0, min(1.0, residency_pct))
        key = _StratumKey(sched_group_id=sched_group_id, lane_name=lane_name, batch=batch)
        with self._lock:
            if key not in self._strata:
                self._strata[key] = _EWMA(alpha=self._ewma_alpha)
            self._strata[key].update(pct)

        logger.debug(
            "Residency recorded [group=%s lane=%s batch=%s]: "
            "pct=%.3f  n=%d  mean=%.3f  std=%.3f",
            sched_group_id, lane_name, batch, pct,
            self._strata[key].n,
            self._strata[key].mean,
            self._strata[key].std,
        )

    # ------------------------------------------------------------------
    # Profile retrieval
    # ------------------------------------------------------------------

    def profile_for(
        self,
        sched_group_id: str,
        lane_name:      str,
        batch:          bool,
    ) -> ResidencyProfile:
        """
        Return a ResidencyProfile blending the cluster prior with any
        group-specific observations collected so far.
        """
        prior = self._batch_prior if batch else self._interactive_prior
        key   = _StratumKey(sched_group_id=sched_group_id, lane_name=lane_name, batch=batch)

        with self._lock:
            acc = self._strata.get(key)

        if acc is None or acc.n == 0:
            return prior

        return self._posterior(prior, acc)

    def observation_count(self, sched_group_id: str, lane_name: str, batch: bool) -> int:
        """Return the number of observations recorded for this stratum."""
        key = _StratumKey(sched_group_id=sched_group_id, lane_name=lane_name, batch=batch)
        with self._lock:
            acc = self._strata.get(key)
        return acc.n if acc else 0

    def all_profiles(self) -> dict[tuple, ResidencyProfile]:
        """Return current posterior profiles for all strata with at least one observation."""
        result = {}
        with self._lock:
            items = list(self._strata.items())
        for key, acc in items:
            if acc.n > 0:
                prior = self._batch_prior if key.batch else self._interactive_prior
                result[(key.sched_group_id, key.lane_name, key.batch)] = \
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

        post_mean = max(0.01, min(0.99, post_mean))
        post_std  = max(0.01, math.sqrt(post_var))

        return ResidencyProfile(mean_pct=post_mean, std_pct=post_std)
