"""
Bayesian lifetime modeling: Weibull survival model in PyMC.

Fits a Bayesian Weibull model per case_type/state group and returns the
posterior so it can be compared against the classical MLE/Cox estimates
from survival_classical.py.
"""
from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.survival_classical import _validated_survival_frame


def fit_bayesian_weibull(
    df: pd.DataFrame,
    group_col: str | None = None,
    draws: int = 2000,
    tune: int = 1000,
    random_seed: int = 42,
) -> az.InferenceData:
    """Fit a non-hierarchical right-censored Bayesian Weibull model.

    ``event=1`` contributes the Weibull log-density. ``event=0`` contributes
    the log survival function at its observed censoring time. Grouped or
    hierarchical fitting is deliberately not implemented yet.
    """
    if group_col is not None:
        raise NotImplementedError(
            "grouped or hierarchical Bayesian Weibull fitting is not implemented"
        )
    if not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if not isinstance(tune, int) or tune < 0:
        raise ValueError("tune must be a nonnegative integer")

    result = _validated_survival_frame(df)
    durations = result["duration"].to_numpy(dtype="float64")
    events = result["event"].to_numpy(dtype="int8")
    epsilon = np.finfo("float64").eps
    durations = np.maximum(durations, epsilon)
    scale_guess = max(float(np.median(durations)), 1.0)

    with pm.Model() as model:
        shape = pm.LogNormal("shape", mu=0.0, sigma=1.0)
        scale = pm.LogNormal("scale", mu=np.log(scale_guess), sigma=1.0)
        observed_duration = pm.Data("duration", durations, mutable=False)
        observed_event = pm.Data("event", events, mutable=False)

        weibull = pm.Weibull.dist(alpha=shape, beta=scale)
        log_density = pm.logp(weibull, observed_duration)
        log_survival = -pt.pow(observed_duration / scale, shape)
        pm.Potential(
            "survival_likelihood",
            pt.sum(pt.switch(pt.eq(observed_event, 1), log_density, log_survival)),
        )
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=2,
            cores=1,
            target_accept=0.9,
            random_seed=random_seed,
            progressbar=False,
            return_inferencedata=True,
        )


def compare_to_classical(bayes_idata: az.InferenceData, classical_estimates: pd.DataFrame) -> pd.DataFrame:
    """
    Side-by-side table: classical point estimate + CI vs. Bayesian posterior
    mean + credible interval, per group. This is the "do they agree?"
    section of the write-up.
    """
    raise NotImplementedError("TODO")
