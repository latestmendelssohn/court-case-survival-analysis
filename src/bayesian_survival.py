"""
Bayesian lifetime modeling: Weibull survival model in PyMC.

Fits a Bayesian Weibull model per case_type/state group and returns the
posterior so it can be compared against the classical MLE/Cox estimates
from survival_classical.py.
"""
from __future__ import annotations

import arviz as az
import pandas as pd
import pymc as pm


def fit_bayesian_weibull(
    df: pd.DataFrame,
    group_col: str | None = None,
    draws: int = 2000,
    tune: int = 1000,
) -> az.InferenceData:
    """
    Fit a (optionally hierarchical, if group_col given) Bayesian Weibull
    survival model accounting for right-censoring.

    Model sketch (fill in during implementation):
        shape ~ HalfNormal(sigma=...)
        scale ~ HalfNormal(sigma=...)      # or hierarchical per group_col
        likelihood: Weibull, with censored observations contributing the
                    survival function S(t) rather than the density f(t)
                    (use pm.Censored or a custom logp).

    Returns
    -------
    arviz.InferenceData
        Posterior samples for downstream comparison/plotting.
    """
    raise NotImplementedError("TODO")


def compare_to_classical(bayes_idata: az.InferenceData, classical_estimates: pd.DataFrame) -> pd.DataFrame:
    """
    Side-by-side table: classical point estimate + CI vs. Bayesian posterior
    mean + credible interval, per group. This is the "do they agree?"
    section of the write-up.
    """
    raise NotImplementedError("TODO")
