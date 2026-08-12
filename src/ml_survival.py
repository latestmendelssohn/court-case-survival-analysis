"""
ML-based survival modeling: Random Survival Forest / Gradient Boosted
Survival, via scikit-survival.

Used as a non-parametric check on the Cox PH assumption and to rank feature
importance for what actually predicts disposal delay.
"""
from __future__ import annotations

import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest


def fit_random_survival_forest(
    X: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    **rsf_kwargs,
) -> RandomSurvivalForest:
    """Fit a Random Survival Forest; return the fitted model."""
    raise NotImplementedError("TODO: build structured y array via "
                               "sksurv.util.Surv.from_arrays(event, duration)")


def fit_gradient_boosted_survival(
    X: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    **gbs_kwargs,
) -> GradientBoostingSurvivalAnalysis:
    raise NotImplementedError("TODO")


def feature_importance_table(model, feature_names: list[str]) -> pd.DataFrame:
    """Return a tidy DataFrame of feature importances, sorted descending."""
    raise NotImplementedError("TODO")


def concordance_index(model, X: pd.DataFrame, duration: pd.Series, event: pd.Series) -> float:
    """Evaluate model discrimination via Harrell's C-index."""
    raise NotImplementedError("TODO")
