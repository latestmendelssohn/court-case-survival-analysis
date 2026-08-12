"""
Minimal censoring-aware ML survival helpers built on scikit-survival.

The models are intended for bounded validation and feature-ranking checks, not
as a replacement for the project's classical survival estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


def _validated_inputs(
    X: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a non-empty pandas DataFrame")
    if X.columns.duplicated().any():
        raise ValueError("X columns must be unique")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise TypeError("X must contain only numeric features")
    if len(X) != len(duration) or len(X) != len(event):
        raise ValueError("X, duration, and event must have the same length")

    features = X.to_numpy(dtype=float)
    durations = pd.to_numeric(duration, errors="coerce").to_numpy(dtype=float)
    events = pd.Series(event).to_numpy()
    if not np.isfinite(features).all():
        raise ValueError("X must contain only finite values")
    if not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError("duration must contain finite nonnegative values")
    if not np.isin(events, [False, True, 0, 1]).all():
        raise ValueError("event must contain only 0 or 1")
    return X, durations, events.astype(bool)


def fit_random_survival_forest(
    X: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    **rsf_kwargs,
) -> RandomSurvivalForest:
    """Fit a Random Survival Forest and return the fitted model."""
    X, durations, events = _validated_inputs(X, duration, event)
    model = RandomSurvivalForest(**rsf_kwargs)
    model.fit(X, Surv.from_arrays(events, durations))
    return model


def fit_gradient_boosted_survival(
    X: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    **gbs_kwargs,
) -> GradientBoostingSurvivalAnalysis:
    """Fit a gradient-boosted survival model and return the fitted model."""
    X, durations, events = _validated_inputs(X, duration, event)
    model = GradientBoostingSurvivalAnalysis(**gbs_kwargs)
    model.fit(X, Surv.from_arrays(events, durations))
    return model


def feature_importance_table(model, feature_names: list[str]) -> pd.DataFrame:
    """Return feature importances sorted from largest to smallest."""
    try:
        importances = np.asarray(model.feature_importances_, dtype=float)
    except (AttributeError, NotImplementedError) as exc:
        raise ValueError(
            "the fitted model does not expose feature importances"
        ) from exc
    if importances.ndim != 1 or len(importances) != len(feature_names):
        raise ValueError("feature_names must match the fitted model's importances")
    if not np.isfinite(importances).all():
        raise ValueError("model feature importances must be finite")
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False, ignore_index=True)
    )


def concordance_index(
    model,
    X: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
) -> float:
    """Evaluate model discrimination using Harrell's censoring-aware C-index."""
    X, durations, events = _validated_inputs(X, duration, event)
    risk = np.asarray(model.predict(X), dtype=float)
    if not np.isfinite(risk).all():
        raise ValueError("model predictions must be finite")
    return float(concordance_index_censored(events, durations, risk)[0])
