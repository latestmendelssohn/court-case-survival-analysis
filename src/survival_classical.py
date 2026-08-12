"""
Classical survival analysis: Kaplan-Meier, Nelson-Aalen, log-rank, Cox PH.

Uses `lifelines`. Consumes the duration/event table produced by
preprocessing.build_survival_table.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test


def _validated_survival_frame(
    df: pd.DataFrame, group_col: str | None = None
) -> pd.DataFrame:
    """Return a copy with validated duration/event columns."""
    required = ["duration", "event"] + ([group_col] if group_col else [])
    missing = [column for column in required if column not in df]
    if missing:
        raise KeyError(f"Missing required survival columns: {missing}")

    result = df.copy()
    result["duration"] = pd.to_numeric(result["duration"], errors="coerce")
    result["event"] = pd.to_numeric(result["event"], errors="coerce")
    if result.empty:
        raise ValueError("Survival data must contain at least one row")
    if result["duration"].isna().any() or (result["duration"] < 0).any():
        raise ValueError("duration must contain finite nonnegative values")
    if result["event"].isna().any() or not result["event"].isin([0, 1]).all():
        raise ValueError("event must contain only 0 and 1")
    result["event"] = result["event"].astype("int8")
    return result


def fit_kaplan_meier(
    df: pd.DataFrame, group_col: str | None = None
) -> KaplanMeierFitter | dict[object, KaplanMeierFitter]:
    """
    Fit KM curve(s). If group_col is given, fit one curve per group and
    return a dict of {group_value: KaplanMeierFitter}. Otherwise return a
    single fitted KaplanMeierFitter.
    """
    result = _validated_survival_frame(df, group_col)
    if group_col is None:
        return KaplanMeierFitter().fit(result["duration"], result["event"])

    fitters = {}
    for group, group_df in result.groupby(group_col, dropna=True, sort=False):
        fitters[group] = KaplanMeierFitter().fit(
            group_df["duration"], group_df["event"], label=str(group)
        )
    if not fitters:
        raise ValueError(f"{group_col} must contain at least one non-null group")
    return fitters


def fit_nelson_aalen(df: pd.DataFrame) -> NelsonAalenFitter:
    """Fit cumulative hazard estimator on the full (or filtered) cohort."""
    result = _validated_survival_frame(df)
    return NelsonAalenFitter().fit(result["duration"], result["event"])


def pairwise_logrank(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Run pairwise log-rank tests between every pair of group_col values.
    Returns a tidy DataFrame: group_a, group_b, statistic, p_value.
    """
    result = _validated_survival_frame(df, group_col)
    groups = [
        (group, group_df)
        for group, group_df in result.groupby(group_col, dropna=True, sort=False)
    ]
    if len(groups) < 2:
        raise ValueError(f"{group_col} must contain at least two non-null groups")

    comparisons = []
    for (group_a, data_a), (group_b, data_b) in combinations(groups, 2):
        test = logrank_test(
            data_a["duration"],
            data_b["duration"],
            event_observed_A=data_a["event"],
            event_observed_B=data_b["event"],
        )
        comparisons.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "statistic": float(test.test_statistic),
                "p_value": float(test.p_value),
            }
        )
    return pd.DataFrame(comparisons)


def fit_cox_ph(df: pd.DataFrame, covariates: list[str]) -> CoxPHFitter:
    """
    Fit Cox proportional hazards model on numeric covariates.
    Caller is responsible for encoding categorical variables and checking the
    PH assumption afterward via `model.check_assumptions(df)`.
    """
    if not covariates:
        raise ValueError("covariates must contain at least one column")
    if len(covariates) != len(set(covariates)):
        raise ValueError("covariates must not contain duplicates")
    if any(column in {"duration", "event"} for column in covariates):
        raise ValueError("duration and event are not covariates")

    result = _validated_survival_frame(df)
    missing = [column for column in covariates if column not in result]
    if missing:
        raise KeyError(f"Missing Cox covariate columns: {missing}")
    for column in covariates:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    covariate_frame = result[covariates]
    if covariate_frame.isna().any().any() or covariate_frame.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Cox covariates must be finite and non-null numeric values")

    return CoxPHFitter().fit(
        result[["duration", "event", *covariates]],
        duration_col="duration",
        event_col="event",
    )
