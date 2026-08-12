"""
Classical survival analysis: Kaplan-Meier, Nelson-Aalen, log-rank, Cox PH.

Uses `lifelines`. Consumes the duration/event table produced by
preprocessing.build_survival_table.
"""
from __future__ import annotations

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test


def fit_kaplan_meier(df: pd.DataFrame, group_col: str | None = None):
    """
    Fit KM curve(s). If group_col is given, fit one curve per group and
    return a dict of {group_value: KaplanMeierFitter}. Otherwise return a
    single fitted KaplanMeierFitter.
    """
    raise NotImplementedError("TODO")


def fit_nelson_aalen(df: pd.DataFrame) -> NelsonAalenFitter:
    """Fit cumulative hazard estimator on the full (or filtered) cohort."""
    raise NotImplementedError("TODO")


def pairwise_logrank(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Run pairwise log-rank tests between every pair of group_col values.
    Returns a tidy DataFrame: group_a, group_b, statistic, p_value.
    """
    raise NotImplementedError("TODO")


def fit_cox_ph(df: pd.DataFrame, covariates: list[str]) -> CoxPHFitter:
    """
    Fit Cox proportional hazards model on the given covariates.
    Caller is responsible for checking the PH assumption afterward via
    `model.check_assumptions(df)`.
    """
    raise NotImplementedError("TODO")
