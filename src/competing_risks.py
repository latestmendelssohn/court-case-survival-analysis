"""
Competing risks: cause-specific hazards + Fine-Gray subdistribution hazard.

Splits the single 'event' column from preprocessing into three competing
event types: judgment, withdrawal, transfer. Every case is either censored
(still pending) or ends via exactly one of these three causes.
"""
from __future__ import annotations

import pandas as pd

COMPETING_EVENTS = ("judgment", "withdrawal", "transfer")


def fit_cause_specific_hazards(df: pd.DataFrame, covariates: list[str]) -> dict:
    """
    Fit one Cox model per competing event type, treating the other event
    types as censored observations for that model (standard cause-specific
    hazards approach).

    Returns
    -------
    dict[str, CoxPHFitter]
        One fitted model per entry in COMPETING_EVENTS.
    """
    raise NotImplementedError("TODO: for each event type, recode event column "
                               "so only that event = 1, others = 0 (censored), "
                               "then fit CoxPHFitter")


def fit_fine_gray(df: pd.DataFrame, covariates: list[str], event_of_interest: str) -> dict:
    """
    Fit the Fine-Gray model for the cumulative incidence function of
    `event_of_interest`, treating other competing events as per the
    subdistribution hazard definition (they remain "at risk" rather than
    being censored, unlike cause-specific hazards).

    Consider `lifelines`' CIF utilities or a dedicated package
    (e.g. `scikit-survival`'s or R's `cmprsk` via rpy2) depending on what's
    available; document the choice here once implemented.
    """
    raise NotImplementedError("TODO")


def cumulative_incidence_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute non-parametric cumulative incidence curves per event type, per group."""
    raise NotImplementedError("TODO")
