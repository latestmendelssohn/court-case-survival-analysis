"""
Cleaning and censoring-flag construction for court case survival analysis.

This is the layer that turns raw case rows into a survival-analysis-ready
table with `duration` and `event` columns, which every downstream method
(KM, competing risks, Bayesian, ML survival) consumes identically.
"""
from __future__ import annotations

import pandas as pd


def build_survival_table(
    df: pd.DataFrame,
    window_end: str,
    filing_col: str = "filing_date",
    decision_col: str = "decision_date",
) -> pd.DataFrame:
    """
    Construct duration + event columns.

    duration = decision_date - filing_date, if decided within the window.
    duration = window_end - filing_date, if still pending (right-censored).
    event = 1 if decided within window, else 0 (censored).

    Parameters
    ----------
    df : pd.DataFrame
        Raw case rows with filing/decision date columns.
    window_end : str
        ISO date string marking the end of the observation window
        (e.g. the last date covered by the 2-year data pull).

    Returns
    -------
    pd.DataFrame
        Same rows, with `duration` (in days) and `event` columns added.
    """
    raise NotImplementedError(
        "TODO: compute duration/event per the docstring; handle rows with "
        "missing filing_date by dropping and logging count dropped"
    )


def encode_disposal_type(df: pd.DataFrame, disposal_col: str = "disposal_type") -> pd.DataFrame:
    """
    Normalize disposal_type into the three competing-risk categories used in
    src/competing_risks.py: 'judgment', 'withdrawal', 'transfer'.

    Anything not mapping cleanly should be logged and dropped or bucketed
    into an explicit 'other' category (decide explicitly, do not silently
    coerce).
    """
    raise NotImplementedError("TODO: map raw disposal codes to standard categories")


def clean_case_types(df: pd.DataFrame, case_type_col: str = "case_type") -> pd.DataFrame:
    """Standardize free-text/coded case types into a fixed category set."""
    raise NotImplementedError("TODO: normalize case_type values")
