"""
Competing-risks event coding and cause-specific hazard models.

The processed data has three named disposal causes plus a large observed
'other' category. `encode_competing_events` preserves that category as code 4,
and `fit_cause_specific_hazards` models it explicitly as the heterogeneous
`other_observed` endpoint. Fine-Gray remains unimplemented.
"""
from __future__ import annotations

import pandas as pd

COMPETING_EVENTS = ("judgment", "withdrawal", "transfer")
COMPETING_EVENT_CODES = {
    "censored": 0,
    "judgment": 1,
    "withdrawal": 2,
    "transfer": 3,
    "other_observed": 4,
}


def encode_competing_events(
    df: pd.DataFrame,
    event_col: str = "event",
    disposal_col: str = "disposal_type",
) -> pd.DataFrame:
    """Add a safe integer code for censored and observed event causes.

    Code 4 preserves observed disposal values outside the three named causes;
    it must not be treated as censoring until those values are resolved.
    """
    required = [event_col, disposal_col]
    missing = [column for column in required if column not in df]
    if missing:
        raise KeyError(f"Missing required competing-risk columns: {missing}")

    result = df.copy()
    event = pd.to_numeric(result[event_col], errors="coerce")
    if event.isna().any() or not event.isin([0, 1]).all():
        raise ValueError(f"{event_col} must contain only 0 and 1")

    labels = result[disposal_col].astype("string").str.strip().str.casefold()
    cause = labels.map(
        {
            "judgment": COMPETING_EVENT_CODES["judgment"],
            "withdrawal": COMPETING_EVENT_CODES["withdrawal"],
            "transfer": COMPETING_EVENT_CODES["transfer"],
        }
    )
    result["competing_event"] = cause.fillna(COMPETING_EVENT_CODES["other_observed"])
    result["competing_event"] = result["competing_event"].where(event.eq(1), 0).astype("int8")
    return result


def fit_cause_specific_hazards(df: pd.DataFrame, covariates: list[str]) -> dict:
    """
    Fit one Cox model per observed cause, including aggregate ``other_observed``.

    For each model, only the target cause is an event; all other causes and
    right-censored rows are treated as censored, which is the standard
    cause-specific hazards construction.
    """
    from src.survival_classical import fit_cox_ph

    prepared = encode_competing_events(df)
    models = {}
    causes = tuple(name for name in COMPETING_EVENT_CODES if name != "censored")
    for cause in causes:
        cause_df = prepared.copy()
        cause_df["event"] = (
            prepared["competing_event"] == COMPETING_EVENT_CODES[cause]
        ).astype("int8")
        if not cause_df["event"].any():
            raise ValueError(f"No observed events for cause: {cause}")
        models[cause] = fit_cox_ph(cause_df, covariates)
    return models


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
    """Return nonparametric cumulative-incidence curves by group and cause."""
    from lifelines import AalenJohansenFitter
    from src.survival_classical import _validated_survival_frame

    result = _validated_survival_frame(df, group_col)
    prepared = encode_competing_events(result)
    curves = []
    causes = tuple(
        (name, code)
        for name, code in COMPETING_EVENT_CODES.items()
        if name != "censored"
    )
    for group, group_df in prepared.groupby(group_col, dropna=True, sort=False):
        for cause, code in causes:
            # Lifelines jitters tied integer-day durations; fixed seed keeps the result reproducible.
            fitter = AalenJohansenFitter(seed=0).fit(
                group_df["duration"],
                group_df["competing_event"],
                event_of_interest=code,
                label=cause,
            )
            curve = fitter.cumulative_density_.reset_index()
            curve.columns = ["duration", "cumulative_incidence"]
            curve[group_col] = group
            curve["cause"] = cause
            curves.append(curve[[group_col, "cause", "duration", "cumulative_incidence"]])

    if not curves:
        raise ValueError(f"{group_col} must contain at least one non-null group")
    return pd.concat(curves, ignore_index=True)
