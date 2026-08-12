"""
Cleaning and censoring-flag construction for court case survival analysis.

This is the layer that turns raw case rows into a survival-analysis-ready
table with `duration` and `event` columns, which every downstream method
(KM, competing risks, Bayesian, ML survival) consumes identically.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


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

    Rows with missing filing dates or dates that produce a negative duration are
    dropped and counted in the module logger.
    """
    if filing_col not in df or decision_col not in df:
        missing = [column for column in (filing_col, decision_col) if column not in df]
        raise KeyError(f"Missing required date columns: {missing}")

    end = pd.to_datetime(window_end, errors="coerce")
    if pd.isna(end):
        raise ValueError(f"Invalid window_end: {window_end!r}")

    result = df.copy()
    filing = pd.to_datetime(result[filing_col], errors="coerce")
    decision = pd.to_datetime(result[decision_col], errors="coerce")

    valid_filing = filing.notna()
    missing_filing = int((~valid_filing).sum())
    if missing_filing:
        logger.warning("Dropped %d rows with missing filing dates", missing_filing)

    result = result.loc[valid_filing].copy()
    filing = filing.loc[valid_filing]
    decision = decision.loc[valid_filing]

    decided_in_window = decision.notna() & decision.le(end)
    observed_end = decision.where(decided_in_window, end)
    valid_duration = filing.le(end) & observed_end.ge(filing)
    invalid_duration = int((~valid_duration).sum())
    if invalid_duration:
        logger.warning("Dropped %d rows with invalid date ordering", invalid_duration)

    result = result.loc[valid_duration].copy()
    filing = filing.loc[valid_duration]
    decision = decision.loc[valid_duration]
    observed_end = observed_end.loc[valid_duration]
    decided_in_window = decided_in_window.loc[valid_duration]

    result[filing_col] = filing
    result[decision_col] = decision
    result["duration"] = observed_end.sub(filing).dt.days.astype("int64")
    result["event"] = decided_in_window.astype("int8")
    return result.reset_index(drop=True)


def _normalized_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def encode_disposal_type(df: pd.DataFrame, disposal_col: str = "disposal_type") -> pd.DataFrame:
    """
    Normalize disposal_type into judgment, withdrawal, transfer, or other.

    Unknown and missing values use the explicit ``other`` bucket and are
    logged rather than silently coerced into a competing event.
    """
    if disposal_col not in df:
        raise KeyError(f"Missing disposal column: {disposal_col}")

    def category(value: object) -> str:
        text = _normalized_text(value)
        if any(token in text for token in ("judg", "decree", "merit")):
            return "judgment"
        if any(token in text for token in ("withdraw", "compromise", "settle")):
            return "withdrawal"
        if "transfer" in text:
            return "transfer"
        return "other"

    result = df.copy()
    encoded = result[disposal_col].map(category)
    other_count = int(encoded.eq("other").sum())
    if other_count:
        logger.warning("Mapped %d disposal values to 'other'", other_count)
    result[disposal_col] = encoded
    return result


def clean_case_types(df: pd.DataFrame, case_type_col: str = "case_type") -> pd.DataFrame:
    """Map common case-type labels to stable broad categories."""
    if case_type_col not in df:
        raise KeyError(f"Missing case-type column: {case_type_col}")

    def category(value: object) -> str:
        text = _normalized_text(value)
        if any(token in text for token in ("criminal", "crim")):
            return "criminal"
        if "civil" in text:
            return "civil"
        if any(token in text for token in ("family", "matrimonial", "divorce")):
            return "family"
        if "commercial" in text:
            return "commercial"
        if any(token in text for token in ("tax", "revenue")):
            return "tax"
        if any(token in text for token in ("labour", "labor", "industrial")):
            return "labour"
        if any(token in text for token in ("writ", "constitutional")):
            return "writ"
        return "other"

    result = df.copy()
    cleaned = result[case_type_col].map(category)
    other_count = int(cleaned.eq("other").sum())
    if other_count:
        logger.warning("Mapped %d case types to 'other'", other_count)
    result[case_type_col] = cleaned
    return result
