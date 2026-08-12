"""Minimal Streamlit viewer for bounded court-case survival results."""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.competing_risks import cumulative_incidence_by_group
from src.survival_classical import fit_kaplan_meier

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cases_clean.parquet"
STATE_CODES = ["01", "03", "08", "10", "13"]
YEARS = [2017, 2018]


@st.cache_data(show_spinner=False)
def load_sample(path: str, state: str, year: int, limit: int) -> pd.DataFrame:
    """Load only the selected bounded cohort from parquet."""
    query = """
    SELECT duration, event, disposal_type, state, year
    FROM read_parquet(?)
    WHERE state = ? AND year = ?
    ORDER BY ddl_case_id
    LIMIT ?
    """
    with duckdb.connect() as con:
        return con.execute(query, [path, state, year, limit]).fetchdf()


def main() -> None:
    st.set_page_config(page_title="Court Case Survival Analysis", layout="wide")
    st.title("Court Case Delay & Pendency")
    st.caption("Bounded Kaplan-Meier and competing-risks views from the processed case data.")

    if not PROCESSED_PATH.exists():
        st.error(f"Processed parquet not found: {PROCESSED_PATH}")
        st.stop()

    st.sidebar.header("Cohort filters")
    state = st.sidebar.selectbox("State code", STATE_CODES)
    year = st.sidebar.selectbox("Filing year", YEARS, index=1)
    limit = st.sidebar.slider("Rows to load", min_value=500, max_value=5000, value=2500, step=500)
    sample = load_sample(str(PROCESSED_PATH), state, year, limit)
    if sample.empty:
        st.warning("No rows matched the selected filters.")
        st.stop()

    events = int(sample["event"].sum())
    censored = int((sample["event"] == 0).sum())
    first, second, third = st.columns(3)
    first.metric("Rows loaded", f"{len(sample):,}")
    second.metric("Observed events", f"{events:,}")
    third.metric("Censored", f"{censored:,}")

    km = fit_kaplan_meier(sample)
    survival = km.survival_function_.reset_index()
    survival.columns = ["duration", "survival_probability"]
    st.subheader("Kaplan-Meier survival curve")
    st.plotly_chart(
        px.line(
            survival,
            x="duration",
            y="survival_probability",
            labels={"duration": "Days", "survival_probability": "Probability case remains open"},
        ),
        use_container_width=True,
    )

    st.subheader("Cumulative incidence by observed cause")
    incidence_input = sample.assign(selected_cohort="selected")
    incidence = cumulative_incidence_by_group(incidence_input, "selected_cohort")
    st.plotly_chart(
        px.line(
            incidence,
            x="duration",
            y="cumulative_incidence",
            color="cause",
            labels={"duration": "Days", "cumulative_incidence": "Cumulative incidence"},
        ),
        use_container_width=True,
    )
    st.caption(
        "The other_observed curve aggregates heterogeneous observed disposal labels and is not one specific mechanism. "
        "All displayed values are bounded cohort results, not full-cohort estimates."
    )


if __name__ == "__main__":
    main()
