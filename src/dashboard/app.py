"""
Streamlit dashboard for the court case survival analysis project.

Run with: streamlit run src/dashboard/app.py

Panels (to implement):
  1. District/state backlog overview table (pendency rate, median time-to-decision).
  2. Kaplan-Meier curve viewer, filterable by case_type / state.
  3. Cumulative incidence curve viewer (competing risks), filterable.
  4. Feature importance chart from the ML survival model.
"""
import streamlit as st

st.set_page_config(page_title="Court Case Survival Analysis", layout="wide")

st.title("Court Case Delay & Pendency: Survival Analysis")
st.caption(
    "Kaplan-Meier, competing risks (Fine-Gray), Bayesian Weibull, and "
    "Random Survival Forest applied to Indian district court case data."
)

st.info("TODO: load processed data + fitted models from src/, wire up the panels described above.")

# TODO: st.sidebar filters for state / case_type
# TODO: KM curve plot (plotly or matplotlib)
# TODO: cumulative incidence plot
# TODO: feature importance bar chart
# TODO: backlog table (st.dataframe)
