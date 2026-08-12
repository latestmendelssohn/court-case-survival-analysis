# Court Case Survival & Competing-Risks Analysis

Applying survival analysis, competing-risks modeling, and Bayesian inference to
Indian district court case data to model case-disposal timelines and identify
high-backlog courts.

## Data

- **Source:** [Dev Data Lab Judicial Data](https://devdatalab.org/judicial-data) — anonymized
  case records scraped from India's eCourts platform.
- **Scope for this project:** 2 recent years (2017–2018) across 3–5 states, to keep the
  dataset tractable on a single machine.
- **Key fields used:** filing date, decision date / pending flag, court, district, state,
  case type, disposal type (judgment / withdrawal / transfer), party gender, judge ID.

## Why survival analysis

A case still open in the system is a **right-censored** observation — it has "survived"
at least until the observation cutoff, but its eventual disposal time is unknown. This is
the same statistical structure as time-to-event data in reliability/lifetime modeling.
Instead of dropping pending cases or treating "pending" as a binary label, this project
models the full time-to-disposal distribution, accounting for censoring properly.

## Method stack (applied as layers on the same dataset)

1. **Classical survival analysis** — Kaplan-Meier estimator for time-to-disposal by case
   type/state, log-rank tests for group differences, Nelson-Aalen cumulative hazard.
2. **Competing risks** — cases end via judgment, withdrawal, or transfer. Modeled jointly
   with cause-specific hazards and the Fine-Gray subdistribution hazard model for the
   cumulative incidence function, instead of collapsing to a single "disposed" event.
3. **Bayesian lifetime modeling** — Bayesian Weibull model (PyMC) per case type/state,
   compared against the classical MLE/Cox estimates from step 1 (do the credible
   intervals agree with the frequentist confidence intervals?).
4. **ML-based survival modeling** — Random Survival Forest / Gradient Boosted Survival
   model predicting time-to-disposal from case features, used as a non-parametric check
   on the proportional-hazards assumption and to rank feature importance (what actually
   predicts delay).
5. **Dashboard** — Streamlit app surfacing KM curves, cumulative incidence curves, hazard
   ratios, and feature importances, filterable by district/state.

## Project structure

```
court-case-survival-analysis/
├── data/
│   ├── raw/              # untouched downloaded CSVs (gitignored)
│   └── processed/        # cleaned parquet files (gitignored)
├── notebooks/
│   ├── 01_data_ingestion_cleaning.ipynb
│   ├── 02_kaplan_meier_logrank.ipynb
│   ├── 03_competing_risks_fine_gray.ipynb
│   ├── 04_bayesian_weibull.ipynb
│   └── 05_survival_forest.ipynb
├── src/
│   ├── data_loader.py         # download/load raw case data (DuckDB/Polars)
│   ├── preprocessing.py       # cleaning, censoring flags, feature engineering
│   ├── survival_classical.py  # KM, Nelson-Aalen, log-rank, Cox PH
│   ├── competing_risks.py     # cause-specific hazards, Fine-Gray
│   ├── bayesian_survival.py   # PyMC Weibull model
│   ├── ml_survival.py         # Random Survival Forest / GBS
│   └── dashboard/
│       └── app.py             # Streamlit dashboard
├── reports/
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## Plan of action (4 weeks)

### Week 1 — Data & cleaning
- Download 2017–2018 case-level CSVs for 3–5 states from Dev Data Lab.
- Load with DuckDB/Polars (avoid full in-memory pandas load).
- Build the censoring indicator: `event = 1` if decided within window, `event = 0`
  (censored) if still pending at window end.
- Compute `duration = decision_date - filing_date` (or `window_end - filing_date` for
  censored cases).
- Clean case-type and disposal-type categories, handle missing judge/court fields.
- Output: `data/processed/cases_clean.parquet`.

### Week 2 — Classical survival + competing risks
- Kaplan-Meier curves for time-to-disposal, stratified by case type and state.
- Log-rank tests for pairwise group comparisons.
- Cox proportional hazards model; check PH assumption (Schoenfeld residuals).
- Recode disposal_type into judgment / withdrawal / transfer as competing events.
- Fit cause-specific hazards models and Fine-Gray model for cumulative incidence.
- Output: comparison table of hazard ratios per case type/state.

### Week 3 — Bayesian layer + ML layer
- Bayesian Weibull survival model in PyMC with weakly informative priors on
  shape/scale, grouped by case type/state (hierarchical if time permits).
- Compare posterior credible intervals against Week 2's classical estimates.
- Random Survival Forest (scikit-survival) using case type, court, filing season,
  state as features; extract feature importances; compare against Cox coefficients.
- Output: model comparison notebook + write-up section on classical vs. Bayesian
  vs. ML agreement/disagreement.

### Week 4 — Dashboard + write-up
- Streamlit dashboard: district-level backlog map/table, KM curve viewer, cumulative
  incidence viewer, filterable by state/case type.
- Write README results section with headline findings (which states/case types have
  the worst backlog, what drives delay).
- Push final commit, tag `v1.0`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```
