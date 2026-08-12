# Court Case Survival & Competing-Risks Analysis

Survival-analysis examples for Indian district court case data, with right censoring,
competing disposal outcomes, a bounded Bayesian Weibull model, and censoring-aware
machine-learning models.

## Data

- **Source:** [Dev Data Lab Judicial Data](https://devdatalab.org/judicial-data) — anonymized
  case records scraped from India's eCourts platform.
- **Scope:** 2017–2018 for state codes `01`, `03`, `08`, `10`, and `13`.
- **Processed output:** `data/processed/cases_clean.parquet` (gitignored).
- **Key fields:** filing and decision dates, pending/event flag, court, district, state,
  case type, disposal type, party gender, and judge identifiers.

## Why survival analysis

A case still open in the system is a **right-censored** observation: it has survived at
least until the observation cutoff, but its eventual disposal time is unknown. This is
the same structure as time-to-event data in reliability and lifetime modeling. The
project keeps pending cases in the analysis instead of treating them as ordinary
non-events.

## Method stack

1. **Classical survival analysis** — Kaplan-Meier, pairwise log-rank tests, Nelson-Aalen
   cumulative hazard, and Cox proportional-hazards models. The Cox diagnostic is run on
   bounded samples and is not treated as a full-cohort conclusion.
2. **Competing risks** — cause-specific hazards and cumulative-incidence curves for
   judgment, withdrawal, transfer, and the heterogeneous `other_observed` category.
   Fine-Gray fitting remains deferred because no supported estimator is installed.
3. **Bayesian lifetime modeling** — a non-hierarchical right-censored Weibull model in
   PyMC. Grouped/hierarchical fitting and a generic classical-comparison helper remain
   deferred.
4. **ML survival modeling** — Random Survival Forest and Gradient Boosted Survival models
   through scikit-survival, evaluated with a censoring-aware C-index on bounded data.
   Gradient Boosting supplies feature importances because scikit-survival does not expose
   them for its Random Survival Forest implementation.
5. **Dashboard** — a bounded Streamlit viewer with state/year filters, summary metrics,
   Kaplan-Meier curves, and cumulative-incidence curves.

## Current status and bounded checks

The processed parquet contains 12,157,119 rows: 6,281,457 observed events and
5,875,662 censored cases. Validation has passed for the ingestion, classical,
competing-risks, Bayesian, ML, and dashboard paths.

The Bayesian convergence check used 250 rows from state `01` in 2018. With two chains,
300 draws, and 300 tuning steps, both parameters had `r_hat = 1.0`, there were no
divergences, and posterior values were finite. The posterior Weibull median was
105.65 days versus a same-sample Kaplan-Meier median of 66.0 days. This is a bounded
smoke result, not a substantive estimate.

The ML check used 5,000 2018 rows, 1,000 per state, with a fixed stratified 75/25 split.
Test C-index values were 0.824063 for the forest and 0.821168 for the boosted model.
These values are bounded validation results, not full-cohort estimates or causal effects.

## Project structure

```
court-case-survival-analysis/
├── data/
│   ├── raw/              # downloaded source files (gitignored)
│   └── processed/        # cleaned parquet files (gitignored)
├── notebooks/
│   ├── 01_data_ingestion_cleaning.ipynb
│   ├── 02_kaplan_meier_logrank.ipynb
│   ├── 03_competing_risks.ipynb
│   ├── 04_bayesian_weibull.ipynb
│   └── 05_survival_ml.ipynb
├── src/
│   ├── data_loader.py         # bounded raw-data loading
│   ├── preprocessing.py       # cleaning and survival-table construction
│   ├── survival_classical.py  # KM, Nelson-Aalen, log-rank, Cox PH
│   ├── competing_risks.py     # event coding, hazards, cumulative incidence
│   ├── bayesian_survival.py   # non-hierarchical PyMC Weibull model
│   ├── ml_survival.py         # Random Survival Forest / Gradient Boosting
│   └── dashboard/
│       └── app.py             # bounded Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

## Completed scope and deferred work

- Week 1 ingestion and cleaning are complete. DuckDB validation confirms no invalid
  events, null durations, negative durations, or missing filing dates in the processed
  parquet.
- The classical survival and four-cause competing-risks helpers are implemented and
  validated on bounded samples.
- The Bayesian and ML notebooks are bounded smoke analyses. They do not establish final
  convergence or full-cohort predictive performance.
- Fine-Gray competing-risks fitting, hierarchical Bayesian fitting, and the generic
  Bayesian/classical comparison helper are intentionally deferred.
- No production deployment, commit, or push is implied by the local validation workflow.

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate       # Windows PowerShell/cmd
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```
