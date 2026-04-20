# STT811 — Telco customer churn

Course project for **STT811** that turns the [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset into **predictive models** and an **actionable retention decision layer**: estimate who is likely to leave, quantify rough revenue at risk, and rank concrete interventions (services, contract, payment method).

## Dataset (download)

The raw data is hosted on Kaggle (free account required):

- **[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**

This repository already includes a cleaned, feature-engineered CSV used by the notebooks and app:

- `data/Telco_churn_cleaned.csv`

If you download from Kaggle, run your own cleaning pipeline to match that schema, or start from the bundled file for reproducibility.

## Why this project

Telecom churn directly affects recurring revenue and acquisition costs. We started this project to:

1. **Predict churn** with transparent models suitable for discussion with stakeholders (not only black-box accuracy).
2. **Explain drivers** of risk (exploratory analysis, coefficients, SHAP-style views for logistic regression).
3. **Move from scores to decisions** by combining predicted churn probability with a simple **expected-loss** estimate and **simulated counterfactuals** (what changes to services or contract would most reduce predicted churn for a given customer).

## What we built and how it meets those goals

| Stage | What we did | Outcome |
|--------|-------------|--------|
| **EDA & cleaning** | Explored distributions, churn rate, missing values; engineered features (e.g. spend ratios, tenure groups, one-hot encodings). | Documented in `clean_preprocess_EDA.ipynb` / `clean_preprocess_copy2.ipynb`. |
| **Class imbalance** | Random under-sampling before training classifiers. | Balanced training signal for churn vs. stay. |
| **Models** | Dummy baseline, **logistic regression**, **random forest**, **XGBoost**, **Cox PH** for survival-style insight. | Logistic regression reached strong holdout performance (about **80% accuracy** on the balanced split in `classification_models.ipynb`), with clear coefficients; tree models were competitive with different tradeoffs. |
| **Survival insight** | Cox model highlights contract length (e.g. month-to-month) as a major risk driver. | Aligns with retention recommendations that often prioritize **contract changes**. |
| **Decision engine** | At-risk customers (churn probability ≥ 0.5); demographic groups for tenure-based **expected loss**; grades **A–D** by probability + loss vs. median; ranked **single-action** and **two-action** simulations. | Documented and implemented in `decision_engine_SHAP.ipynb`; same logic powers `churn_logic.py` and the Streamlit app. |
| **Explainability** | SHAP for the fitted logistic regression. | Connects model output to feature contributions for individual customers. |

Together, this delivers both **measurable performance** and a **repeatable business rulebook** for prioritizing retention effort.

## Clone this project

If the project is in a Git repository (replace the URL with yours):

```bash
git clone https://github.com/<your-username>/STT811-Project.git
cd STT811-Project
```

If you received a `.zip`, extract it and `cd` into the folder that contains `README.md`, `data/`, and the notebooks.

## Environment setup

Python 3.9+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Optional (for running the full notebooks): `jupyter`, `lifelines`, `plotly`.

## Streamlit app

Single-file app with **sidebar navigation** (question-style labels): introduction, dataset, preprocessing, EDA, **interactive model comparison**, decision engine, and SHAP wrap-up. **Static charts** live under `assets/figures/`; regenerate if needed:

```bash
python generate_figure_assets.py
streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Notebooks (reference)

- `clean_preprocess_EDA.ipynb` — EDA and preprocessing narrative  
- `classification_models.ipynb` — baselines, logistic regression, RF, XGBoost, Cox PH  
- `decision_engine_SHAP.ipynb` — decision engine, SHAP  

---

*STT811 — statistical learning / applied churn analytics project.*
