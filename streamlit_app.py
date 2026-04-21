"""
STT811 course project — multi-section Streamlit app (single file, sidebar navigation).
Static figures live in assets/figures (run generate_figure_assets.py if missing).
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from churn_logic import decode_customer_profile, load_engine
from train_models import build_train_test

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "assets" / "figures"
RAW_XLSX = BASE_DIR / "data" / "Telco_customer_churn.xlsx"
# Max width for static PNGs so charts fit typical laptop widths with the sidebar (~620–700px).
FIGURE_DISPLAY_WIDTH_PX = 640

# Used only if openpyxl is missing (e.g. old venv). Matches project’s bundled raw Excel snapshot.
_FALLBACK_RAW_MISSINGNESS: dict = {
    "n_rows": 7043,
    "n_cols": 33,
    "n_cols_with_missing": 1,
    "missing_by_col": {"Churn Reason": 5174},
    "_used_fallback": True,
}


@st.cache_data(show_spinner=False)
def raw_excel_missingness_stats() -> dict | None:
    """Missing-value counts from the raw Excel (for the preprocessing page)."""
    if not RAW_XLSX.exists():
        return None
    try:
        raw = pd.read_excel(RAW_XLSX, engine="openpyxl")
    except ImportError:
        return dict(_FALLBACK_RAW_MISSINGNESS)
    miss = raw.isna().sum()
    with_miss = miss[miss > 0]
    return {
        "n_rows": int(len(raw)),
        "n_cols": int(raw.shape[1]),
        "n_cols_with_missing": int(len(with_miss)),
        "missing_by_col": {str(k): int(v) for k, v in with_miss.items()},
    }

NAV_QUESTIONS = [
    "Why this project?",
    "Dataset overview",
    "Data preparation for modeling",
    "What story does this data tell?",
    "Model comparison",
    "What drives a customer to churn?",
    "How should we retain the customers?",
    "TLDR: Too long, didn't read?",
]


def _fig_path(name: str) -> Path:
    return FIGURES_DIR / name


def numeric_distribution_summary_by_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Table: counts and central tendency of key numerics split by Churn (0/1)."""
    rows = []
    n = len(df)
    for val, label in [(0, "Stayed (0)"), (1, "Churned (1)")]:
        sub = df[df["Churn"] == val]
        row = {
            "Churn status": label,
            "Customers": len(sub),
            "% of rows": round(100.0 * len(sub) / n, 2) if n else 0.0,
            "Median tenure (mo)": float(sub["Tenure Months"].median()),
            "Median monthly ($)": float(sub["Monthly Charges"].median()),
        }
        if "Total Charges" in df.columns:
            row["Median total ($)"] = float(sub["Total Charges"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_numeric_by_churn_figure() -> None:
    """Write `02_numeric_by_churn.png` if missing (fresh clone / Cloud without committed PNGs)."""
    out = FIGURES_DIR / "02_numeric_by_churn.png"
    if out.exists():
        return
    csv_path = BASE_DIR / "data" / "Telco_churn_cleaned.csv"
    if not csv_path.exists():
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 4.6))
    specs = [("Tenure Months", "Tenure (months)"), ("Monthly Charges", "Monthly charges ($)")]
    for ax, (col, xlabel) in zip(axes, specs):
        for val, name, color in [(0, "Stayed", "#2ecc71"), (1, "Churned", "#e74c3c")]:
            s = df.loc[df["Churn"] == val, col]
            ax.hist(
                s,
                bins=28,
                alpha=0.55,
                label=name,
                color=color,
                edgecolor="white",
                linewidth=0.25,
            )
        ax.set_ylabel("Count")
        ax.set_xlabel(xlabel)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Numeric distributions by churn status", fontsize=11, y=1.02)
    fig.tight_layout()
    try:
        fig.savefig(out, dpi=96, bbox_inches="tight")
    except OSError:
        pass
    plt.close(fig)


def show_static_figure(filename: str, caption: str | None = None) -> None:
    p = _fig_path(filename)
    if p.exists():
        st.image(str(p), caption=caption, width=FIGURE_DISPLAY_WIDTH_PX)
    else:
        st.warning(
            f"Figure **{filename}** was not found. From the project folder, run: "
            "`python generate_figure_assets.py`"
        )


def plot_confusion_matrix(cm: list, title: str):
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(cm),
        display_labels=["Stayed (0)", "Churned (1)"],
    )
    disp.plot(ax=ax, colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    return fig


@st.cache_resource
def get_engine():
    return load_engine()


@st.cache_resource
def get_classification_bundle():
    return build_train_test()


def render_intro() -> None:
    st.header("Introduction")
    st.markdown(
        "Losing customers hurts recurring revenue and makes marketing spend less efficient. "
        "This project asks: **who is likely to leave**, **what that might cost**, and "
        "**what concrete steps** could change the outcome."
    )
    with st.expander("What “success” means for us", expanded=False):
        st.markdown(
            "- A **clear story** leaders can follow without reading code.\n"
            "- Accurate Churn **predictions** to maximize teams' workload efficiency.\n"
            "- **Transparency** so we can explain *why* someone is flagged.\n"
            "- A **decision layer** that goes beyond a single probability score."
        )
    # with st.expander("Optional: terms we use in plain language", expanded=False):
    #     st.markdown(
    #         "**Churn** — the customer leaves in the observation window.  \n"
    #         "**Features** — facts we know before predicting (bill, tenure, services, etc.).  \n"
    #         "**Model** — a learned rule that maps features to risk.  \n"
    #         "**Baseline** — a silly-but-simple guess we compare against so improvements are real."
    #     )


def render_dataset() -> None:
    st.header("First look at the data")
    st.markdown(
        "We use the public **Telco Customer Churn** table (IBM sample on Kaggle). Each row is "
        "one account with services, contract, billing, and a label for whether they churned."
    )
    st.markdown(
        "[Download the original dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) "
        "(free account)."
    )
    eng = get_engine()
    df = eng.df_clean
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (cleaned file)", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric(
        "Approx. churn rate",
        f"{100 * df['Churn'].mean():.1f}%",
    )
    st.subheader("Preview: first rows (head)")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Summary")
    st.markdown(
        "The cleaned file is almost entirely **numeric flags and amounts** (categories were turned into 0/1 columns). "
        "Below is a **readable snapshot** of the fields that matter most for “who pays what” and “how long they’ve stayed”."
       
    )
    story_cols = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "Avg Monthly Spend",
        "Bill_Shock_Ratio",
        "Churn",
    ]
    present = [c for c in story_cols if c in df.columns]
    rows = []
    for col in present:
        s = df[col]
        label = col.replace("_", " ")
        rows.append(
            {
                "Field": label,
                "Typical (median)": float(s.median()),
                "Average (mean)": float(s.mean()),
                "Low": float(s.min()),
                "High": float(s.max()),
            }
        )
    snap = pd.DataFrame(rows)
    st.dataframe(
        snap.round(3),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        "**How to read this**\n"
        "- **Tenure / charges / totals** — if the **average** is noticeably higher than the **typical (median)** value, "
        "a minority of high bills or long tenures are pulling the mean up.\n"
        "- **Churn** — the **average** is the share who churned (matches the headline churn rate). The median is just "
        "the “middle” label (0 or 1) and is less informative for storytelling.\n"
        "- **Avg Monthly Spend / Bill shock** — engineered fields; see *Data preparation* and *What story does this data tell?* "
        "for what they capture."
    )

    with st.expander("Detailed numeric summary (all numeric columns)", expanded=False):
        num_only = df.select_dtypes(include=[np.number])
        st.caption("Same information as `describe()`, but limited to numeric columns and rounded for scanning.")
        st.dataframe(num_only.describe().T.round(4), use_container_width=True)

    st.subheader("Feature types (numeric vs. categorical)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    c1, c2 = st.columns(2)
    c1.metric("Numeric columns", len(num_cols))
    c2.metric("Categorical / non-numeric columns", len(cat_cols))
    with st.expander("Show numeric columns", expanded=False):
        st.write(num_cols)
    with st.expander("Show categorical / non-numeric columns", expanded=False):
        st.write(cat_cols)
    # st.info(
    #     "**What we were looking for:** Does each row read like a real customer—who they are, "
    #     "what they bought, what they pay—and is “left vs. stayed” labeled clearly? "
    #     "**What the preview shows:** Most people stayed; churners are the smaller slice, so a model "
    #     "that always guesses “stayed” could look almost right while still missing almost everyone who left. "
    #     "Many columns are choices (contract type, how they pay, add-ons), not raw numbers—later we convert "
    #     "those into simple on/off flags so the model can learn from them."
    # )
    with st.expander("Meta data", expanded=False):
        st.markdown(
            "- **Who:** gender, senior flag, partner, dependents.\n"
            "- **Account age & money:** tenure months, monthly charges, total charges, engineered spend ratios.\n"
            "- **Products:** phone, internet type, security, backup, streaming, support, etc.\n"
            "- **Contract & payment:** month-to-month vs. term, auto-pay vs. check.\n"
            "- **Target:** `Churn` (1 = left, 0 = stayed)."
        )


def render_preprocessing() -> None:
    st.header("How we made the data ready to train")
    st.markdown(
        "Raw tables need cleaning, sensible encodings, and honest handling of imbalance "
        "before any algorithm sees the numbers."
    )

    st.subheader("1) Cleaning & feature table")
    st.markdown(
        "Missing or inconsistent entries were fixed in the notebooks; categorical fields were "
        "turned into **one-hot** columns (each category becomes its own 0/1 flag) so models can "
        "measure the effect of, say, fiber internet vs. DSL."
    )
    st.subheader("Missing values — how much, and how we handled it")
    stats = raw_excel_missingness_stats()
    if stats:
        n = stats["n_rows"]
        churn_reason = stats["missing_by_col"].get("Churn Reason", 0)
        pct = 100.0 * churn_reason / n if n else 0.0
        st.markdown(
            f"In the raw **`Telco_customer_churn.xlsx`** ({n:,} rows, {stats['n_cols']} columns), "
            f"**only one field has missing values: `Churn Reason`**, missing in **{churn_reason:,}** rows "
            f"(**{pct:.1f}%** of the file). Every other column is complete in that extract."
        )
        if stats.get("_used_fallback"):
            st.warning(
                "**openpyxl** is not installed, so these counts are a **bundled snapshot**, not read live from the "
                "Excel file. Install with `pip install openpyxl` (it is listed in `requirements.txt`) and redeploy "
                "to recompute from `data/Telco_customer_churn.xlsx`."
            )
    elif not RAW_XLSX.exists():
        st.info(
            "Raw **`Telco_customer_churn.xlsx`** was not found under `data/`, so missingness percentages are skipped. "
            "The handling notes below still apply to how the cleaned CSV was built."
        )
    st.markdown(
        "**How we handled it**\n"
        "- **`Churn Reason`:** Treated as **post-outcome / structural** missing for people who did not churn. "
        "We **do not include it** in the modeling table—using it would **leak** information tied to churn.\n"
        "- **`Total Charges` (in cleaning):** Stored as text in the raw file with blanks where tenure was 0; "
        "we **coerced to numeric** and filled the small number of resulting NaNs with **`Monthly Charges`** "
        "(same idea as the course notebook: a brand-new account’s total spend should align with the current bill).\n"
        "- **Cleaned CSV:** The file the app trains on (`Telco_churn_cleaned.csv`) already reflects these fixes and "
        "does not carry `Churn Reason` as a feature."
    )
    with st.expander("Why one-hot encoding?", expanded=False):
        st.markdown(
            "Algorithms need numbers. One-hot avoids pretending that “contract type” is a single "
            "ordered number (1,2,3) when it is not. After encoding, each level has its own switch."
        )

    st.subheader("2) Class imbalance — random under-sampling")
    show_static_figure(
        "04_balance_before_after.png",
        "Left: real mix of stay vs. churn. Right: balanced training set after under-sampling.",
    )
    st.success(
        "**What we wanted from this chart:** a before/after check that training isn’t dominated "
        "by the majority class. **What we see:** after under-sampling, both classes have equal weight "
        "so the model pays attention to churners instead of always guessing “stay.”"
    )
    with st.expander("Expand: trade-off of under-sampling", expanded=False):
        st.markdown(
            "We **discard** some majority rows to balance classes. That costs a bit of information "
            "on “stay” patterns but stops the model from learning to always predict the common label. "
            "Alternatives exist (over-sampling, class weights); we matched the course notebook for consistency."
        )

    st.subheader("3) Scaling for linear models")
    st.markdown(
        "For **logistic regression**, numeric columns are scaled so large-magnitude fields (like total charges) "
        "do not overpower smaller ones. Tree models in our notebook use the raw table; that’s normal—trees "
        "split on thresholds and don’t need the same scaling."
    )
    with st.expander("Optional: why scaling matters (short)", expanded=False):
        st.markdown(
            "Logistic regression finds weights for each feature. If one feature is on a huge numeric scale, "
            "its weight would need to be tiny to compensate. Scaling puts features on a comparable footing."
        )
        st.caption("No formula here—just the intuition.")

    st.subheader("4) Train / test split")
    st.markdown(
        "We hold out **20%** of the balanced rows to report honest accuracy and error tables. "
        "The split uses a fixed random seed so results are repeatable."
    )


def render_eda() -> None:
    st.header("How to read the charts below")
    st.markdown(
        "Exploratory analysis checks whether patterns make business sense **before** we trust a model. "
        "Each figure below states what we were looking for, what showed up, and why it matters."
    )

    st.subheader("Churn vs. stay — how skewed is the outcome?")
    show_static_figure("01_churn_distribution.png", "Counts of churned vs. stayed customers.")
    st.success(
        "**Goal:** see whether churn is rare enough that “always predict stay” could look deceptively accurate. "
        "**Finding:** churners are the smaller group—so we must balance or weight classes in training and "
        "read metrics carefully."
    )

    st.subheader("Numeric distributions by churn status")
    st.markdown(
        "**What this is:** `Churn` is 0 (stayed) or 1 (left). The **table** summarizes how many customers fall in "
        "each group and the **typical** tenure and bill in each. The **chart** shows the **full distribution** "
        "(histograms) of tenure and monthly charges for stayed vs churned—more detail than a box plot about "
        "where the mass of customers sits."
    )
    eng = get_engine()
    st.dataframe(
        numeric_distribution_summary_by_churn(eng.df_clean).round(3),
        use_container_width=True,
        hide_index=True,
    )
    show_static_figure(
        "02_numeric_by_churn.png",
        "Overlapping histograms: tenure (top) and monthly charges (bottom), stayed vs churned.",
    )
    st.success(
        "**Reading the shapes:** churned customers often pile up at **shorter tenure** and can sit **higher** on "
        "monthly charges—consistent with newer, higher-bill accounts leaving. The table’s medians make that "
        "comparison explicit without hiding skew."
    )

    st.subheader("Which numeric fields move with churn?")
    show_static_figure(
        "05_correlations_with_churn.png",
        "Strength of linear association with churn (engineered + raw numerics).",
    )
    st.success(
        "**Goal:** spot drivers that line up with domain sense. **Finding:** several engineered "
        "and billing fields correlate with churn; this guided feature work and matches later model emphasis."
    )

    st.divider()
    st.header("What story does this data tell?")

    st.subheader("Correlation heatmap (numeric features)")
    show_static_figure(
        "09_correlation_heatmap.png",
        "Correlation among numeric/encoded fields (blue = negative, red = positive).",
    )
    st.success(
        "**What it tells us:** feature groups cluster together (services, contract/payment flags, billing/tenure). "
        "It also highlights where **one-hot columns are mutually exclusive** (strong negative blocks) and where "
        "billing variables co-move (tenure ↔ total charges). This informs which features may be redundant and "
        "why linear models can still work well."
    )

    st.subheader("Univariate (single-variable) highlights")
    show_static_figure("01_churn_distribution.png", "Outcome balance (stay vs. churn).")

    st.subheader("Bivariate (relationship) highlights")
    show_static_figure(
        "10_churn_rate_contract_month_to_month.png",
        "Month-to-month contract flag vs churn rate.",
    )
    show_static_figure(
        "11_churn_rate_electronic_check.png",
        "Electronic check payment flag vs churn rate.",
    )
    show_static_figure(
        "12_churn_rate_fiber_optic.png",
        "Fiber optic internet flag vs churn rate.",
    )
    st.success(
        "**Takeaway:** the biggest churn lifts concentrate in **month-to-month contracts**, **electronic check**, "
        "and **fiber optic** customers—exactly the kinds of levers that translate into actionable retention plays."
    )

    st.subheader("Feature engineering (after EDA)")
    st.markdown(
        "We engineered a few columns to convert raw billing/service facts into **signals a model can learn from** "
        "and a human can interpret."
    )
    st.markdown(
        "**Preprocessing & Feature Engineering Summary**\n"
        "- **Created new columns** (ratios/aggregates) to capture patterns like bill shock and service bundle size.\n"
        "- **Encoded categoricals** using one-hot columns so models can learn per-category effects.\n"
        "- **Scaled numeric features** for logistic regression so coefficients are comparable.\n"
    )
    with st.expander("Engineered Features Detail & Rationale", expanded=False):
        st.markdown(
            "- **Avg Monthly Spend**: `Total Charges / Tenure Months` (with safe handling for tenure=0).  \n"
            "  **Why**: estimates a customer’s historical typical bill; provides baseline context.\n"
            "- **Bill_Shock_Ratio**: `Monthly Charges / Avg Monthly Spend`.  \n"
            "  **Why**: detects “bill shock” (current bill higher than historical average), a common churn trigger.\n"
            "- **Tenure Group**: bucketed tenure.  \n"
            "  **Why**: captures non-linear lifecycle effects (early churn risk vs. long-term stability).\n"
            "- **Total_Addon_Services**: count of add-on services (security/backup/support/streaming…).  \n"
            "  **Why**: approximates product stickiness / bundle depth.\n"
            "- **Is_Auto_Pay**: derived from payment method flags.  \n"
            "  **Why**: auto-pay reduces friction; often correlates with retention.\n"
        )


def model_rationale_block(selected: str) -> None:
    st.markdown("#### How to interpret this pick")
    if selected.startswith("Naive"):
        st.warning(
            "**Why we include it:** sets a floor—if a smart model can’t beat “always guess the majority class,” "
            "something is wrong. **Why we don’t ship it:** it ignores every customer fact; precision on the "
            "minority class is effectively zero."
        )
    elif selected.startswith("Logistic"):
        st.info(
            "**Why it became our “policy” model:** strong accuracy on the held-out set, coefficients we can "
            "discuss with stakeholders, and straightforward **what-if** simulations (flip a service flag, "
            "re-score). **Trade-off:** it assumes smooth relationships; it won’t capture every wild interaction "
            "trees might."
        )
    elif selected.startswith("Random"):
        st.warning(
            "**Strengths:** strong accuracy, captures non-linear patterns, gives feature importance. "
            "**Why we didn’t drive the decision engine with it:** harder to explain case-by-case, and "
            "simulating “change one lever” is messier than with a transparent linear score. Great benchmark."
        )
    else:
        st.warning(
            "**Strengths:** gradient boosting often wins tabular contests. **In our run:** accuracy was "
            "slightly **below** random forest and logistic regression on the same split. **Why not primary:** "
            "more moving parts for non-technical readers, and we already prioritized interpretability for "
            "retention actions."
        )


def render_models() -> None:
    st.header("Model training — compare and question")
    st.markdown(
        "We trained several models on the **same balanced split** as the course notebook. "
        "Use the menu to inspect metrics and confusion patterns. Numbers update from your machine’s run "
        "(cached after first load)."
    )
    bundle = get_classification_bundle()
    overview = pd.DataFrame(bundle["metrics"]).T
    if "roc_auc" in overview.columns:
        overview["roc_auc"] = overview["roc_auc"].apply(
            lambda x: round(x, 3) if pd.notna(x) else "—"
        )
    st.subheader("Snapshot — all models")
    st.dataframe(
        overview.round(4),
        use_container_width=True,
    )
    st.caption(
        "Accuracy is only one view; for churn, false negatives (missing leavers) and false alarms matter too."
    )

    selected = st.selectbox(
        "Which model do you want to inspect in detail?",
        bundle["models_order"],
        index=2,
    )
    model_rationale_block(selected)

    m = bundle["metrics"][selected]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision (churn class)", f"{m['precision']:.3f}")
    c3.metric("Recall (churn class)", f"{m['recall']:.3f}")
    c4.metric("F1", f"{m['f1']:.3f}")
    if pd.notna(m.get("roc_auc")):
        st.metric("ROC AUC (where available)", f"{m['roc_auc']:.3f}")

    cm = bundle["confusion_matrices"][selected]
    st.markdown("**Confusion matrix** — counts of predicted vs. actual labels on the held-out 20%.")
    fig = plot_confusion_matrix(cm, selected)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, width=FIGURE_DISPLAY_WIDTH_PX)
    st.success(
        "**How to read it:** top-left and bottom-right are “got it right” for stay and churn; off-diagonal cells "
        "are costly mistakes—either waving through a leaver or alarming a loyal customer."
    )

    if selected.startswith("Random"):
        st.subheader("Random forest — which features dominated?")
        st.dataframe(bundle["rf_top_features"], use_container_width=True, hide_index=True)
        st.caption("Importance is relative within this model; use alongside SHAP/logistic story for consistency.")

    with st.expander("Optional: why not always use the fanciest model?", expanded=False):
        st.markdown(
            "Leaders often ask for “the best accuracy.” In operations, **trust, speed to explain, and ability "
            "to simulate actions** matter as much as a point of accuracy. Here, logistic regression offered the "
            "best blend for a **live decision story**, even though forests and boosting were competitive benchmarks."
        )


def render_decision_engine() -> None:
    st.header("How should we retain customers?")
    st.markdown(
        "A probability alone doesn’t tell a call center **whom to phone first** or **what to offer**. "
        "The decision engine combines churn risk, a rough **value at risk**, and **simulated interventions** "
        "(add security, change contract, move off electronic check, etc.)."
    )
    with st.expander("Expand: grades A–D in plain language", expanded=False):
        st.markdown(
            "**A —** highest risk and highest expected loss → urgent, high-touch outreach.  \n"
            "**B —** high risk but lower expected loss → strong outreach, lighter incentive.  \n"
            "**C —** medium risk, higher loss → digital nudge + targeted offer.  \n"
            "**D —** medium risk, lower loss → lightweight email reminder."
        )

    eng = get_engine()
    df = eng.df_clean

    st.subheader("Try a single customer")
    if "customer_idx" not in st.session_state:
        st.session_state.customer_idx = 0

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Pick random at-risk customer"):
            Xs = eng.scaler.transform(df[eng.feature_columns])
            probs = eng.lr_model.predict_proba(Xs)[:, 1]
            at_risk = df.index[probs >= 0.5].tolist()
            if at_risk:
                pick = random.choice(at_risk)
                st.session_state.customer_idx = int(df.index.get_loc(pick))
            else:
                st.warning("No at-risk customers found.")
        idx = st.number_input(
            "Row index (0-based)",
            min_value=0,
            max_value=len(df) - 1,
            step=1,
            key="customer_idx",
        )

    row = df.iloc[int(idx)]
    churn_prob = eng.predict_churn_prob(row)
    result = eng.decision_engine(row)

    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability (logistic)", f"{churn_prob:.3f}")
        m2.metric("Risk cutoff used", "0.5")
        if result and result.get("at_risk"):
            m3.metric("Grade", result["grade_info"]["grade"])
        else:
            m3.metric("Grade", "—")

    st.subheader("Profile")
    prof = decode_customer_profile(row)
    pc = st.columns(4)
    for i, (k, v) in enumerate(prof):
        pc[i % 4].markdown(f"**{k}**  \n{v}")

    if not result.get("at_risk"):
        st.success(
            f"Probability **{churn_prob:.3f}** is below **0.5**. Under our rules, no graded retention path fires."
        )
        return

    li = result["loss_info"]
    gi = result["grade_info"]
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("##### Expected loss (illustrative)")
        st.write(f"Reference tenure context: **{li['avg_tenure']}** mo — _{li['source']}_")
        st.write(f"Remaining tenure (estimate): **{li['remaining_tenure']}** months")
        st.write(
            f"Expected loss (20% margin assumption): **${li['expected_loss']:,.2f}** "
            f"vs. cohort median **${result['loss_median']:.2f}**"
        )
        st.caption(f"Discounted variant (×50% uncertainty): ${li['expected_loss_adj']:,.2f}")
    with a2:
        st.markdown("##### Suggested response band")
        st.info(f"**Grade {gi['grade']}** — {gi['desc']}")
        st.caption(f"Incentive hint: {gi['incentive']}")

    st.subheader("Ranked interventions (model re-score after each change)")
    if result["action_df"] is not None and not result["action_df"].empty:
        st.dataframe(
            result["action_df"][
                ["Action", "Group", "Churn Prob", "Prob After", "Reduction", "Below 0.5"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No eligible counterfactual actions for this profile.")

    # if result.get("recommended_table") is not None and not result["recommended_table"].empty:
    #     st.markdown("**Combined view** (combo + top singles when single actions don’t clear the risk bar)")
    #     st.dataframe(result["recommended_table"], use_container_width=True, hide_index=True)

    st.subheader("Batch spot-check")
    n = st.slider("How many random at-risk rows?", 1, 20, 5, key="batch_n")
    if st.button("Run batch", key="run_batch"):
        Xs = eng.scaler.transform(df[eng.feature_columns])
        probs = eng.lr_model.predict_proba(Xs)[:, 1]
        at_risk_idx = df.index[probs > 0.5].tolist()
        random.seed(42)
        sample = random.sample(at_risk_idx, min(n, len(at_risk_idx)))
        rows_out = []
        for i in sample:
            r = df.loc[i]
            res = eng.decision_engine(r)
            if not res or not res.get("at_risk"):
                continue
            top = ""
            if res["grade_info"]["action"] != "Email Only" and not res["action_df"].empty:
                top = str(res["action_df"].iloc[0]["Action"])
            rows_out.append(
                {
                    "index": i,
                    "churn_prob": round(res["churn_prob"], 3),
                    "exp_loss": res["loss_info"]["expected_loss"],
                    "grade": res["grade_info"]["grade"],
                    "top_action": top or "Email only",
                }
            )
        st.dataframe(pd.DataFrame(rows_out), use_container_width=True, hide_index=True)


def render_shap_churn_drivers() -> None:
    st.header("What drives a customer to churn?")
    st.markdown(
        "We use **SHAP** (SHapley Additive exPlanations) to explain **why** the logistic regression flags a customer.\n\n"
        "Important detail: for logistic regression, **SHAP explains the model’s log-odds score** (the value *before* "
        "the sigmoid turns it into a probability). So SHAP answers: **which features pushed this customer’s score "
        "above or below a typical baseline**."
    )

    st.subheader("Global view — average impact direction")
    show_static_figure(
        "06_shap_summary_bar.png",
        "Average SHAP magnitude across many held-out customers (logistic regression).",
    )
    st.success(
        "**What it means:** this is a ranked list of the **strongest levers** in the model on average. "
        "It’s great for training and prioritization (what topics matter most), but it’s not yet a per-customer story."
    )

    st.subheader("Local view — one customer waterfall")
    show_static_figure(
        "07_shap_waterfall.png",
        "How each feature nudges risk for one high-scoring account.",
    )
    st.success(
        "**How to read it:** start at the **baseline log-odds** (typical customer). Each bar is a feature contribution "
        "(red pushes churn risk **up**, blue pushes it **down**). Add them up to get the customer’s final score; "
        "that final score maps to the churn probability shown elsewhere."
    )

    with st.expander("Optional: tiny bit of theory (still no heavy math)", expanded=False):
        st.markdown(
            "**Did we do it correctly?** Yes — using `shap.LinearExplainer` for `sklearn` logistic regression is the "
            "standard choice, and the SHAP values sum back to the model margin (log-odds) under the explainer’s "
            "baseline convention.\n\n"
            "**Why SHAP helps the story:** coefficients tell you global direction (“month-to-month increases risk”), "
            "but SHAP turns that into a **case narrative** (“this customer is high-risk mainly because they are "
            "month-to-month + have low tenure + pay by electronic check”)."
        )


def render_tldr() -> None:
    st.header("TLDR: Too long, didn't read")
    st.markdown(
        "A one-screen recap if you only have a minute before the project demo or read-out."
    )
    st.markdown(
        "- **Problem:** Telco customers leave; we want to **spot risk early**, **prioritize who to call**, "
        "and **suggest concrete levers** (contract, add-ons, payment method).\n"
        "- **Data:** Public IBM-style churn table on Kaggle; we use a **cleaned** CSV with engineered features.\n"
        "- **Prep:** One-hot categories, **balance** classes for training, **scale** inputs for logistic regression, "
        "hold out 20% for honest scores.\n"
        "- **Models:** Naive baseline → **logistic regression** (clear, simulatable) vs. random forest & XGBoost "
        "as benchmarks.\n"
        "- **Retention:** **Grades A–D** from risk + rough value at risk; **ranked “what-if” actions** re-score the customer.\n"
        "- **Trust:** **SHAP** bar + waterfall show **what is pulling this person toward churn**."
    )
    st.subheader("Closing the loop")
    st.markdown(
        "**What we set out to do:** connect data → models → explainability → operational grades.  \n"
        "**What we delivered:** a reproducible pipeline, honest benchmarks, an interpretable champion model for "
        "policy simulation, and SHAP views that help humans trust the machine’s focus areas."
    )


def main() -> None:
    st.set_page_config(
        page_title="STT811 — Telco churn story",
        page_icon="📡",
        layout="wide",
    )
    ensure_numeric_by_churn_figure()

    with st.sidebar:
        st.markdown("### Where do you want to go?")
        page = st.radio(
            " ",
            NAV_QUESTIONS,
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown(
            "**Data source:** [Kaggle — Telco Customer Churn]"
            "(https://www.kaggle.com/datasets/blastchar/telco-customer-churn)"
        )
        st.caption(
            "Figures in `assets/figures`. **`02_numeric_by_churn.png`** is created automatically if missing "
            "(needs `data/Telco_churn_cleaned.csv`). Full set: `python generate_figure_assets.py`."
        )

    st.title("Customer churn prediction")

    if page == NAV_QUESTIONS[0]:
        render_intro()
    elif page == NAV_QUESTIONS[1]:
        render_dataset()
    elif page == NAV_QUESTIONS[2]:
        render_preprocessing()
    elif page == NAV_QUESTIONS[3]:
        render_eda()
    elif page == NAV_QUESTIONS[4]:
        render_models()
    elif page == NAV_QUESTIONS[5]:
        render_shap_churn_drivers()
    elif page == NAV_QUESTIONS[6]:
        render_decision_engine()
    elif page == NAV_QUESTIONS[7]:
        render_tldr()


if __name__ == "__main__":
    main()
