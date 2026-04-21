"""
Generate static PNGs for the Streamlit app (EDA, balance, SHAP).
Run from project root: python generate_figure_assets.py
Requires: matplotlib, shap, xgboost, imbalanced-learn, scikit-learn
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Telco_churn_cleaned.csv"
OUT = ROOT / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 96


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_churn_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    vc = df["Churn"].value_counts().sort_index()
    labels = ["Stayed (0)", "Churned (1)"]
    ax.bar(labels, [vc.get(0, 0), vc.get(1, 0)], color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Customers")
    ax.set_title("Churn vs. stay (full cleaned dataset)")
    plt.tight_layout()
    _save(fig, "01_churn_distribution.png")


def fig_numeric_by_churn(df: pd.DataFrame) -> None:
    """Overlapping histograms for tenure and monthly charges by churn (replaces box plots)."""
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 4.6))
    specs = [
        ("Tenure Months", "Tenure (months)"),
        ("Monthly Charges", "Monthly charges ($)"),
    ]
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
    plt.tight_layout()
    _save(fig, "02_numeric_by_churn.png")


def fig_balance_rus(X, y) -> None:
    rus = RandomUnderSampler(random_state=42)
    Xb, yb = rus.fit_resample(X, y)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
    for ax, series, title in zip(
        axes,
        [y.value_counts().sort_index(), pd.Series(yb).value_counts().sort_index()],
        ["Before balancing", "After random under-sampling"],
    ):
        ax.bar(["Stayed", "Churned"], [series.get(0, 0), series.get(1, 0)], color=["#3498db", "#9b59b6"])
        ax.set_title(title)
        ax.set_ylabel("Rows")
    plt.suptitle("Training data: class balance")
    plt.tight_layout()
    _save(fig, "04_balance_before_after.png")


def fig_numeric_correlations(df: pd.DataFrame) -> None:
    num_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != "Churn" and df[c].notna().all()
    ]
    cor = df[num_cols + ["Churn"]].corr()["Churn"].drop("Churn").sort_values(key=abs, ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    cor.plot(kind="barh", ax=ax, color="#34495e")
    ax.set_title("Top numeric correlations with churn")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    _save(fig, "05_correlations_with_churn.png")


def fig_corr_heatmap(df: pd.DataFrame) -> None:
    num = df.select_dtypes(include=[np.number]).copy()
    if "Churn" in num.columns:
        # Keep churn, but move it to the end for readability
        cols = [c for c in num.columns if c != "Churn"] + ["Churn"]
        num = num[cols]
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation heatmap (numeric features)")
    plt.tight_layout()
    _save(fig, "09_correlation_heatmap.png")


def fig_churn_rate_by_category(df: pd.DataFrame, col: str, out_name: str, title: str) -> None:
    if col not in df.columns:
        return
    tmp = df[[col, "Churn"]].copy()
    tmp[col] = tmp[col].astype(str)
    rate = tmp.groupby(col)["Churn"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    rate.plot(kind="bar", ax=ax, color="#8e44ad")
    ax.set_title(title)
    ax.set_ylabel("Churn rate")
    ax.set_xlabel(col)
    ax.set_ylim(0, min(1.0, max(0.6, float(rate.max()) + 0.05)))
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    _save(fig, out_name)


def fig_shap_plots() -> None:
    df = pd.read_csv(DATA)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_rus, y_rus, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)

    explainer = shap.LinearExplainer(lr, X_train_s)
    sv = explainer.shap_values(X_test_s)
    if isinstance(sv, list):
        sv = sv[1]

    X_test_df = pd.DataFrame(X_test_s, columns=X.columns)

    plt.figure(figsize=(5.8, 4.0))
    shap.summary_plot(
        sv,
        X_test_df,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title("Which features push churn risk up or down (average impact)")
    plt.tight_layout()
    plt.savefig(OUT / "06_shap_summary_bar.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # Waterfall for one high-risk row
    row_i = int(np.argmax(lr.predict_proba(X_test_s)[:, 1]))
    exp = shap.Explanation(
        values=sv[row_i],
        base_values=explainer.expected_value
        if not isinstance(explainer.expected_value, (list, np.ndarray))
        else float(np.ravel(explainer.expected_value)[-1]),
        data=X_test_s[row_i],
        feature_names=list(X.columns),
    )
    plt.figure(figsize=(6.0, 4.2))
    shap.plots.waterfall(exp, max_display=14, show=False)
    plt.title("Example customer: how each factor moves the risk score")
    plt.tight_layout()
    plt.savefig(OUT / "07_shap_waterfall.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(DATA)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print("Writing figures to", OUT)
    fig_churn_distribution(df)
    fig_numeric_by_churn(df)
    fig_balance_rus(X, y)
    fig_numeric_correlations(df)
    fig_corr_heatmap(df)
    fig_churn_rate_by_category(
        df,
        "Contract_Month-to-month",
        "10_churn_rate_contract_month_to_month.png",
        "Churn rate — month-to-month contract flag",
    )
    fig_churn_rate_by_category(
        df,
        "Payment Method_Electronic check",
        "11_churn_rate_electronic_check.png",
        "Churn rate — electronic check payment flag",
    )
    fig_churn_rate_by_category(
        df,
        "Internet Service_Fiber optic",
        "12_churn_rate_fiber_optic.png",
        "Churn rate — fiber optic internet flag",
    )
    fig_shap_plots()
    print("Done.")


if __name__ == "__main__":
    main()
