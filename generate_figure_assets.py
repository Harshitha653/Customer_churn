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
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Telco_churn_cleaned.csv"
OUT = ROOT / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def fig_churn_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df["Churn"].value_counts().sort_index()
    labels = ["Stayed (0)", "Churned (1)"]
    ax.bar(labels, [vc.get(0, 0), vc.get(1, 0)], color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Customers")
    ax.set_title("Churn vs. stay (full cleaned dataset)")
    plt.tight_layout()
    fig.savefig(OUT / "01_churn_distribution.png", dpi=150)
    plt.close()


def fig_tenure_by_churn(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    stay = df.loc[df["Churn"] == 0, "Tenure Months"]
    churn = df.loc[df["Churn"] == 1, "Tenure Months"]
    ax.boxplot([stay, churn], tick_labels=["Stayed", "Churned"])
    ax.set_ylabel("Tenure (months)")
    ax.set_title("How long customers stayed — by outcome")
    plt.tight_layout()
    fig.savefig(OUT / "02_tenure_by_churn.png", dpi=150)
    plt.close()


def fig_monthly_charges(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    stay = df.loc[df["Churn"] == 0, "Monthly Charges"]
    churn = df.loc[df["Churn"] == 1, "Monthly Charges"]
    ax.boxplot([stay, churn], tick_labels=["Stayed", "Churned"])
    ax.set_ylabel("Monthly charges ($)")
    ax.set_title("Monthly bill — by churn outcome")
    plt.tight_layout()
    fig.savefig(OUT / "03_monthly_charges_by_churn.png", dpi=150)
    plt.close()


def fig_balance_rus(X, y) -> None:
    rus = RandomUnderSampler(random_state=42)
    Xb, yb = rus.fit_resample(X, y)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
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
    fig.savefig(OUT / "04_balance_before_after.png", dpi=150)
    plt.close()


def fig_numeric_correlations(df: pd.DataFrame) -> None:
    num_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != "Churn" and df[c].notna().all()
    ]
    cor = df[num_cols + ["Churn"]].corr()["Churn"].drop("Churn").sort_values(key=abs, ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    cor.plot(kind="barh", ax=ax, color="#34495e")
    ax.set_title("Top numeric correlations with churn")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(OUT / "05_correlations_with_churn.png", dpi=150)
    plt.close()


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

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        sv,
        X_test_df,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title("Which features push churn risk up or down (average impact)")
    plt.tight_layout()
    plt.savefig(OUT / "06_shap_summary_bar.png", dpi=150, bbox_inches="tight")
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
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(exp, max_display=14, show=False)
    plt.title("Example customer: how each factor moves the risk score")
    plt.tight_layout()
    plt.savefig(OUT / "07_shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(DATA)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print("Writing figures to", OUT)
    fig_churn_distribution(df)
    fig_tenure_by_churn(df)
    fig_monthly_charges(df)
    fig_balance_rus(X, y)
    fig_numeric_correlations(df)
    fig_shap_plots()
    print("Done.")


if __name__ == "__main__":
    main()
