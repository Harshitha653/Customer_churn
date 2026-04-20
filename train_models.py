"""
Train classifiers on the same pipeline as classification_models.ipynb (RUS + split).
Used by Streamlit for interactive model comparison.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).resolve().parent / "data" / "Telco_churn_cleaned.csv"


def load_xy():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y, df


def build_train_test():
    X, y, df_clean = load_xy()
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_rus, y_rus, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_names = list(X_train.columns)

    naive = DummyClassifier(strategy="most_frequent")
    naive.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(eval_metric="logloss", random_state=42)
    xgb.fit(X_train, y_train)

    preds = {
        "Naive baseline (majority class)": naive.predict(X_test),
        "Logistic regression": lr.predict(X_test_scaled),
        "Random forest": rf.predict(X_test),
        "XGBoost": xgb.predict(X_test),
    }
    prob_models = {
        "Logistic regression": lr.predict_proba(X_test_scaled)[:, 1],
        "Random forest": rf.predict_proba(X_test)[:, 1],
        "XGBoost": xgb.predict_proba(X_test)[:, 1],
    }

    def metrics(name, y_pred, y_prob=None):
        m = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        if y_prob is not None:
            try:
                m["roc_auc"] = float(roc_auc_score(y_test, y_prob))
            except ValueError:
                m["roc_auc"] = None
        else:
            m["roc_auc"] = None
        return m

    metrics_by_model = {}
    for name, y_pred in preds.items():
        y_prob = prob_models.get(name)
        metrics_by_model[name] = metrics(name, y_pred, y_prob)

    cm_by_model = {
        name: confusion_matrix(y_test, y_pred).tolist()
        for name, y_pred in preds.items()
    }

    rf_importance = (
        pd.DataFrame({"Feature": feature_names, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(12)
    )

    return {
        "y_test": y_test,
        "X_test": X_test,
        "feature_names": feature_names,
        "preds": preds,
        "metrics": metrics_by_model,
        "confusion_matrices": cm_by_model,
        "rf_top_features": rf_importance,
        "models_order": list(preds.keys()),
    }
