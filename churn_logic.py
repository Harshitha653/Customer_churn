"""
Core churn model and decision engine (aligned with classification_models.ipynb
and decision_engine_SHAP.ipynb).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco_churn_cleaned.csv"

THRESHOLD = 0.5
MIN_GROUP_COUNT = 20
MARGIN_RATE = 0.2
TENURE_ASSUMPTION = 0.5

ACTION_GROUPS = {
    "online_security": {"Add Online Security": {"Online Security_Yes": 1, "Online Security_No": 0}},
    "online_backup": {"Add Online Backup": {"Online Backup_Yes": 1, "Online Backup_No": 0}},
    "device_protection": {
        "Add Device Protection": {"Device Protection_Yes": 1, "Device Protection_No": 0}
    },
    "tech_support": {"Add Tech Support": {"Tech Support_Yes": 1, "Tech Support_No": 0}},
    "contract": {
        "Switch to 1-Year Contract": {"Contract_Month-to-month": 0, "Contract_One year": 1},
        "Switch to 2-Year Contract": {"Contract_Month-to-month": 0, "Contract_Two year": 1},
    },
    "payment": {
        "Switch to Mailed Check": {
            "Payment Method_Electronic check": 0,
            "Payment Method_Mailed check": 1,
        },
        "Switch to Bank Transfer": {
            "Payment Method_Electronic check": 0,
            "Payment Method_Bank transfer (automatic)": 1,
        },
        "Switch to Credit Card": {
            "Payment Method_Electronic check": 0,
            "Payment Method_Credit card (automatic)": 1,
        },
    },
}

ACTIONS: dict[str, dict[str, int]] = {}
ACTION_TO_GROUP: dict[str, str] = {}
for group_name, actions in ACTION_GROUPS.items():
    for action_name, changes in actions.items():
        ACTIONS[action_name] = changes
        ACTION_TO_GROUP[action_name] = group_name

GROUP_COLS = ["Gender", "Senior Citizen", "Partner", "Dependents"]


def load_engine():
    """Load data, fit scaler + logistic regression, build group stats and loss median."""
    df_clean = pd.read_csv(DATA_PATH)
    X = df_clean.drop("Churn", axis=1)
    y = df_clean["Churn"]
    feature_columns = X.columns.tolist()

    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_rus, y_rus, test_size=0.2, random_state=42
    )

    no_churn_df = df_clean[df_clean["Churn"] == 0].copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    group_avg = (
        no_churn_df.groupby(GROUP_COLS)["Tenure Months"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_tenure", "count": "n"})
    )
    global_avg = no_churn_df["Tenure Months"].mean()
    group_avg_sorted = group_avg.sort_values("avg_tenure", ascending=False).reset_index(drop=True)
    group_avg_sorted["rank"] = group_avg_sorted.index + 1
    total_groups = len(group_avg_sorted)

    def get_group_info(customer_row: pd.Series) -> tuple[float, str]:
        mask = pd.Series([True] * len(group_avg_sorted), index=group_avg_sorted.index)
        for col in GROUP_COLS:
            mask &= group_avg_sorted[col] == customer_row[col]
        matched = group_avg_sorted[mask]
        if not matched.empty and matched.iloc[0]["n"] >= MIN_GROUP_COUNT:
            avg_t = float(matched.iloc[0]["avg_tenure"])
            rank = int(matched.iloc[0]["rank"])
            source = f"Group avg_tenure rank {rank} of {total_groups}"
        else:
            avg_t = float(global_avg)
            source = "Overall no-churn average used (group too small)"
        return avg_t, source

    X_clean_scaled = scaler.transform(df_clean[feature_columns])
    all_probs = lr_model.predict_proba(X_clean_scaled)[:, 1]
    high_risk_df = df_clean[all_probs > THRESHOLD].copy()

    losses = []
    for _, row in high_risk_df.iterrows():
        avg_t, _ = get_group_info(row)
        remaining = max(avg_t - row["Tenure Months"], 0)
        losses.append(remaining * row["Monthly Charges"] * MARGIN_RATE)
    loss_median = float(pd.Series(losses).median())

    return EngineState(
        df_clean=df_clean,
        feature_columns=feature_columns,
        scaler=scaler,
        lr_model=lr_model,
        group_avg_sorted=group_avg_sorted,
        global_avg=float(global_avg),
        total_groups=total_groups,
        loss_median=loss_median,
        get_group_info=get_group_info,
    )


class EngineState:
    __slots__ = (
        "df_clean",
        "feature_columns",
        "scaler",
        "lr_model",
        "group_avg_sorted",
        "global_avg",
        "total_groups",
        "loss_median",
        "_get_group_info_fn",
    )

    def __init__(
        self,
        df_clean: pd.DataFrame,
        feature_columns: list[str],
        scaler: StandardScaler,
        lr_model: LogisticRegression,
        group_avg_sorted: pd.DataFrame,
        global_avg: float,
        total_groups: int,
        loss_median: float,
        get_group_info,
    ):
        self.df_clean = df_clean
        self.feature_columns = feature_columns
        self.scaler = scaler
        self.lr_model = lr_model
        self.group_avg_sorted = group_avg_sorted
        self.global_avg = global_avg
        self.total_groups = total_groups
        self.loss_median = loss_median
        self._get_group_info_fn = get_group_info

    def get_group_info(self, customer_row: pd.Series) -> tuple[float, str]:
        return self._get_group_info_fn(customer_row)

    def estimate_expected_loss(self, customer_row: pd.Series) -> dict:
        avg_t, source = self.get_group_info(customer_row)
        current_tenure = customer_row["Tenure Months"]
        if current_tenure > avg_t:
            remaining = min(12, max(72 - current_tenure, 0))
            source = source + " / exceeds group avg → remaining = min(12, 72 - tenure)"
        else:
            remaining = avg_t - current_tenure
        expected_loss = remaining * customer_row["Monthly Charges"] * MARGIN_RATE
        expected_loss_adj = expected_loss * TENURE_ASSUMPTION
        return {
            "expected_loss": round(float(expected_loss), 2),
            "expected_loss_adj": round(float(expected_loss_adj), 2),
            "avg_tenure": round(float(avg_t), 2),
            "remaining_tenure": round(float(remaining), 2),
            "source": source,
        }

    def classify_response(self, churn_prob: float, expected_loss: float) -> dict:
        prob_grade = "High" if churn_prob >= 0.7 else "Mid"
        loss_grade = "Upper" if expected_loss >= self.loss_median else "Lower"
        grade_map = {
            ("High", "Upper"): {
                "grade": "A",
                "action": "Top2 Combo",
                "incentive": "Active financial support",
                "desc": "Immediate call + Top-2 combo + Active financial support",
            },
            ("High", "Lower"): {
                "grade": "B",
                "action": "Top1 Action",
                "incentive": "Minor financial support",
                "desc": "Phone call + Top-1 action + Minor financial support",
            },
            ("Mid", "Upper"): {
                "grade": "C",
                "action": "Top1 Action",
                "incentive": "Minor financial support",
                "desc": "Email + Top-1 action + Minor financial support",
            },
            ("Mid", "Lower"): {
                "grade": "D",
                "action": "Email Only",
                "incentive": "None",
                "desc": "Email notification only",
            },
        }
        return grade_map[(prob_grade, loss_grade)]

    def simulate_single_actions(self, customer_row: pd.Series, churn_prob: float) -> pd.DataFrame:
        results = []
        for action_name, changes in ACTIONS.items():
            modified = customer_row[self.feature_columns].copy()
            if not all(f in modified.index for f in changes):
                continue
            if all(modified.get(f, None) == v for f, v in changes.items()):
                continue
            for f, v in changes.items():
                modified[f] = v
            modified_scaled = self.scaler.transform(
                pd.DataFrame([modified], columns=self.feature_columns)
            )
            new_prob = float(self.lr_model.predict_proba(modified_scaled)[0, 1])
            results.append(
                {
                    "Action": action_name,
                    "Group": ACTION_TO_GROUP[action_name],
                    "Churn Prob": round(churn_prob, 3),
                    "Prob After": round(new_prob, 3),
                    "Reduction": round(churn_prob - new_prob, 3),
                    "Below 0.5": "✓" if new_prob < THRESHOLD else "✗",
                }
            )
        return pd.DataFrame(results).sort_values("Reduction", ascending=False)

    def simulate_top2_combo(
        self, customer_row: pd.Series, action_df: pd.DataFrame, churn_prob: float
    ) -> dict | None:
        if len(action_df) < 2:
            return None
        top1 = action_df.iloc[0]
        others = action_df[action_df["Group"] != top1["Group"]]
        if others.empty:
            return None
        top2 = others.iloc[0]
        modified = customer_row[self.feature_columns].copy()
        for action_name in [top1["Action"], top2["Action"]]:
            for f, v in ACTIONS[action_name].items():
                modified[f] = v
        modified_scaled = self.scaler.transform(
            pd.DataFrame([modified], columns=self.feature_columns)
        )
        new_prob = float(self.lr_model.predict_proba(modified_scaled)[0, 1])
        return {
            "Action": f"{top1['Action']}  +  {top2['Action']}",
            "Churn Prob": round(churn_prob, 3),
            "Prob After": round(new_prob, 3),
            "Reduction": round(churn_prob - new_prob, 3),
            "Below 0.5": "✓" if new_prob < THRESHOLD else "✗",
        }

    def predict_churn_prob(self, customer_row: pd.Series) -> float:
        row_scaled = self.scaler.transform(
            pd.DataFrame([customer_row[self.feature_columns]], columns=self.feature_columns)
        )
        return float(self.lr_model.predict_proba(row_scaled)[0, 1])

    def decision_engine(self, customer_row: pd.Series) -> dict | None:
        churn_prob = self.predict_churn_prob(customer_row)
        if churn_prob < THRESHOLD:
            return {
                "at_risk": False,
                "churn_prob": churn_prob,
                "threshold": THRESHOLD,
            }

        loss_info = self.estimate_expected_loss(customer_row)
        grade_info = self.classify_response(churn_prob, loss_info["expected_loss"])
        action_df = self.simulate_single_actions(customer_row, churn_prob)

        recommended_table = None
        combo_info = None
        if grade_info["action"] == "Email Only":
            pass
        else:
            if not action_df.empty:
                best_single = action_df.iloc[0]
                single_achieves = bool(best_single["Prob After"] < THRESHOLD)
                if single_achieves:
                    recommended_table = action_df.head(5).reset_index(drop=True)
                else:
                    combo_info = self.simulate_top2_combo(customer_row, action_df, churn_prob)
                    rows = []
                    if combo_info:
                        rows.append(
                            {
                                "Action": combo_info["Action"],
                                "Churn Prob": combo_info["Churn Prob"],
                                "Prob After": combo_info["Prob After"],
                                "Reduction": combo_info["Reduction"],
                                "Below 0.5": combo_info["Below 0.5"],
                            }
                        )
                    for _, r in action_df.head(4).iterrows():
                        rows.append(r.to_dict())
                    recommended_table = pd.DataFrame(rows)

        return {
            "at_risk": True,
            "churn_prob": churn_prob,
            "threshold": THRESHOLD,
            "loss_info": loss_info,
            "loss_median": self.loss_median,
            "grade_info": grade_info,
            "action_df": action_df,
            "recommended_table": recommended_table,
            "combo_info": combo_info,
        }


def decode_customer_profile(customer_row: pd.Series) -> list[tuple[str, str]]:
    def find_category(prefix: str) -> str:
        cols = [c for c in customer_row.index if c.startswith(prefix) and customer_row[c] == 1]
        return cols[0].replace(prefix, "") if cols else "Unknown"

    service_cols = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]
    services = [s for s in service_cols if customer_row.get(f"{s}_Yes", 0) == 1]

    return [
        ("Tenure", f"{int(customer_row.get('Tenure Months', 0))} months"),
        ("Monthly", f"${customer_row.get('Monthly Charges', 0):.2f}"),
        ("Gender", "Male" if customer_row.get("Gender", 0) == 1 else "Female"),
        ("Senior", "Yes" if customer_row.get("Senior Citizen", 0) == 1 else "No"),
        ("Partner", "Yes" if customer_row.get("Partner", 0) == 1 else "No"),
        ("Dependents", "Yes" if customer_row.get("Dependents", 0) == 1 else "No"),
        ("Internet", find_category("Internet Service_")),
        ("Contract", find_category("Contract_")),
        ("Payment", find_category("Payment Method_")),
        ("Services", ", ".join(services) if services else "None"),
    ]
