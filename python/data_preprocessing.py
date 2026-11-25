"""Data cleaning and preparation for the Telecom Customer Churn project.

Encoding / scaling decisions
- Tariff_Plan (1/2), Status (1/2) and Complains (0/1) are binary codes already stored as numbers,
  so they are kept as numeric columns for the models and mapped to readable labels for reporting.
- Charge_Amount is an ordinal band (0 = lowest) and stays numeric so its order is preserved.
- Age_Group repeats Age, so it is dropped from the model inputs.
- Scaling is only needed by Logistic Regression (done inside its pipeline with StandardScaler);
  tree models (Random Forest, XGBoost, LightGBM) do not need scaled inputs.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Customer_Churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

FEATURES = ["Call_Failure", "Complains", "Subscription_Length", "Charge_Amount", "Seconds_of_Use",
            "Frequency_of_use", "Frequency_of_SMS", "Distinct_Called_Numbers", "Tariff_Plan",
            "Status", "Age", "Customer_Value"]
TARGET = "Churn"


def load_raw(path=RAW_PATH):
    """Read the raw file and normalise column names (spaces -> underscores)."""
    raw = pd.read_csv(path)
    raw.columns = [" ".join(c.split()).replace(" ", "_") for c in raw.columns]
    return raw


def clean(raw):
    """Remove duplicate customer records, add an ID and readable labels."""
    df = raw.drop_duplicates().reset_index(drop=True)
    df.insert(0, "CustomerID", ["C" + str(i + 1).zfill(4) for i in range(len(df))])
    df["Status_Label"] = df.Status.map({1: "Active", 2: "Non-active"})
    df["Tariff_Plan_Label"] = df.Tariff_Plan.map({1: "Pay as you go", 2: "Contractual"})
    df["Complains_Label"] = df.Complains.map({0: "No", 1: "Yes"})
    df["Age_Group_Label"] = df.Age_Group.map({1: "<=15", 2: "16-25", 3: "26-35", 4: "36-45", 5: "46+"})
    df["Subscription_Band"] = pd.cut(df.Subscription_Length, [-1, 12, 24, 36, 100],
                                     labels=["0-12m", "13-24m", "25-36m", "37m+"]).astype(str)
    df["Churn_Label"] = df.Churn.map({0: "Stayed", 1: "Churned"})
    return df


def save_clean(df):
    """Save the cleaned dataset (inputs, labels and target only, no model columns)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cols = (["CustomerID"] + FEATURES + ["Status_Label", "Tariff_Plan_Label", "Age_Group_Label",
                                          "Subscription_Band", "Churn"])
    out = df[cols].sort_values("CustomerID")
    out.to_csv(PROCESSED_DIR / "churn_clean.csv", index=False)
    return out


def get_clean_data():
    raw = load_raw()
    return raw, clean(raw)


if __name__ == "__main__":
    raw, df = get_clean_data()
    print(f"Raw rows: {len(raw)} | duplicates removed: {len(raw) - len(df)} | unique customers: {len(df)}")
    print(f"Missing values: {int(raw.isna().sum().sum())} | churn rate: {df.Churn.mean() * 100:.1f}%")
    save_clean(df)
    print("Saved data/processed/churn_clean.csv")
