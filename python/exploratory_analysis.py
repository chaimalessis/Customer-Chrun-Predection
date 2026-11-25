"""Exploration, outliers and leakage documentation (guideline steps 1-4)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import get_clean_data, RESULTS_DIR

NUM = ["Call_Failure", "Subscription_Length", "Seconds_of_Use", "Frequency_of_use",
       "Frequency_of_SMS", "Distinct_Called_Numbers", "Customer_Value"]

LEAKAGE_REVIEW = pd.DataFrame([
    ("Call_Failure", "9-month summary", "Yes", "Low"),
    ("Complains", "Flag from the 9-month window", "Yes",
     "Medium: complaint may be filed at the point of leaving; confirm with the business"),
    ("Subscription_Length", "Account age at observation", "Yes", "Low"),
    ("Charge_Amount", "9-month billing band", "Yes", "Low"),
    ("Seconds_of_Use", "9-month total", "Yes", "Low"),
    ("Frequency_of_use", "9-month total", "Yes", "Low"),
    ("Frequency_of_SMS", "9-month total", "Yes", "Low"),
    ("Distinct_Called_Numbers", "9-month total", "Yes", "Low"),
    ("Tariff_Plan", "Contract type at observation", "Yes", "Low"),
    ("Status", "Active/non-active at observation", "Yes",
     "Medium: non-active may already mean disengaged; confirm it is set before the 3-month gap"),
    ("Age / Age_Group", "Demographic", "Yes", "Low (duplicate information: Age_Group dropped)"),
    ("Customer_Value", "Calculated value at observation", "Yes", "Low (also used for value at risk)"),
], columns=["Feature", "Source", "Available_At_Prediction", "Leakage_Risk"])


def outlier_table(df):
    """Mean, median, skew and IQR outliers (upper side) with their churn rate."""
    rows = []
    for c in NUM:
        q1, q3 = df[c].quantile([.25, .75])
        hi = q3 + 1.5 * (q3 - q1)
        out = df[df[c] > hi]
        rows.append(dict(Variable=c, Mean=df[c].mean(), Median=df[c].median(), Max=df[c].max(),
                         Skew=df[c].skew(), Outliers_IQR=len(out),
                         Outlier_Churn_Rate_pct=100 * out.Churn.mean() if len(out) else np.nan))
    return pd.DataFrame(rows).round(2)


def quartile_table(df):
    """Churn rate in each quartile of every numeric variable."""
    parts = []
    for c in NUM:
        q = pd.qcut(df[c].rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
        t = (df.groupby(q, observed=True)
               .agg(n=("Churn", "size"), churn_pct=("Churn", "mean"), min=(c, "min"), max=(c, "max"))
               .reset_index().rename(columns={c: "Quartile"}))
        t["churn_pct"] = (t.churn_pct * 100).round(1)
        t.insert(0, "Variable", c)
        parts.append(t)
    return pd.concat(parts)


def distribution_figure(df, path):
    """Histogram and boxplot of every numeric variable."""
    fig, axes = plt.subplots(2, len(NUM), figsize=(3 * len(NUM), 6))
    for k, c in enumerate(NUM):
        axes[0, k].hist(df[c], bins=30, color="#1F6FB5")
        axes[0, k].set_title(c, fontsize=9)
        axes[1, k].boxplot(df[c])
    axes[0, 0].set_ylabel("Customers")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    plt.switch_backend("Agg")  # no window when run as a script
    raw, df = get_clean_data()
    print("Rows", len(raw), "| missing", int(raw.isna().sum().sum()), "| duplicate rows", int(raw.duplicated().sum()))
    print("After cleaning", len(df), "| churn rate", round(df.Churn.mean() * 100, 1), "%")
    eda, bins = outlier_table(df), quartile_table(df)
    print(eda.to_string(index=False))
    print(bins.to_string(index=False))
    for c in ["Complains", "Tariff_Plan", "Status", "Age_Group"]:
        print(df.groupby(c).Churn.agg(["size", "mean"]).round(3))
    print(LEAKAGE_REVIEW.to_string(index=False))
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)
    eda.to_csv(RESULTS_DIR / "eda_outliers.csv", index=False)
    bins.to_csv(RESULTS_DIR / "eda_churn_by_quartile.csv", index=False)
    LEAKAGE_REVIEW.to_csv(RESULTS_DIR / "leakage_review.csv", index=False)
    distribution_figure(df, RESULTS_DIR / "figures" / "distributions.png")


if __name__ == "__main__":
    main()
