"""Score all customers: churn probability, value at risk, risk band, value tier and retention action."""
import numpy as np
import pandas as pd
import sklearn.base as skb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from data_preprocessing import get_clean_data, FEATURES, RESULTS_DIR, PROCESSED_DIR
from model_training import train_and_compare
from model_evaluation import evaluate, RETENTION_COST, SAVE_RATE

ACTION_PLAN = {
    "1 - Personal call + best offer": ("High risk and High-Value", "Account manager calls with the strongest offer", "Immediate"),
    "2 - Targeted discount / SMS offer": ("High risk, other tiers", "Targeted discount or SMS offer, tested against a control group", "This week"),
    "3 - Proactive check-in": ("Medium risk, Mid or High value", "Light contact to understand problems before they grow", "This month"),
    "4 - Automated nurture": ("Medium risk, Standard value", "Automated messages only, no manual cost", "Automated"),
    "5 - Normal service": ("Low risk", "No action; contacting costs more than it saves", "None"),
}


def risk_band(p):
    return "High" if p >= 0.5 else ("Medium" if p >= 0.2 else "Low")


def retention_action(r):
    if r.Risk_Band == "High" and r.Value_Tier == "High-Value":
        return "1 - Personal call + best offer"
    if r.Risk_Band == "High":
        return "2 - Targeted discount / SMS offer"
    if r.Risk_Band == "Medium" and r.Value_Tier != "Standard":
        return "3 - Proactive check-in"
    if r.Risk_Band == "Medium":
        return "4 - Automated nurture"
    return "5 - Normal service"


def score_customers(df, res, thr):
    """Out-of-fold calibrated probabilities for ALL customers (each is scored by a model that never saw it)."""
    X, y = res["X"], res["y"]
    bm = res["models"][res["best"]]
    cvm = CalibratedClassifierCV(skb.clone(bm), method="isotonic", cv=5)
    oof = cross_val_predict(cvm, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1),
                            method="predict_proba")[:, 1]
    df = df.copy()
    df["Churn_Probability"] = oof.round(4)
    df["Value_At_Risk"] = (df.Churn_Probability * df.Customer_Value).round(2)
    df["Retention_Cost"] = RETENTION_COST
    df["Expected_Retention_Benefit"] = (df.Value_At_Risk * SAVE_RATE - RETENTION_COST).round(2)
    df["Contact_Recommended"] = np.where(df.Expected_Retention_Benefit > 0, "Yes", "No")
    df["Predicted_Churn"] = (df.Churn_Probability >= thr).astype(int)
    df["Risk_Band"] = df.Churn_Probability.map(risk_band)
    df["Value_Tier"] = pd.qcut(df.Customer_Value, [0, .5, .8, 1],
                               labels=["Standard", "Mid-Value", "High-Value"]).astype(str)
    df["Retention_Action"] = df.apply(retention_action, axis=1)
    df["Priority_Rank"] = df.Value_At_Risk.rank(ascending=False, method="first").astype(int)
    return df.sort_values("Priority_Rank"), roc_auc_score(y, oof)


def risk_segments(df):
    """One row per risk band: size, value, value at risk, actual churn and the value-tier mix."""
    g = df.groupby("Risk_Band").agg(Customers=("Churn", "size"), Total_Value=("Customer_Value", "sum"),
                                    Value_At_Risk=("Value_At_Risk", "sum"),
                                    Actual_Churn_Rate_pct=("Churn", lambda s: 100 * s.mean())).round(1)
    mix = pd.crosstab(df.Risk_Band, df.Value_Tier).reindex(columns=["High-Value", "Mid-Value", "Standard"], fill_value=0)
    return g.join(mix).reindex(["High", "Medium", "Low"]).reset_index()


def retention_recommendations(df):
    g = df.groupby("Retention_Action").agg(Customers=("Churn", "size"), Value_At_Risk=("Value_At_Risk", "sum"),
                                           Expected_Benefit=("Expected_Retention_Benefit", "sum")).round(0).reset_index()
    g["Rule"] = g.Retention_Action.map(lambda a: ACTION_PLAN[a][0])
    g["What_The_Team_Does"] = g.Retention_Action.map(lambda a: ACTION_PLAN[a][1])
    g["Priority"] = g.Retention_Action.map(lambda a: ACTION_PLAN[a][2])
    return g[["Retention_Action", "Rule", "Customers", "Value_At_Risk", "Expected_Benefit",
              "What_The_Team_Does", "Priority"]]


def main():
    _, df = get_clean_data()
    res = train_and_compare(df)
    thr = evaluate(res)["threshold"]
    scored, auc = score_customers(df, res, thr)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    scored.to_csv(PROCESSED_DIR / "churn_scored.csv", index=False)
    seg, rec = risk_segments(scored), retention_recommendations(scored)
    seg.to_csv(RESULTS_DIR / "customer_risk_segments.csv", index=False)
    rec.to_csv(RESULTS_DIR / "retention_recommendations.csv", index=False)
    print("Out-of-fold AUC:", round(auc, 4))
    print("Churn rate:", round(scored.Churn.mean() * 100, 1), "% | value at risk:", round(scored.Value_At_Risk.sum()),
          "| customers to contact:", int((scored.Contact_Recommended == "Yes").sum()))
    print(seg.to_string(index=False))
    print(rec.to_string(index=False))


if __name__ == "__main__":
    main()
