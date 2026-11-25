"""Cost-based threshold, calibration and feature importance for the selected model."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from data_preprocessing import get_clean_data, FEATURES, RESULTS_DIR
from model_training import train_and_compare

# Business assumptions
RETENTION_COST = 20.0   # cost of contacting / making an offer to one customer
SAVE_RATE = 0.40        # share of true churners saved by the offer
# False negative cost = lost customer value; false positive cost = retention cost


def threshold_cost_analysis(pc, yte, value):
    """Total cost = retention cost x customers contacted + value of the churners missed."""
    rows = []
    y = np.asarray(yte)
    for t in np.arange(0.05, 0.96, 0.01):
        pr = pc >= t
        fp = pr & (y == 0)
        fn = ~pr & (y == 1)
        tp = pr & (y == 1)
        cost = RETENTION_COST * pr.sum() + (value * fn).sum()
        saved = (value * tp).sum() * SAVE_RATE
        rows.append(dict(Threshold=round(t, 2), Contacted=int(pr.sum()), TP=int(tp.sum()),
                         FP=int(fp.sum()), FN=int(fn.sum()),
                         Precision=tp.sum() / max(pr.sum(), 1), Recall=tp.sum() / y.sum(),
                         Total_Cost=cost, Net_Benefit=saved - RETENTION_COST * pr.sum()))
    tc = pd.DataFrame(rows).round(4)
    thr = float(tc.loc[tc.Total_Cost.idxmin(), "Threshold"])
    tc["Recommended"] = np.where(tc.Threshold == thr, "Yes", "No")
    return tc, thr


def calibration_table(yte, pc, n_bins=8):
    fr, mp = calibration_curve(yte, pc, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"Predicted_Probability": mp.round(4), "Actual_Churn_Rate": fr.round(4)})


def feature_importance(model, Xte, yte):
    pi = permutation_importance(model, Xte, yte, scoring="average_precision", n_repeats=10, random_state=0)
    return (pd.DataFrame({"Feature": FEATURES, "Importance": pi.importances_mean.round(4)})
            .sort_values("Importance", ascending=False))


def evaluate(res):
    """Run all evaluation steps on the output of model_training.train_and_compare."""
    bm = res["models"][res["best"]]
    _, pc = res["cal"][res["best"]]
    Xte, yte = res["Xte"], res["yte"]
    tc, thr = threshold_cost_analysis(pc, yte, Xte.Customer_Value.values)
    tn, fp, fn, tp = confusion_matrix(yte, (pc >= thr).astype(int)).ravel()
    return dict(threshold_cost=tc, threshold=thr, calibration=calibration_table(yte, pc),
                importance=feature_importance(bm, Xte, yte),
                confusion=dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)))


def main():
    _, df = get_clean_data()
    res = train_and_compare(df)
    ev = evaluate(res)
    RESULTS_DIR.mkdir(exist_ok=True)
    ev["threshold_cost"].to_csv(RESULTS_DIR / "threshold_cost_analysis.csv", index=False)
    ev["calibration"].to_csv(RESULTS_DIR / "calibration_results.csv", index=False)
    ev["importance"].to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    print("Cost-optimal threshold:", ev["threshold"])
    print("Confusion matrix at that threshold:", ev["confusion"])
    print(ev["calibration"].to_string(index=False))
    print(ev["importance"].to_string(index=False))


if __name__ == "__main__":
    main()
