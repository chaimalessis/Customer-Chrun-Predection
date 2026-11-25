"""Train and compare the four models (guideline steps 5-8)."""
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from data_preprocessing import get_clean_data, FEATURES, RESULTS_DIR

warnings.filterwarnings("ignore")
SEED = 42


def split(df):
    """Separate the target and create the stratified 75/25 train and test sets."""
    X, y = df[FEATURES], df.Churn
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    return X, y, Xtr, Xte, ytr, yte


def build_models(ytr):
    """Logistic Regression baseline + three tree-based models, all weighted for class imbalance."""
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    return {
        "Logistic Regression (baseline)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced",
                                                random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                                 random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, class_weight="balanced", random_state=SEED,
                                   verbose=-1),
    }


def train_and_compare(df):
    """Fit every model, add an isotonic-calibrated version, and compare them at threshold 0.5."""
    X, y, Xtr, Xte, ytr, yte = split(df)
    models = build_models(ytr)
    rows, cal = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        pr = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pr).ravel()
        cm = CalibratedClassifierCV(m, method="isotonic", cv=5).fit(Xtr, ytr)
        pc = cm.predict_proba(Xte)[:, 1]
        cal[name] = (cm, pc)
        rows.append(dict(Model=name, Precision=precision_score(yte, pr), Recall=recall_score(yte, pr),
                         F1=f1_score(yte, pr), ROC_AUC=roc_auc_score(yte, p),
                         PR_AUC=average_precision_score(yte, p), Brier_Raw=brier_score_loss(yte, p),
                         Brier_Calibrated=brier_score_loss(yte, pc), TP=tp, FP=fp, FN=fn, TN=tn))
    mc = pd.DataFrame(rows).round(4)
    best = mc.sort_values(["PR_AUC", "ROC_AUC"], ascending=False).iloc[0].Model
    mc["Selected"] = np.where(mc.Model == best, "Yes", "No")
    return dict(models=models, cal=cal, comparison=mc, best=best, X=X, y=y,
                Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)


def main():
    _, df = get_clean_data()
    res = train_and_compare(df)
    RESULTS_DIR.mkdir(exist_ok=True)
    res["comparison"].to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(res["comparison"].to_string(index=False))
    print("Selected model:", res["best"])


if __name__ == "__main__":
    main()
