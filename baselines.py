"""Logistic regression (classic proxy-means test) and gradient boosting on the same split as train.py.

    python baselines.py --fold 0
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

import config as C
from features import Preprocessor, make_split


def recall_at_precision(y, p, w, target=0.7):
    prec, rec, _ = precision_recall_curve(y, p, sample_weight=w)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.TABLE_PATH))
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    tr_idx, va_idx = make_split(df, a.fold)
    tr, va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
    pre = Preprocessor().fit(tr)

    def design(d):
        xn, xc = pre.transform(d)
        onehot = [np.eye(card)[xc[:, i]] for i, card in enumerate(pre.cardinalities)]
        return np.hstack([xn] + onehot), np.hstack([xn, xc.astype(np.float32)])

    (lin_tr, gb_tr), (lin_va, gb_va) = design(tr), design(va)
    y_tr, y_va = tr[C.POVERTY].to_numpy(), va[C.POVERTY].to_numpy()
    w_tr, w_va = tr[C.WEIGHT].to_numpy(), va[C.WEIGHT].to_numpy()
    cat_mask = [False] * pre.n_numeric + [True] * len(pre.cat_cols)

    models = {
        "logistic_regression": (LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"), lin_tr, lin_va),
        "gradient_boosting": (HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=C.SEED), gb_tr, gb_va),
    }
    for name, (m, xtr, xva) in models.items():
        m.fit(xtr, y_tr, sample_weight=w_tr)
        p = m.predict_proba(xva)[:, 1]
        print(f"{name:<22} PR-AUC={average_precision_score(y_va, p, sample_weight=w_va):.4f}  "
              f"recall@P0.7={recall_at_precision(y_va, p, w_va):.4f}")


if __name__ == "__main__":
    main()
