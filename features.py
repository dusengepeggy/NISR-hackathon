"""Preprocessing and cluster-grouped splits. Everything is fit on the training fold only."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config as C


def make_split(df, fold=0, n_folds=C.N_FOLDS, seed=C.SEED):
    """Stratified on poverty, grouped on enumeration area so no cluster spans train and validation."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(sgkf.split(df, df[C.POVERTY], groups=df[C.CLUSTER]))
    tr, va = splits[fold]
    assert set(df.iloc[tr][C.CLUSTER]).isdisjoint(df.iloc[va][C.CLUSTER]), "cluster leakage"
    return tr, va


class Preprocessor:
    def fit(self, df, min_count=C.MIN_CATEGORY_COUNT):
        feats = [c for c in df.columns if c not in C.NON_FEATURES]
        self.cat_cols = [c for c in feats if df[c].dtype == object or str(df[c].dtype) in ("category", "string", "str")]
        self.num_cols = [c for c in feats if c not in self.cat_cols]
        # numeric
        X = df[self.num_cols].astype(float)
        self.median = X.median()
        self.miss_cols = [c for c in self.num_cols if X[c].isna().any()]
        self.log_cols = [c for c in self.num_cols if X[c].min() >= 0 and X[c].skew() > 2]
        Xt = self._numeric_raw(df)
        self.mean, self.std = Xt.mean(), Xt.std().replace(0, 1)
        # categorical: index 0 = unseen/rare ("other")
        self.vocab = {}
        for c in self.cat_cols:
            counts = df[c].astype(str).value_counts()
            keep = counts[counts >= min_count].index
            self.vocab[c] = {v: i + 1 for i, v in enumerate(keep)}
        self.cardinalities = [len(self.vocab[c]) + 1 for c in self.cat_cols]
        return self

    def _numeric_raw(self, df):
        X = df[self.num_cols].astype(float)
        flags = {f"{c}__missing": X[c].isna().astype(float) for c in self.miss_cols}
        X = X.fillna(self.median)
        for c in self.log_cols:
            X[c] = np.log1p(X[c].clip(lower=0))
        return pd.concat([X, pd.DataFrame(flags, index=df.index)], axis=1)

    def transform(self, df):
        Xn = self._numeric_raw(df)
        if hasattr(self, "mean"):
            Xn = (Xn - self.mean) / self.std
        x_num = Xn.to_numpy(dtype=np.float32)
        x_cat = np.stack([df[c].astype(str).map(self.vocab[c]).fillna(0).to_numpy(dtype=np.int64)
                          for c in self.cat_cols], axis=1) if self.cat_cols else np.zeros((len(df), 0), np.int64)
        return x_num, x_cat

    @property
    def n_numeric(self):
        return len(self.num_cols) + len(self.miss_cols)
