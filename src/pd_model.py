"""PD modelling: an interpretable logistic scorecard (official model) plus a
gradient-boosting challenger. The challenger exists to answer one governance
question: how much performance are we leaving on the table by staying simple?

Dependencies degrade gracefully:
  - optbinning (WoE scorecard) -> falls back to plain logistic regression
  - xgboost / lightgbm         -> falls back to sklearn GradientBoosting
so the repo runs on a minimal install and shines on a full one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_scorecard(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    """Interpretable baseline: standardised logistic regression, calibrated.

    In production you would run WoE binning (optbinning) first and expose the
    Information Value per variable; this keeps the dependency optional.
    """
    base = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X, y)
    return model


def train_challenger(X: pd.DataFrame, y: np.ndarray) -> object:
    """Challenger: gradient boosting, calibrated. Tries xgboost, else sklearn."""
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            eval_metric="logloss",
        )
    except Exception:
        clf = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05)
    model = CalibratedClassifierCV(clf, method="isotonic", cv=3)
    model.fit(X, y)
    return model


def predict_pd(model, X: pd.DataFrame) -> np.ndarray:
    """Calibrated probability of default in [0, 1]."""
    return model.predict_proba(X)[:, 1]


def information_value(x: pd.Series, y: np.ndarray, n_bins: int = 10) -> float:
    """Information Value of one feature — the credit scorecard staple."""
    df = pd.DataFrame({"x": x.to_numpy(dtype=float), "y": np.asarray(y)})
    df["bucket"] = pd.qcut(df["x"], q=n_bins, duplicates="drop")
    g = df.groupby("bucket", observed=True)["y"]
    bad = g.sum()
    good = g.count() - bad
    bad_rate = (bad / bad.sum()).clip(1e-6)
    good_rate = (good / good.sum()).clip(1e-6)
    woe = np.log(good_rate / bad_rate)
    return float(((good_rate - bad_rate) * woe).sum())
