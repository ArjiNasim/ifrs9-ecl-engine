"""Model validation / challenge metrics — the second-line-of-defence layer.

Everything a model-risk reviewer would compute to decide whether a PD model
can be trusted: discrimination, calibration, stability and backtesting.
Pure numpy/pandas so it can be run against any model's scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def auc_gini(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """AUC via the Mann-Whitney U relation, plus Gini = 2*AUC - 1."""
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc), float(2 * auc - 1)


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max separation between good/bad cumulative curves."""
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(score, dtype=float)})
    df = df.sort_values("s")
    cum_bad = (df["y"] == 1).cumsum() / max((df["y"] == 1).sum(), 1)
    cum_good = (df["y"] == 0).cumsum() / max((df["y"] == 0).sum(), 1)
    return float(np.abs(cum_bad - cum_good).max())


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between a reference (train) and a new (test/live) distribution.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
    The classic model-monitoring signal.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    e_perc = np.histogram(expected, bins=quantiles)[0] / len(expected)
    a_perc = np.histogram(actual, bins=quantiles)[0] / len(actual)
    eps = 1e-6
    e_perc = np.clip(e_perc, eps, None)
    a_perc = np.clip(a_perc, eps, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def calibration_table(y_true: np.ndarray, pd_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate by PD bucket (calibration check)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(pd_pred, dtype=float)})
    df["bucket"] = pd.qcut(df["pd"], q=n_bins, duplicates="drop")
    g = df.groupby("bucket", observed=True)
    out = pd.DataFrame(
        {
            "n": g.size(),
            "pd_predicted": g["pd"].mean(),
            "default_rate_observed": g["y"].mean(),
        }
    )
    out["gap"] = out["pd_predicted"] - out["default_rate_observed"]
    return out.reset_index()


def backtest_by_grade(df: pd.DataFrame, grade_col: str, y_col: str, pd_col: str) -> pd.DataFrame:
    """Predicted vs realised default rate per rating grade / cohort."""
    g = df.groupby(grade_col, observed=True)
    out = pd.DataFrame(
        {
            "n": g.size(),
            "pd_predicted": g[pd_col].mean(),
            "default_rate_observed": g[y_col].mean(),
        }
    )
    out["gap"] = out["pd_predicted"] - out["default_rate_observed"]
    return out.reset_index()


def validation_report(
    y_true: np.ndarray,
    pd_pred: np.ndarray,
    pd_pred_train: np.ndarray | None = None,
) -> dict:
    """One-call summary a reviewer can drop into a committee pack."""
    auc, gini = auc_gini(y_true, pd_pred)
    report = {
        "auc": round(auc, 4),
        "gini": round(gini, 4),
        "ks": round(ks_statistic(y_true, pd_pred), 4),
        "mean_pd_predicted": round(float(np.mean(pd_pred)), 4),
        "default_rate_observed": round(float(np.mean(y_true)), 4),
    }
    if pd_pred_train is not None:
        report["psi_train_vs_test"] = round(
            population_stability_index(pd_pred_train, pd_pred), 4
        )
    return report
