"""Lending Club-specific data layer for the IFRS 9 ECL engine.

The generic `data_prep.py` stays schema-agnostic; this module knows the real
Lending Club column names and turns a raw export into the exact inputs the
ECL engine expects (pd_12m, pd_lifetime, pd_origination, lgd, ead,
days_past_due, defaulted).

Dataset: Kaggle `wordsforthewise/lending-club` (accepted loans), or any
Lending Club "accepted" CSV. Default is fixed as Charged Off / Default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- Default definition (fixed up front, held constant) --------------------
LC_DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
LC_PAID_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

# Origination features available at underwriting time (no leakage).
LC_FEATURE_COLS = [
    "loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low",
    "revol_util", "open_acc", "pub_rec", "delinq_2yrs", "inq_last_6mths",
    "term_months", "emp_length_years",
]


def load_lending_club(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Load an accepted-loans CSV. Use nrows to sample on a small machine."""
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_deepchecks() -> pd.DataFrame:
    """Tokenless Lending Club sample bundled with the `deepchecks` library.

    No Kaggle account needed: `pip install deepchecks` then call this. It is a
    curated subset (fewer columns, ~10k+ loans) — enough for a real-data v1.
    Recovery columns may be absent, so LGD falls back to the 45% segment
    assumption (handled in estimate_lgd_lc).
    """
    from deepchecks.tabular.datasets.classification import lending_club

    df = lending_club.load_data(data_format="Dataframe", as_train_test=False)
    df = df.reset_index()
    df.columns = [str(c).strip() for c in df.columns]
    # deepchecks schema has `sub_grade` but no `grade`; derive it (A1 -> A).
    if "grade" not in df.columns and "sub_grade" in df.columns:
        df["grade"] = df["sub_grade"].astype(str).str[0]
    # loan_status is the numeric 0/1 target; polarity is auto-detected in
    # clean_and_label, so it is left untouched here.
    return df


def clean_and_label(df: pd.DataFrame, completed_only: bool = True) -> pd.DataFrame:
    """Parse rates/terms, define default, drop live loans for a clean label.

    completed_only keeps only Fully Paid vs Charged Off/Default, so the target
    is actually observed — never train a PD model on `Current` loans.
    """
    out = df.copy()

    # Percent strings -> floats
    for col in ("int_rate", "revol_util"):
        if col in out and out[col].dtype == object:
            out[col] = out[col].astype(str).str.rstrip("%").replace("nan", np.nan).astype(float)

    # term " 36 months" -> 36
    if "term" in out:
        out["term_months"] = out["term"].astype(str).str.extract(r"(\d+)").astype(float)

    # emp_length "10+ years" -> 10, "< 1 year" -> 0
    if "emp_length" in out:
        out["emp_length_years"] = (
            out["emp_length"].astype(str).str.extract(r"(\d+)").astype(float).fillna(0)
        )

    # issue_d -> datetime. Handles "Dec-2015", already-datetime, or other formats.
    if "issue_d" in out:
        issue = pd.to_datetime(out["issue_d"], format="%b-%Y", errors="coerce")
        if issue.isna().all():
            issue = pd.to_datetime(out["issue_d"], errors="coerce")
        out["issue_date"] = issue

    # Derive grade from sub_grade if the source lacks it (deepchecks schema).
    if "grade" not in out.columns and "sub_grade" in out.columns:
        out["grade"] = out["sub_grade"].astype(str).str[0]

    if pd.api.types.is_numeric_dtype(out["loan_status"]):
        # Already-binary target (deepchecks). Auto-detect polarity: the default
        # rate must rise from grade A to grade G. If the 1-class decreases with
        # worse grade, it means "good", so default is the 0-class.
        t = out["loan_status"].astype(int)
        order = {g: i for i, g in enumerate("ABCDEFG")}
        gi = out.get("grade", pd.Series("D", index=out.index)).map(order)
        gi = gi.fillna(gi.mean())
        corr = np.corrcoef(gi, t)[0, 1] if gi.std() > 0 else 1.0
        out["defaulted"] = (t if corr >= 0 else 1 - t).astype(int)
        # every deepchecks row is a resolved loan, so no completed-only filter
    else:
        out["defaulted"] = out["loan_status"].isin(LC_DEFAULT_STATUSES).astype(int)
        if completed_only:
            keep = out["loan_status"].isin(LC_DEFAULT_STATUSES | LC_PAID_STATUSES)
            out = out[keep].copy()

    return out


def days_past_due_from_status(df: pd.DataFrame) -> pd.Series:
    """Approximate days-past-due from the loan status (for staging/SICR)."""
    mapping = {
        "Current": 0,
        "Fully Paid": 0,
        "In Grace Period": 10,
        "Late (16-30 days)": 23,
        "Late (31-120 days)": 75,
        "Default": 120,
        "Charged Off": 120,
    }
    return df["loan_status"].map(mapping).fillna(0).astype(int).rename("days_past_due")


def temporal_split(df: pd.DataFrame, cutoff: str = "2016-01-01"):
    """Train on loans issued before cutoff, test on/after — never random."""
    train = df[df["issue_date"] < cutoff].copy()
    test = df[df["issue_date"] >= cutoff].copy()
    return train, test


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature matrix (origination-time only), robustly imputed.

    Drops any feature that is entirely missing in this slice, median-imputes the
    rest, and backstops with 0 so no NaN can ever reach the model (real Lending
    Club exports have sparse columns that would otherwise break the fit).
    """
    cols = [c for c in LC_FEATURE_COLS if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")                 # drop all-NaN columns
    X = X.fillna(X.median(numeric_only=True))       # median impute
    return X.fillna(0)                              # backstop


def estimate_lgd_lc(df: pd.DataFrame, fallback: float = 0.45) -> pd.Series:
    """LGD = 1 - recoveries / exposure-at-default.

    EAD-at-default ~ principal outstanding at charge-off =
    funded_amnt - total_rec_prncp. For non-defaulted loans LGD is not observed,
    so a documented segment fallback (Basel-style 45%) is used.
    """
    # Recovery-based LGD needs the recoveries column. If the source doesn't
    # carry it (e.g. the reduced deepchecks sample), use the documented
    # segment fallback for the whole book rather than inventing LGD = 100%.
    if "recoveries" not in df.columns:
        return pd.Series(fallback, index=df.index, name="lgd")
    funded = df.get("funded_amnt", df.get("loan_amnt"))
    rec_prncp = df.get("total_rec_prncp", pd.Series(0, index=df.index))
    recoveries = df["recoveries"]
    ead_default = (funded - rec_prncp).clip(lower=1)
    lgd = 1.0 - (recoveries / ead_default)
    lgd = lgd.where(df["defaulted"] == 1, other=fallback)
    return lgd.clip(0, 1).rename("lgd")


def estimate_ead_lc(df: pd.DataFrame) -> pd.Series:
    """EAD ~ current outstanding principal = funded_amnt - principal repaid."""
    funded = df.get("funded_amnt", df.get("loan_amnt"))
    rec_prncp = df.get("total_rec_prncp", pd.Series(0, index=df.index))
    return (funded - rec_prncp).clip(lower=0).rename("ead")


def pd_origination_from_grade(df: pd.DataFrame, default_rate_by_grade: dict) -> pd.Series:
    """Proxy for PD at origination: the historical default rate of the loan's
    grade at issuance. Used by the SICR relative test. Documented simplification
    for a static snapshot that has no per-loan PD time series."""
    return df["grade"].map(default_rate_by_grade).fillna(
        np.mean(list(default_rate_by_grade.values()))
    ).rename("pd_origination")
