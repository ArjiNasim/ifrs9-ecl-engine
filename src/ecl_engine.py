"""IFRS 9 ECL engine: staging, SICR classification and expected credit loss.

This module is the core of the project. It is deliberately dependency-light
(pandas + numpy only) and fully unit-tested so the "number production" logic
can be trusted and challenged independently of the models that feed it.

Key formulas
------------
ECL = PD x LGD x EAD
  - Stage 1 : 12-month PD
  - Stage 2 : lifetime PD
  - Stage 3 : defaulted -> PD = 1, so ECL = LGD x EAD

Staging (IFRS 9)
----------------
  - Stage 3 : asset is in default (flag or >= 90 days past due)
  - Stage 2 : Significant Increase in Credit Risk (SICR) since origination
  - Stage 1 : everything else
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Column names expected on the input frame. Kept in one place so a change
# in the data layer is a one-line fix here.
COL_PD_12M = "pd_12m"
COL_PD_LIFETIME = "pd_lifetime"
COL_PD_ORIGINATION = "pd_origination"
COL_LGD = "lgd"
COL_EAD = "ead"
COL_DPD = "days_past_due"
COL_DEFAULTED = "defaulted"


def assign_stage(
    df: pd.DataFrame,
    sicr_relative_threshold: float = 2.0,
    sicr_absolute_threshold: float = 0.03,
    dpd_backstop_stage2: int = 30,
    dpd_default: int = 90,
) -> pd.Series:
    """Classify each exposure into IFRS 9 Stage 1, 2 or 3.

    SICR (Stage 2) triggers if ANY of:
      - current 12m PD >= relative_threshold x origination PD (relative test)
      - current 12m PD - origination PD >= absolute_threshold  (absolute test)
      - days past due >= dpd_backstop_stage2                   (30dpd backstop)

    The thresholds are parameters on purpose: the whole point of the dashboard
    is to show how provisions move when a committee tightens or loosens them.
    """
    pd_now = df[COL_PD_12M].to_numpy(dtype=float)
    pd_orig = df[COL_PD_ORIGINATION].to_numpy(dtype=float)
    dpd = df[COL_DPD].to_numpy(dtype=float)
    defaulted = df[COL_DEFAULTED].to_numpy(dtype=bool)

    is_stage3 = defaulted | (dpd >= dpd_default)

    with np.errstate(divide="ignore", invalid="ignore"):
        relative_ratio = np.where(pd_orig > 0, pd_now / pd_orig, np.inf)
    sicr = (
        (relative_ratio >= sicr_relative_threshold)
        | ((pd_now - pd_orig) >= sicr_absolute_threshold)
        | (dpd >= dpd_backstop_stage2)
    )

    stage = np.ones(len(df), dtype=int)
    stage = np.where(sicr, 2, stage)
    stage = np.where(is_stage3, 3, stage)
    return pd.Series(stage, index=df.index, name="stage")


def compute_ecl(df: pd.DataFrame, stage: pd.Series | None = None) -> pd.DataFrame:
    """Return the input frame with `stage` and `ecl` columns added.

    ECL uses the 12-month PD in Stage 1 and the lifetime PD in Stage 2.
    In Stage 3 the asset is in default, so PD is forced to 1 and
    ECL collapses to LGD x EAD.
    """
    out = df.copy()
    if stage is None:
        stage = assign_stage(out)
    out["stage"] = stage.to_numpy()

    pd_used = np.where(
        out["stage"].to_numpy() == 1,
        out[COL_PD_12M].to_numpy(dtype=float),
        out[COL_PD_LIFETIME].to_numpy(dtype=float),
    )
    pd_used = np.where(out["stage"].to_numpy() == 3, 1.0, pd_used)

    out["pd_used"] = pd_used
    out["ecl"] = pd_used * out[COL_LGD].to_numpy(dtype=float) * out[COL_EAD].to_numpy(dtype=float)
    return out


def portfolio_summary(df_with_ecl: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ECL and exposure by stage, the headline committee table."""
    g = df_with_ecl.groupby("stage")
    summary = pd.DataFrame(
        {
            "n_exposures": g.size(),
            "ead": g[COL_EAD].sum(),
            "ecl": g["ecl"].sum(),
        }
    )
    summary["coverage_ratio"] = summary["ecl"] / summary["ead"].replace(0, np.nan)
    total = summary[["n_exposures", "ead", "ecl"]].sum()
    total["coverage_ratio"] = total["ecl"] / total["ead"] if total["ead"] else np.nan
    summary.loc["Total"] = total
    return summary
