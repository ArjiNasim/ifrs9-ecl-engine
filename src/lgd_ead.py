"""LGD and EAD estimation.

Both parameters are usually the weakest part of an ECL model, so the
assumptions are made explicit and easy to stress separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Basel foundation-IRB style fallback LGD for unsecured exposures.
DEFAULT_UNSECURED_LGD = 0.45


def estimate_lgd(
    df: pd.DataFrame,
    recovery_col: str | None = None,
    exposure_col: str = "ead",
    fallback_lgd: float = DEFAULT_UNSECURED_LGD,
) -> pd.Series:
    """LGD = 1 - (recoveries / exposure) when recoveries exist, else a
    documented segment assumption. Clipped to [0, 1]."""
    if recovery_col and recovery_col in df.columns:
        lgd = 1.0 - (df[recovery_col].to_numpy(dtype=float) / df[exposure_col].replace(0, np.nan))
        lgd = pd.Series(lgd, index=df.index).fillna(fallback_lgd).clip(0, 1)
    else:
        lgd = pd.Series(fallback_lgd, index=df.index)
    return lgd.rename("lgd")


def estimate_ead(
    df: pd.DataFrame,
    outstanding_col: str | None = None,
    limit_col: str | None = None,
    ccf: float = 0.5,
) -> pd.Series:
    """EAD = outstanding balance, plus CCF x undrawn limit when a limit exists."""
    if outstanding_col and outstanding_col in df.columns:
        ead = df[outstanding_col].to_numpy(dtype=float)
        if limit_col and limit_col in df.columns:
            undrawn = np.clip(df[limit_col].to_numpy(dtype=float) - ead, 0, None)
            ead = ead + ccf * undrawn
    elif limit_col and limit_col in df.columns:
        ead = df[limit_col].to_numpy(dtype=float)
    else:
        raise ValueError("Provide outstanding_col and/or limit_col to estimate EAD.")
    return pd.Series(ead, index=df.index, name="ead")
