"""Generate a small SYNTHETIC fixture so the pipeline and tests run out of the box.

IMPORTANT: this is a development smoke-test fixture ONLY. For the real portfolio
project, replace it with public data (Lending Club, Taiwan credit-card, ...).
A reviewer should see real data in notebooks/ and results in the README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pd_orig = rng.beta(2, 30, n)                      # PD at origination
    drift = rng.normal(0, 0.03, n)                    # credit-quality drift since
    pd_12m = np.clip(pd_orig + drift, 1e-4, 0.99)
    pd_lifetime = np.clip(pd_12m * rng.uniform(1.5, 3.0, n), 1e-4, 0.999)
    dpd = rng.choice([0, 0, 0, 15, 45, 120], size=n, p=[0.6, 0.15, 0.1, 0.08, 0.04, 0.03])
    defaulted = ((dpd >= 90) | (rng.random(n) < pd_12m)).astype(int)
    ead = rng.uniform(1_000, 40_000, n).round(2)
    recoveries = (ead * rng.uniform(0, 0.6, n) * defaulted).round(2)
    grade = rng.choice(list("ABCDEFG"), size=n, p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.07, 0.03])
    issue_date = pd.to_datetime("2018-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 4, n), unit="D"
    )
    return pd.DataFrame(
        {
            "loan_id": np.arange(n),
            "issue_date": issue_date,
            "grade": grade,
            "pd_origination": pd_orig,
            "pd_12m": pd_12m,
            "pd_lifetime": pd_lifetime,
            "days_past_due": dpd,
            "defaulted": defaulted,
            "ead": ead,
            "recoveries": recoveries,
            "loan_amount": ead,
            "dti": rng.uniform(0, 40, n).round(2),
            "annual_inc": rng.uniform(20_000, 150_000, n).round(0),
        }
    )


if __name__ == "__main__":
    df = make_sample()
    out = "data/sample_portfolio.parquet"
    df.to_parquet(out)
    print(f"Wrote {len(df)} synthetic exposures to {out} (DEV FIXTURE ONLY).")
