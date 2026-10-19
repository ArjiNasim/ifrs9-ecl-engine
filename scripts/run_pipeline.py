"""Smoke-test the whole pipeline end to end on the dev fixture.

Run:  python scripts/run_pipeline.py
Produces data/portfolio_with_ecl.parquet and prints the committee summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.make_sample_data import make_sample
from src.ecl_engine import compute_ecl, portfolio_summary
from src.lgd_ead import estimate_ead, estimate_lgd
from src.macro_overlay import weighted_ecl
from src.validation import validation_report


def main() -> None:
    df = make_sample()
    df["lgd"] = estimate_lgd(df, recovery_col="recoveries", exposure_col="ead")
    df["ead"] = estimate_ead(df, outstanding_col="ead")

    out = compute_ecl(df)
    print("\n=== Portfolio ECL by stage ===")
    print(portfolio_summary(out).round(2).to_string())

    print("\n=== ECL by macro scenario ===")
    print(weighted_ecl(df).round(2).to_string(index=False))

    print("\n=== PD validation (on fixture PDs) ===")
    print(validation_report(df["defaulted"].to_numpy(), df["pd_12m"].to_numpy()))

    out.to_parquet("data/portfolio_with_ecl.parquet")
    print("\nWrote data/portfolio_with_ecl.parquet")


if __name__ == "__main__":
    main()
