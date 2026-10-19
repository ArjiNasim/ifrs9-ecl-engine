"""Unit tests for the ECL engine — the number-production logic must be provable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.ecl_engine import assign_stage, compute_ecl, portfolio_summary


def _frame(**overrides):
    base = dict(
        pd_12m=0.02,
        pd_lifetime=0.10,
        pd_origination=0.02,
        lgd=0.45,
        ead=10_000.0,
        days_past_due=0,
        defaulted=False,
    )
    base.update(overrides)
    return pd.DataFrame([base])


def test_zero_pd_gives_zero_ecl():
    df = _frame(pd_12m=0.0, pd_lifetime=0.0)
    out = compute_ecl(df)
    assert out["ecl"].iloc[0] == 0.0


def test_stage1_uses_12m_pd():
    df = _frame(pd_12m=0.02, pd_lifetime=0.20, pd_origination=0.02)
    out = compute_ecl(df)
    assert out["stage"].iloc[0] == 1
    np.testing.assert_allclose(out["ecl"].iloc[0], 0.02 * 0.45 * 10_000)


def test_defaulted_is_stage3_and_pd_one():
    df = _frame(defaulted=True)
    out = compute_ecl(df)
    assert out["stage"].iloc[0] == 3
    # PD forced to 1 -> ECL = LGD x EAD
    np.testing.assert_allclose(out["ecl"].iloc[0], 0.45 * 10_000)


def test_90dpd_is_stage3():
    assert assign_stage(_frame(days_past_due=95)).iloc[0] == 3


def test_30dpd_backstop_triggers_stage2():
    assert assign_stage(_frame(days_past_due=45)).iloc[0] == 2


def test_relative_sicr_triggers_stage2():
    # PD tripled since origination -> relative test (>= 2x) fires
    df = _frame(pd_12m=0.06, pd_origination=0.02)
    assert assign_stage(df).iloc[0] == 2


def test_absolute_sicr_triggers_stage2():
    df = _frame(pd_12m=0.055, pd_origination=0.02)  # +3.5pp >= 3pp absolute
    assert assign_stage(df).iloc[0] == 2


def test_stage2_uses_lifetime_pd():
    df = _frame(pd_12m=0.06, pd_origination=0.02, pd_lifetime=0.30)
    out = compute_ecl(df)
    assert out["stage"].iloc[0] == 2
    np.testing.assert_allclose(out["ecl"].iloc[0], 0.30 * 0.45 * 10_000)


def test_portfolio_summary_total_matches():
    df = pd.concat([_frame(), _frame(defaulted=True), _frame(days_past_due=45)], ignore_index=True)
    out = compute_ecl(df)
    summary = portfolio_summary(out)
    np.testing.assert_allclose(summary.loc["Total", "ecl"], out["ecl"].sum())
