"""Streamlit dashboard: four tabs — Portfolio, Scenarios, Counterparty, Monitoring.

Run:  streamlit run app/streamlit_app.py
Reads the figured outputs; recomputes staging/scenarios reactively so a
committee can see provisions move. Skeleton — wire real model outputs in.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from scripts.make_sample_data import make_sample
from src.ecl_engine import assign_stage, compute_ecl, portfolio_summary
from src.lgd_ead import estimate_lgd
from src.macro_overlay import ScenarioSet, Scenario, weighted_ecl
from src.validation import population_stability_index, validation_report

st.set_page_config(page_title="IFRS 9 ECL Engine", layout="wide")


@st.cache_data
def load_portfolio() -> pd.DataFrame:
    path = Path("data/portfolio_with_ecl.parquet")
    df = pd.read_parquet(path) if path.exists() else make_sample()
    if "lgd" not in df:
        df["lgd"] = estimate_lgd(df, recovery_col="recoveries", exposure_col="ead")
    return df


df = load_portfolio()
st.title("IFRS 9 ECL Engine & Model-Challenge Dashboard")

tab_pf, tab_sc, tab_cp, tab_mon = st.tabs(
    ["Portfolio", "Scenarios", "Counterparty", "Monitoring"]
)

# --- Portfolio -------------------------------------------------------------
with tab_pf:
    out = compute_ecl(df)
    summary = portfolio_summary(out)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total ECL", f"{out['ecl'].sum():,.0f}")
    c2.metric("Total EAD", f"{out['ead'].sum():,.0f}")
    c3.metric("Coverage", f"{out['ecl'].sum() / out['ead'].sum():.2%}")
    st.subheader("ECL by stage")
    st.dataframe(summary.round(2))
    if "grade" in out:
        st.subheader("ECL by grade")
        st.bar_chart(out.groupby("grade")["ecl"].sum())

# --- Scenarios -------------------------------------------------------------
with tab_sc:
    st.subheader("SICR sensitivity and macro scenarios")
    rel = st.slider("SICR relative threshold (x origination PD)", 1.1, 5.0, 2.0, 0.1)
    out_s = compute_ecl(df, assign_stage(df, sicr_relative_threshold=rel))
    st.metric("Total ECL at this SICR threshold", f"{out_s['ecl'].sum():,.0f}")
    st.caption("Lower threshold -> more Stage 2 -> higher provisions.")

    w_adv = st.slider("Adverse scenario weight", 0.0, 1.0, 0.3, 0.05)
    sset = ScenarioSet(
        [Scenario("baseline", 1.0, 1 - w_adv - 0.2),
         Scenario("adverse", 1.6, w_adv),
         Scenario("upside", 0.8, 0.2)]
    )
    st.dataframe(weighted_ecl(df, sset).round(0))

# --- Counterparty ----------------------------------------------------------
with tab_cp:
    st.subheader("Single-exposure drill-down")
    out = compute_ecl(df)
    idx = st.number_input("Row index", 0, len(out) - 1, 0)
    row = out.iloc[int(idx)]
    st.json(
        {
            "stage": int(row["stage"]),
            "pd_used": round(float(row["pd_used"]), 4),
            "lgd": round(float(row["lgd"]), 4),
            "ead": round(float(row["ead"]), 2),
            "ecl": round(float(row["ecl"]), 2),
        }
    )
    st.caption("Wire a SHAP waterfall of the challenger model here.")

# --- Monitoring ------------------------------------------------------------
with tab_mon:
    st.subheader("Model monitoring")
    rep = validation_report(df["defaulted"].to_numpy(), df["pd_12m"].to_numpy())
    st.json(rep)
    psi = population_stability_index(df["pd_12m"].sample(frac=0.5, random_state=1),
                                     df["pd_12m"].sample(frac=0.5, random_state=2))
    flag = "OK" if psi < 0.1 else ("WATCH" if psi < 0.25 else "ALERT")
    st.metric("PSI (demo split)", f"{psi:.3f}", flag)
