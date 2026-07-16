"""
TCC Bangladesh — interactive heat-stress dashboard.

Reads the pipeline outputs in ../outputs directly and renders district-level
projections, validation, trends, and risk categories interactively.

Run:  py -3.14 -m streamlit run dashboard/app.py
"""
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
O = os.path.join(ROOT, "outputs")
FIG = os.path.join(O, "figures")
FC = os.path.join(O, "forecasts")

from tcc_pipeline import choropleth as ch
from tcc_pipeline import config as C

st.set_page_config(page_title="TCC Bangladesh Heat-Stress Dashboard",
                   page_icon="🌡️", layout="wide")

CONDITIONS = {  # display -> forecast column
    "Heat index (°C)": "heat_index",
    "Wet-bulb temperature (°C)": "wet_bulb",
    "WBGT — sun (°C)": "wbgt_sun",
    "WBGT — shade (°C)": "wbgt_shade",
    "Cooling degree-days (°C·d/d)": "cdd",
    "UV index": "uv",
    "Dew point (°C)": "dew_point",
}
TREND_VARS = {
    "Maximum temperature": "tmax", "Mean temperature": "temperature",
    "Minimum temperature": "tmin", "Dew point": "dew_point",
    "Relative humidity": "humidity", "Heat index": "heat_index",
    "Wet-bulb temperature": "wet_bulb", "WBGT (shade)": "wbgt_shade",
}


# --------------------------------------------------------------------------- #
@st.cache_data
def load_zones():
    z = pd.read_csv(os.path.join(ROOT, "tcc_pipeline", "assets", "zones.csv"))
    return z


@st.cache_data
def load_csv(rel):
    p = os.path.join(O, rel)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


@st.cache_data
def load_forecast(district):
    fc = pd.read_csv(os.path.join(FC, f"{district}_forecast.csv"))
    fc["date"] = pd.to_datetime(fc["date"])
    return fc


@st.cache_data
def district_means():
    rows = {}
    for f in os.listdir(FC):
        d = f.replace("_forecast.csv", "")
        fc = pd.read_csv(os.path.join(FC, f))
        rows[d] = {c: float(fc[c].mean()) for c in CONDITIONS.values() if c in fc.columns}
    return pd.DataFrame(rows).T


@st.cache_resource
def load_geoms():
    return ch.load_district_geometry()


zones = load_zones()
DISTRICTS = sorted(zones["district"])
ZONE_OF = dict(zip(zones["district"], zones["zone_name"]))
EXEMPLARS = set(zones[zones["is_exemplar"]]["district"])

st.sidebar.title("🌡️ TCC Bangladesh")
st.sidebar.caption("District-level heat-stress projection & validation")
page = st.sidebar.radio("View", ["Overview", "District explorer",
                                 "Condition maps", "Observed trends",
                                 "Validation & downloads"])


# --------------------------------------------------------------------------- #
def overview():
    st.title("Heat-stress projection for the 64 districts of Bangladesh")
    st.caption("Seven Temperature-Correlated Conditions · 1980–2024 baseline · "
               "2025–2026 projection · validated against observed 2025")

    ev = load_csv("metrics/external_validation_2025.csv")
    tr = load_csv("trends/mann_kendall_sen_1980_2024.csv")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts", "64")
    c2.metric("Heat-stress conditions", "7")
    if not ev.empty:
        hi = ev[ev.variable == "heat_index"]["val_R2"].median()
        c3.metric("Heat-index validation R² (median)", f"{hi:.3f}")
    if not tr.empty:
        tx = tr[tr.variable == "tmax"]["sen_slope_per_year"].median() * 10
        c4.metric("Max-temp trend (median)", f"+{tx:.2f} °C/decade")

    st.divider()
    a, b = st.columns([1, 1.1])
    with a:
        p = os.path.join(FIG, "map_zonation.png")
        if os.path.exists(p):
            st.image(p, caption="Six climatic zones", width="stretch")
    with b:
        st.subheader("What this shows")
        st.markdown(
            "- **District explorer** — per-district 24-month projection with "
            "confidence bands, 2025 validation, and risk-category breakdown.\n"
            "- **Condition maps** — choropleth of any of the seven conditions "
            "across all 64 districts.\n"
            "- **Observed trends** — 1980–2024 Mann–Kendall/Sen warming by district.\n"
            "- **Validation & downloads** — accuracy tables and raw-data export.")
        st.info("Skill is reported honestly: at monthly resolution the models "
                "match climatology. Their value is spatially complete, "
                "uncertainty-aware, trend-consistent projection — not lower "
                "monthly error.")


def district_explorer():
    st.title("District explorer")
    d = st.selectbox("District", DISTRICTS,
                     index=DISTRICTS.index("Dhaka") if "Dhaka" in DISTRICTS else 0)
    st.caption(f"Zone: {ZONE_OF.get(d, '—')}" + ("  ·  exemplar" if d in EXEMPLARS else ""))
    fc = load_forecast(d)

    left, right = st.columns([1.3, 1])
    with left:
        cond = st.selectbox("Condition", list(CONDITIONS.keys()))
        col = CONDITIONS[cond]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(fc["date"], fc[col], "o-", color="#c0392b", lw=1.8, ms=3, label="Projected")
        lo95, hi95 = f"{col}_lo95", f"{col}_hi95"
        if lo95 in fc.columns:
            ax.fill_between(fc["date"], fc[lo95], fc[hi95], color="#c0392b",
                            alpha=0.15, label="95% CI")
            ax.fill_between(fc["date"], fc[f"{col}_lo68"], fc[f"{col}_hi68"],
                            color="#c0392b", alpha=0.28, label="68% CI")
        ax.set_title(f"{d} — projected {cond}"); ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=8); fig.tight_layout()
        st.pyplot(fig)

        tp = os.path.join(FIG, f"threepanel_{d}_heat_index.png")
        if os.path.exists(tp):
            st.image(tp, caption="Heat-index record · 2025 validation · 2026 projection",
                     width="stretch")
    with right:
        st.subheader("2025 validation")
        ev = load_csv("metrics/external_validation_2025.csv")
        sub = ev[ev.district == d][["variable", "val_R2", "val_RMSE", "val_MAE"]]
        st.dataframe(sub.set_index("variable").round(3), width="stretch", height=330)

        st.subheader("Projected risk months / year")
        thr = {"heat_index": (32, "Heat index ≥32 (Ext. Caution)"),
               "wbgt_sun": (27.8, "WBGT-sun ≥27.8 (Moderate)"),
               "uv": (8, "UV ≥8 (Very High)"),
               "dew_point": (24, "Dew pt ≥24 (Extreme Oppr.)")}
        y25 = fc[fc["date"].dt.year == 2025]
        rows = []
        for c, (t, lab) in thr.items():
            if c in y25.columns:
                rows.append({"Hazard": lab, "Months/yr": int((y25[c] >= t).sum())})
        st.dataframe(pd.DataFrame(rows).set_index("Hazard"), width="stretch")


def condition_maps():
    st.title("Condition maps — projected 2025–2026 annual mean")
    cond = st.selectbox("Condition", list(CONDITIONS.keys()))
    col = CONDITIONS[cond]
    geoms = load_geoms()
    dm = district_means()
    vals = dm[col].to_dict()

    a, b = st.columns([1, 1])
    with a:
        fig, ax = plt.subplots(figsize=(6.5, 7.5))
        sm = ch.choropleth(ax, geoms, vals, cmap="YlOrRd", title=cond)
        fig.colorbar(sm, ax=ax, shrink=0.7)
        st.pyplot(fig)
    with b:
        st.subheader("Ranked by projected annual mean")
        s = pd.Series(vals).sort_values(ascending=False)
        s.index.name = "district"
        st.dataframe(s.round(2).rename("value").to_frame(),
                     width="stretch", height=560)


def observed_trends():
    st.title("Observed warming trends, 1980–2024")
    name = st.selectbox("Variable", list(TREND_VARS.keys()))
    var = TREND_VARS[name]
    tr = load_csv("trends/mann_kendall_sen_1980_2024.csv")
    sub = tr[tr.variable == var].copy()
    sub["slope/decade"] = sub["sen_slope_per_year"] * 10
    vals = dict(zip(sub["district"], sub["slope/decade"]))
    geoms = load_geoms()

    a, b = st.columns([1, 1])
    with a:
        fig, ax = plt.subplots(figsize=(6.5, 7.5))
        sm = ch.choropleth(ax, geoms, vals, cmap="RdBu_r", diverging=True,
                           title=f"{name}: Sen slope (°C/decade)")
        fig.colorbar(sm, ax=ax, shrink=0.7)
        st.pyplot(fig)
    with b:
        inc = int((sub.trend == "increasing").sum())
        dec = int((sub.trend == "decreasing").sum())
        no = int((sub.trend == "no trend").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Increasing", inc); m2.metric("Decreasing", dec); m3.metric("No trend", no)
        st.metric("Median slope", f"{sub['slope/decade'].median():+.3f} °C/decade")
        st.dataframe(
            sub[["district", "slope/decade", "p_value", "trend"]]
            .sort_values("slope/decade", ascending=False)
            .set_index("district").round(3),
            width="stretch", height=460)


def validation_downloads():
    st.title("Validation, skill & downloads")
    ev = load_csv("metrics/external_validation_2025.csv")
    if not ev.empty:
        st.subheader("External validation vs observed 2025 (median across 64)")
        agg = (ev.groupby("variable")
               .agg(n=("district", "count"), R2=("val_R2", "median"),
                    RMSE=("val_RMSE", "median"), MAE=("val_MAE", "median"))
               .round(3))
        st.dataframe(agg, width="stretch")

    sk = load_csv("metrics/skill_scores.csv")
    if not sk.empty:
        st.subheader("Skill vs baselines (median)")
        st.dataframe(sk.groupby("variable")[["skill_vs_naive", "skill_vs_clim"]]
                     .median().round(3), width="stretch")

    st.divider()
    st.subheader("Download data")
    files = {
        "External validation 2025": "metrics/external_validation_2025.csv",
        "Best-model metrics": "metrics/best_model_test_metrics.csv",
        "Skill scores": "metrics/skill_scores.csv",
        "Trends 1980–2024": "trends/mann_kendall_sen_1980_2024.csv",
        "Zone annual summary": "summary/zone_summary_annual.csv",
    }
    cols = st.columns(len(files))
    for c, (label, rel) in zip(cols, files.items()):
        p = os.path.join(O, rel)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                c.download_button(label, fh.read(), file_name=os.path.basename(p),
                                  mime="text/csv", width="stretch")
    wb = os.path.join(O, "TCC_metrics_workbook.xlsx")
    if os.path.exists(wb):
        with open(wb, "rb") as fh:
            st.download_button("📊 Full reviewer workbook (.xlsx)", fh.read(),
                               file_name="TCC_metrics_workbook.xlsx")


{"Overview": overview, "District explorer": district_explorer,
 "Condition maps": condition_maps, "Observed trends": observed_trends,
 "Validation & downloads": validation_downloads}[page]()
