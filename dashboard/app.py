"""
TCC Bangladesh — interactive heat-stress dashboard ("Heat Ember" edition).

Reads the pipeline outputs in ../outputs directly and renders district-level
projections, validation, trends, and risk categories with interactive Plotly
charts on a dark glassmorphism theme (see style.css + ../.streamlit/config.toml).

Run:  py -3.14 -m streamlit run dashboard/app.py
"""
import html
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
O = os.path.join(ROOT, "outputs")
FIG = os.path.join(O, "figures")
FC = os.path.join(O, "forecasts")

st.set_page_config(page_title="TCC Bangladesh · Heat-Stress Dashboard",
                   page_icon="🌡️", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Design tokens — keep in sync with dashboard/style.css.
# Chart colors were validated with the dataviz palette checks (CVD, lightness
# band, chroma, contrast) against the dark card surface #151b28.
ACCENT = "#ff6b4a"   # UI accent (buttons, glows)
EMBER  = "#ef5330"   # chart series color (in dark lightness band)
AMBER  = "#ffb86b"
COOL   = "#4da3ff"
INK    = "#eef1f8"
INK2   = "#aab3c5"
MUTED  = "#7c869c"
CARD   = "#151b28"
PAGE   = "#0b0e15"
GRID   = "rgba(238,241,248,0.07)"
BASELINE = "rgba(238,241,248,0.18)"
FONT   = "Inter, 'Segoe UI', sans-serif"
FONT_H = "'Space Grotesk', 'Segoe UI', sans-serif"

# Sequential heat ramp: Inferno sampled 0.15→0.97 (dark-anchored: low recedes
# into the surface, high glows). Perceptually ordered.
HEAT_SCALE = [[i / 7, c] for i, c in enumerate(
    ["#2b0c50", "#5d126c", "#8d2367", "#bb3853",
     "#e05933", "#f68812", "#f8bf2b", "#fbf388"])]
# Diverging: cool ↔ warm with a dark neutral midpoint (never white on dark).
DIV_SCALE = [[0.0, COOL], [0.5, "#30323b"], [1.0, "#ff5a4e"]]

# 6-zone categorical set — validated all-pairs (choropleth) on the dark surface.
ZONE_COLOR = {1: "#e66767", 2: "#d95926", 3: "#c98500",
              4: "#3987e5", 5: "#d55181", 6: "#008300"}
ZONE_SHORT = {1: "Northwest Extreme Heat", 2: "Central Urban Heat-Island",
              3: "Southwest Hot-Dry", 4: "Coastal Humid Heat",
              5: "Haor Wetland Humid", 6: "Hill Tracts (Elevated)"}

CONDITIONS = {  # display -> forecast column
    "Heat index (°C)": "heat_index",
    "Wet-bulb temperature (°C)": "wet_bulb",
    "WBGT — sun (°C)": "wbgt_sun",
    "WBGT — shade (°C)": "wbgt_shade",
    "Cooling degree-days (°C·d/d)": "cdd",
    "UV index": "uv",
    "Dew point (°C)": "dew_point",
}
UNITS = {"heat_index": "°C", "wet_bulb": "°C", "wbgt_sun": "°C",
         "wbgt_shade": "°C", "cdd": "°C·d/d", "uv": "", "dew_point": "°C"}
TREND_VARS = {
    "Maximum temperature": "tmax", "Mean temperature": "temperature",
    "Minimum temperature": "tmin", "Dew point": "dew_point",
    "Relative humidity": "humidity", "Heat index": "heat_index",
    "Wet-bulb temperature": "wet_bulb", "WBGT (shade)": "wbgt_shade",
}
VAR_LABEL = {"temperature": "Mean temperature", "tmax": "Max temperature",
             "tmin": "Min temperature", "humidity": "Relative humidity",
             "dew_point": "Dew point", "heat_index": "Heat index",
             "wet_bulb": "Wet-bulb", "wbgt_shade": "WBGT shade",
             "wbgt_sun": "WBGT sun", "uv": "UV index", "cdd": "Cooling DD",
             "solar_radiation": "Solar radiation",
             "heatstroke_hazard": "Heatstroke hazard"}

PLOTLY_CONF = {"displayModeBar": False, "scrollZoom": False}
esc = html.escape


# --------------------------------------------------------------------------- #
def load_css():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "style.css"), encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_zones():
    return pd.read_csv(os.path.join(ROOT, "tcc_pipeline", "assets", "zones.csv"))


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
def geojson_fc():
    """Lightweight simplified geometry (see dashboard/build_geo.py).

    The raw ADM2 file is ~46 MB — shipping it to the browser hangs Plotly, so
    the app requires the precomputed ~0.35 MB version and regenerates it on
    the fly if it is missing.
    """
    import json
    light = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "assets", "districts_light.geojson")
    if not os.path.exists(light):
        from dashboard import build_geo
        build_geo.main()
    with open(light, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# HTML components (all interpolated strings are escaped)
def hero(title, sub, chips=()):
    chip_html = "".join(
        f'<span class="tcc-chip {cls}"><span class="dot"></span>{esc(t)}</span>'
        for t, cls in chips)
    st.markdown(
        f'<div class="tcc-hero"><h1>{esc(title)}</h1>'
        f'<p class="sub">{esc(sub)}</p>'
        + (f'<div class="tcc-chips">{chip_html}</div>' if chips else "")
        + "</div>", unsafe_allow_html=True)


def kpi_row(cards):
    """cards: list of (icon, label, value, sub)."""
    parts = []
    for ico, label, value, sub in cards:
        parts.append(
            f'<div class="tcc-kpi">'
            + (f'<div class="ico">{ico}</div>' if ico else "")
            + f'<div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>'
            + (f'<div class="s">{esc(sub)}</div>' if sub else "")
            + "</div>")
    st.markdown(f'<div class="tcc-kpi-row">{"".join(parts)}</div>',
                unsafe_allow_html=True)


def feature(ico, title, text):
    st.markdown(
        f'<div class="tcc-feature"><div class="fi">{ico}</div>'
        f'<div><b>{esc(title)}</b><span>{esc(text)}</span></div></div>',
        unsafe_allow_html=True)


def meter(label, months):
    if months <= 2:
        cls, tag = "good", "LOW"
    elif months <= 5:
        cls, tag = "warning", "ELEVATED"
    elif months <= 8:
        cls, tag = "serious", "HIGH"
    else:
        cls, tag = "critical", "SEVERE"
    pct = round(months / 12 * 100)
    st.markdown(
        f'<div class="tcc-meter {cls}"><div class="head">'
        f'<span class="name">{esc(label)}</span>'
        f'<span class="val">{months}/12 mo<span class="tag">{tag}</span></span></div>'
        f'<div class="bar"><div class="fill" style="width:{pct}%"></div></div></div>',
        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Plotly helpers
def polish(fig, h=380, **kw):
    layout = dict(
        height=h,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK2),
        hoverlabel=dict(bgcolor="#101524", bordercolor="rgba(255,255,255,0.14)",
                        font=dict(family=FONT, size=13, color=INK)),
        margin=dict(l=6, r=6, t=34, b=6))
    layout.update(kw)
    fig.update_layout(**layout)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def forecast_fig(fc, col, cond_label, district):
    unit = UNITS.get(col, "")
    x = fc["date"]
    fig = go.Figure()
    lo95, hi95 = f"{col}_lo95", f"{col}_hi95"
    if lo95 in fc.columns:
        fig.add_trace(go.Scatter(x=x, y=fc[hi95], mode="lines",
                                 line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=fc[lo95], mode="lines", fill="tonexty",
                                 fillcolor="rgba(239,83,48,0.10)",
                                 line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=fc[f"{col}_hi68"], mode="lines",
                                 line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=fc[f"{col}_lo68"], mode="lines",
                                 fill="tonexty", fillcolor="rgba(239,83,48,0.20)",
                                 line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
    fig.add_trace(go.Scatter(
        x=x, y=fc[col], mode="lines+markers", name="Projected",
        line=dict(color=EMBER, width=2.4),
        marker=dict(size=6, color=EMBER,
                    line=dict(color=CARD, width=1.5)),
        hovertemplate="%{y:.2f} " + unit + "<extra></extra>",
        showlegend=False))
    fig.add_vline(x=pd.Timestamp("2026-01-01"), line_width=1,
                  line_color="rgba(238,241,248,0.22)")
    fig.add_annotation(x=pd.Timestamp("2026-01-15"), yref="paper", y=1.02,
                       text="2026 →", showarrow=False,
                       font=dict(color=MUTED, size=11), xanchor="left")
    polish(fig, h=390,
           title=dict(text=f"{district} — {cond_label}",
                      font=dict(family=FONT_H, size=17, color=INK), x=0),
           hovermode="x unified")
    fig.update_xaxes(gridcolor=GRID, showspikes=True, spikemode="across",
                     spikesnap="cursor", spikedash="solid", spikethickness=1,
                     spikecolor="rgba(238,241,248,0.30)",
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False,
                     title=dict(text=unit, font=dict(color=MUTED)),
                     tickfont=dict(color=MUTED))
    return fig


def map_fig(vals, colorscale, unit="", zmid=None, hover_suffix="", h=580):
    """Continuous choropleth with hover. vals: {district: value}."""
    zones = load_zones()
    zone_of = dict(zip(zones["district"], zones["zone_name"]))
    locs = [d for d, v in vals.items() if v is not None and np.isfinite(v)]
    z = [vals[d] for d in locs]
    custom = [zone_of.get(d, "") for d in locs]
    kw = {}
    if zmid is not None:
        m = max(abs(min(z)), abs(max(z)))
        kw = dict(zmid=zmid, zmin=-m, zmax=m)
    fig = go.Figure(go.Choropleth(
        geojson=geojson_fc(), locations=locs, z=z, customdata=custom,
        colorscale=colorscale, marker_line_color=PAGE, marker_line_width=0.6,
        colorbar=dict(thickness=12, outlinewidth=0, len=0.72,
                      ticksuffix=f" {unit}" if unit else "",
                      tickfont=dict(color=MUTED, size=11)),
        hovertemplate="<b>%{location}</b><br>%{z:.2f} " + unit + hover_suffix
                      + "<br><span style='color:#aab3c5'>%{customdata}</span>"
                        "<extra></extra>",
        **kw))
    fig.update_geos(bgcolor="rgba(0,0,0,0)", visible=False,
                    fitbounds="locations", projection_type="mercator")
    polish(fig, h=h, margin=dict(l=0, r=0, t=6, b=6))
    return fig


def zone_map_fig(h=620):
    zones = load_zones()
    fig = go.Figure()
    for zid in sorted(ZONE_COLOR):
        sub = zones[zones["zone_id"] == zid]
        if sub.empty:
            continue
        full = sub["zone_name"].iloc[0]
        fig.add_trace(go.Choropleth(
            geojson=geojson_fc(), locations=list(sub["district"]),
            z=[1] * len(sub),
            colorscale=[[0, ZONE_COLOR[zid]], [1, ZONE_COLOR[zid]]],
            showscale=False, name=ZONE_SHORT[zid], showlegend=True,
            marker_line_color=PAGE, marker_line_width=0.7,
            hovertemplate=f"<b>%{{location}}</b><br>{full}<extra></extra>"))
    fig.update_geos(bgcolor="rgba(0,0,0,0)", visible=False,
                    fitbounds="locations", projection_type="mercator")
    polish(fig, h=h, margin=dict(l=0, r=0, t=6, b=0),
           legend=dict(orientation="h", y=-0.02, x=0,
                       font=dict(size=11.5, color=INK2),
                       itemsizing="constant"))
    return fig


# --------------------------------------------------------------------------- #
zones = load_zones()
DISTRICTS = sorted(zones["district"])
ZONE_OF = dict(zip(zones["district"], zones["zone_name"]))
EXEMPLARS = set(zones[zones["is_exemplar"]]["district"])

with st.sidebar:
    st.markdown(
        '<div class="tcc-brand"><div class="mark">🌡️</div>'
        '<div><div class="t1">TCC Bangladesh</div>'
        '<div class="t2">HEAT-STRESS INTELLIGENCE</div></div></div>',
        unsafe_allow_html=True)
    page = st.radio("Navigate",
                    ["🏠 Overview", "📍 District explorer", "🗺️ Condition maps",
                     "📈 Observed trends", "📥 Validation & downloads"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown(
        '<div class="tcc-side-note"><b>Seven Temperature-Correlated Conditions</b> '
        'projected for all 64 districts of Bangladesh — 1980–2024 baseline, '
        '2025–26 outlook, validated against observed 2025.</div>',
        unsafe_allow_html=True)
    st.divider()
    st.markdown(
        '<div class="tcc-side-note">Live app · '
        '<a href="https://tcc-64-districts-mohammad-khalid.streamlit.app/">'
        'tcc-64-districts</a><br>Built with Streamlit + Plotly</div>',
        unsafe_allow_html=True)

load_css()


# --------------------------------------------------------------------------- #
def overview():
    hero("Bangladesh is heating up — district by district.",
         "Seven heat-stress conditions projected for every one of the 64 "
         "districts, with honest uncertainty and out-of-sample validation.",
         chips=[("64 districts", ""), ("7 conditions", "amber"),
                ("1980–2024 baseline", "cool"), ("2025–26 projection", ""),
                ("validated vs observed 2025", "amber")])

    ev = load_csv("metrics/external_validation_2025.csv")
    tr = load_csv("trends/mann_kendall_sen_1980_2024.csv")
    cards = [("🗺️", "Districts covered", "64", "six climatic zones"),
             ("🌡️", "Heat-stress conditions", "7", "heat index → UV → CDD")]
    if not ev.empty:
        hi = ev[ev.variable == "heat_index"]["val_R2"].median()
        cards.append(("🎯", "Heat-index validation R²", f"{hi:.3f}",
                      "median across 64 districts, 2025"))
    if not tr.empty:
        tx = tr[tr.variable == "tmax"]["sen_slope_per_year"].median() * 10
        cards.append(("📈", "Max-temp warming", f"+{tx:.2f} °C",
                      "per decade · median Sen slope"))
    kpi_row(cards)

    st.divider()
    a, b = st.columns([1.08, 1], gap="large")
    with a:
        st.subheader("Six climatic zones")
        st.plotly_chart(zone_map_fig(), width="stretch", theme=None,
                        config=PLOTLY_CONF)
        st.caption("Hover any district for its zone assignment.")
    with b:
        st.subheader("Explore the dashboard")
        feature("📍", "District explorer",
                "Per-district 24-month projection with 68/95% confidence "
                "bands, 2025 validation scores, and hazard-month meters.")
        feature("🗺️", "Condition maps",
                "Interactive choropleth of any of the seven conditions across "
                "all 64 districts — hover for values, ranked table alongside.")
        feature("📈", "Observed trends",
                "1980–2024 Mann–Kendall / Sen-slope warming, district by "
                "district, on a diverging cool–warm scale.")
        feature("📥", "Validation & downloads",
                "Accuracy vs observed 2025, skill vs baselines, and one-click "
                "CSV / workbook export.")
        st.info("Skill is reported honestly: at monthly resolution the models "
                "match climatology. Their value is spatially complete, "
                "uncertainty-aware, trend-consistent projection — not lower "
                "monthly error.")


def district_explorer():
    hero("District explorer",
         "24-month projection, uncertainty bands, validation and hazard "
         "outlook for any district.")
    f1, f2 = st.columns([1, 1])
    with f1:
        d = st.selectbox("District", DISTRICTS,
                         index=DISTRICTS.index("Dhaka") if "Dhaka" in DISTRICTS else 0)
    with f2:
        cond = st.selectbox("Condition", list(CONDITIONS.keys()))
    col = CONDITIONS[cond]
    zone = ZONE_OF.get(d, "—")
    chips = [(zone, "cool")] + ([("exemplar district", "amber")] if d in EXEMPLARS else [])
    st.markdown('<div class="tcc-chips">' + "".join(
        f'<span class="tcc-chip {c}"><span class="dot"></span>{esc(t)}</span>'
        for t, c in chips) + "</div>", unsafe_allow_html=True)

    fc = load_forecast(d)
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.plotly_chart(forecast_fig(fc, col, cond, d), width="stretch",
                        theme=None, config=PLOTLY_CONF)
        st.caption("Shaded bands: 68% and 95% confidence intervals. "
                   "Hover for monthly values; drag to zoom, double-click to reset.")
        tp = os.path.join(FIG, f"threepanel_{d}_heat_index.png")
        if os.path.exists(tp):
            with st.expander("Manuscript three-panel figure (record · validation · projection)"):
                st.image(tp, width="stretch")
    with right:
        st.subheader("2025 validation")
        ev = load_csv("metrics/external_validation_2025.csv")
        if not ev.empty:
            sub = ev[ev.district == d][["variable", "val_R2", "val_RMSE", "val_MAE"]].copy()
            sub["variable"] = sub["variable"].map(lambda v: VAR_LABEL.get(v, v))
            st.dataframe(sub.set_index("variable").round(3), width="stretch",
                         height=280)
        st.subheader("Projected hazard months · 2025")
        thr = {"heat_index": (32, "Heat index ≥ 32 °C (Extreme Caution)"),
               "wbgt_sun": (27.8, "WBGT-sun ≥ 27.8 °C (Moderate)"),
               "uv": (8, "UV index ≥ 8 (Very High)"),
               "dew_point": (24, "Dew point ≥ 24 °C (Extremely Oppressive)")}
        y25 = fc[fc["date"].dt.year == 2025]
        for c, (t, lab) in thr.items():
            if c in y25.columns:
                meter(lab, int((y25[c] >= t).sum()))


def condition_maps():
    hero("Condition maps",
         "Projected 2025–26 annual mean of any condition, across all 64 districts.")
    cond = st.selectbox("Condition", list(CONDITIONS.keys()))
    col = CONDITIONS[cond]
    unit = UNITS.get(col, "")
    dm = district_means()
    vals = dm[col].to_dict()
    s = pd.Series(vals).sort_values(ascending=False)

    kpi_row([("🔥", "Hottest district", s.index[0], f"{s.iloc[0]:.2f} {unit}".strip()),
             ("❄️", "Lowest district", s.index[-1], f"{s.iloc[-1]:.2f} {unit}".strip()),
             ("↔️", "Spread across districts", f"{(s.iloc[0] - s.iloc[-1]):.2f} {unit}".strip(),
              "max − min, annual mean")])

    a, b = st.columns([1.15, 1], gap="large")
    with a:
        st.plotly_chart(map_fig(vals, HEAT_SCALE, unit), width="stretch",
                        theme=None, config=PLOTLY_CONF)
        st.caption("Hover for values · brighter = hotter (dark-anchored ramp).")
    with b:
        st.subheader("Ranked by projected annual mean")
        df = pd.DataFrame({"District": s.index, "Value": s.values})
        st.dataframe(df, hide_index=True, width="stretch", height=520,
                     column_config={
                         "Value": st.column_config.ProgressColumn(
                             f"annual mean {('(' + unit + ')') if unit else ''}",
                             format="%.2f",
                             min_value=float(s.min()), max_value=float(s.max()))})


def observed_trends():
    hero("Observed warming trends",
         "Mann–Kendall significance and Sen-slope magnitude over 1980–2024.")
    name = st.selectbox("Variable", list(TREND_VARS.keys()))
    var = TREND_VARS[name]
    unit = "%" if var == "humidity" else "°C"
    tr = load_csv("trends/mann_kendall_sen_1980_2024.csv")
    sub = tr[tr.variable == var].copy()
    sub["slope_dec"] = sub["sen_slope_per_year"] * 10

    inc = int((sub.trend == "increasing").sum())
    dec = int((sub.trend == "decreasing").sum())
    no = int((sub.trend == "no trend").sum())
    kpi_row([("📈", "Increasing", str(inc), "districts, p < 0.05"),
             ("➖", "No significant trend", str(no), "districts"),
             ("📉", "Decreasing", str(dec), "districts, p < 0.05"),
             ("⏱️", "Median slope", f"{sub['slope_dec'].median():+.3f} {unit}",
              "per decade · Sen estimator")])

    a, b = st.columns([1.15, 1], gap="large")
    with a:
        vals = dict(zip(sub["district"], sub["slope_dec"]))
        st.plotly_chart(map_fig(vals, DIV_SCALE, unit, zmid=0.0,
                                hover_suffix="/decade"),
                        width="stretch", theme=None, config=PLOTLY_CONF)
        st.caption("Diverging scale centred at zero — warm = warming, cool = cooling.")
    with b:
        st.subheader("All districts")
        t = (sub[["district", "slope_dec", "p_value", "trend"]]
             .sort_values("slope_dec", ascending=False)
             .rename(columns={"district": "District", "slope_dec": "Slope/decade",
                              "p_value": "p", "trend": "Trend"}))
        st.dataframe(t, hide_index=True, width="stretch", height=520,
                     column_config={
                         "Slope/decade": st.column_config.NumberColumn(
                             f"Slope ({unit}/dec)", format="%+.3f"),
                         "p": st.column_config.NumberColumn(format="%.4f")})


def validation_downloads():
    hero("Validation, skill & downloads",
         "How well the projection held up against observed 2025 — and every "
         "table as CSV.")
    ev = load_csv("metrics/external_validation_2025.csv")
    sk = load_csv("metrics/skill_scores.csv")

    a, b = st.columns([1.2, 1], gap="large")
    with a:
        if not ev.empty:
            st.subheader("External validation vs observed 2025")
            agg = (ev.groupby("variable")["val_R2"].median()
                   .sort_values())
            labels = [VAR_LABEL.get(v, v) for v in agg.index]
            fig = go.Figure(go.Bar(
                x=agg.values, y=labels, orientation="h",
                marker=dict(color=EMBER, cornerradius=4),
                width=0.55,
                texttemplate="%{x:.3f}", textposition="outside",
                textfont=dict(color=INK2, size=12), cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>median R² %{x:.3f}<extra></extra>"))
            polish(fig, h=max(300, 34 * len(agg) + 60),
                   title=dict(text="Median R² by variable (64 districts)",
                              font=dict(family=FONT_H, size=16, color=INK), x=0))
            fig.update_xaxes(gridcolor=GRID, zeroline=True,
                             zerolinecolor=BASELINE, zerolinewidth=1,
                             range=[min(-0.05, float(agg.min()) * 1.15), 1.12],
                             tickfont=dict(color=MUTED))
            fig.update_yaxes(tickfont=dict(color=INK2))
            st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONF)
    with b:
        if not ev.empty:
            st.subheader("Medians across districts")
            agg = (ev.groupby("variable")
                   .agg(n=("district", "count"), R2=("val_R2", "median"),
                        RMSE=("val_RMSE", "median"), MAE=("val_MAE", "median"))
                   .round(3))
            agg.index = [VAR_LABEL.get(v, v) for v in agg.index]
            st.dataframe(agg, width="stretch", height=260)
        if not sk.empty:
            st.subheader("Skill vs baselines")
            skm = (sk.groupby("variable")[["skill_vs_naive", "skill_vs_clim"]]
                   .median().round(3))
            skm.index = [VAR_LABEL.get(v, v) for v in skm.index]
            st.dataframe(skm, width="stretch")

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


{"🏠 Overview": overview, "📍 District explorer": district_explorer,
 "🗺️ Condition maps": condition_maps, "📈 Observed trends": observed_trends,
 "📥 Validation & downloads": validation_downloads}[page]()
