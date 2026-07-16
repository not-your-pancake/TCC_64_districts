#!/usr/bin/env python3
"""
District maps and multi-condition figures for the TCC pipeline (all 64 districts).

Generates, into outputs/figures/:
  map_zonation.png          study-area climatic zones (categorical)
  map_warming_trends.png    1980-2024 Sen slope per district (4 conditions)
  map_projected_means.png   projected 2025-2026 annual means (6 conditions)
  map_hazard_months.png     projected months/year in each condition's hazard class
  heatmap_multi.png         zonal monthly climatology, all key conditions

Uses the already-produced outputs -- no retraining. Run: py -3.14 build_maps.py
"""
from __future__ import annotations
import os, sys
from collections import Counter  # noqa: F401 (kept for interactive use)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tcc_pipeline import config as C
from tcc_pipeline import choropleth as ch

O = os.path.join(HERE, "outputs")
FIG = os.path.join(O, "figures")
FC = os.path.join(O, "forecasts")
geoms = ch.load_district_geometry()

zones_df = pd.read_csv(os.path.join(HERE, "tcc_pipeline", "assets", "zones.csv"))
ZID = dict(zip(zones_df["district"], zones_df["zone_id"]))
ZNAME = dict(zip(zones_df["zone_id"], zones_df["zone_name"]))
EXEMPLARS = set(zones_df[zones_df["is_exemplar"]]["district"])
ZSHORT = {1: "1. NW Extreme", 2: "2. Central Urban", 3: "3. SW Hot-Dry",
          4: "4. Coastal Humid", 5: "5. Haor Wetland", 6: "6. Hill Tract"}


def district_series(colname):
    out = {}
    for f in os.listdir(FC):
        d = f.replace("_forecast.csv", "")
        fc = pd.read_csv(os.path.join(FC, f))
        if colname in fc.columns:
            out[d] = float(fc[colname].mean())
    return out


def trend_series(var):
    tr = pd.read_csv(os.path.join(O, "trends", "mann_kendall_sen_1980_2024.csv"))
    sub = tr[tr["variable"] == var]
    return {r.district: r.sen_slope_per_year * 10 for r in sub.itertuples()}


def hazard_months(colname, threshold, year=2025):
    """Projected months in `year` at or above `threshold`, per district (0-12)."""
    out = {}
    for f in os.listdir(FC):
        d = f.replace("_forecast.csv", "")
        fc = pd.read_csv(os.path.join(FC, f))
        if colname not in fc.columns:
            continue
        fc["date"] = pd.to_datetime(fc["date"])
        v = fc[fc["date"].dt.year == year][colname].values
        out[d] = int((v >= threshold).sum())
    return out


def zone_month_matrix(colname):
    acc = {z: {m: [] for m in range(1, 13)} for z in range(1, 7)}
    for f in os.listdir(FC):
        d = f.replace("_forecast.csv", "")
        z = ZID.get(d)
        if z is None:
            continue
        fc = pd.read_csv(os.path.join(FC, f))
        if colname not in fc.columns:
            continue
        fc["date"] = pd.to_datetime(fc["date"])
        for m, val in fc.groupby(fc["date"].dt.month)[colname].mean().items():
            acc[z][int(m)].append(val)
    mat = np.full((6, 12), np.nan)
    for z in range(1, 7):
        for m in range(1, 13):
            if acc[z][m]:
                mat[z - 1, m - 1] = np.mean(acc[z][m])
    return mat


# ---- Figure: study-area zonation ----
zone_colors = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#16a085", 5: "#2980b9", 6: "#8e44ad"}
fig, ax = plt.subplots(figsize=(8, 9))
ch.choropleth_categorical(ax, geoms, {d: ZID[d] for d in geoms}, zone_colors,
                          label_districts=EXEMPLARS)
ax.set_title("Climatic zonation of Bangladesh (64 districts, 6 zones)",
             fontsize=13, fontweight="bold")
legend = [Patch(facecolor=zone_colors[z], edgecolor="white",
                label=ZNAME[z].split("(")[0].strip()) for z in sorted(zone_colors)]
ax.legend(handles=legend, loc="lower left", fontsize=7, title="Climatic zone", framealpha=0.9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "map_zonation.png"), dpi=150, bbox_inches="tight")
plt.close(fig); print("wrote map_zonation.png")

# ---- Figure: warming trends (2x2 diverging) ----
trend_specs = [("tmax", "Max temperature"), ("heat_index", "Heat index"),
               ("wet_bulb", "Wet-bulb"), ("wbgt_shade", "WBGT (shade)")]
fig, axes = plt.subplots(2, 2, figsize=(13, 15))
for ax, (var, label) in zip(axes.flat, trend_specs):
    sm = ch.choropleth(ax, geoms, trend_series(var), cmap="RdBu_r", diverging=True,
                       title=f"{label}: 1980-2024 trend")
    fig.colorbar(sm, ax=ax, shrink=0.7).set_label("Sen slope (degC/decade)", fontsize=8)
fig.suptitle("Observed warming trends by district (Sen slope, 1980-2024)",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(FIG, "map_warming_trends.png"), dpi=150, bbox_inches="tight")
plt.close(fig); print("wrote map_warming_trends.png")

# ---- Figure: projected annual means (2x3 sequential) ----
mean_specs = [("heat_index", "Heat index (degC)"), ("wbgt_sun", "WBGT sun (degC)"),
              ("wet_bulb", "Wet-bulb (degC)"), ("dew_point", "Dew point (degC)"),
              ("cdd", "CDD (degC d/d)"), ("uv", "UV index")]
fig, axes = plt.subplots(2, 3, figsize=(17, 15))
for ax, (var, label) in zip(axes.flat, mean_specs):
    sm = ch.choropleth(ax, geoms, district_series(var), cmap="YlOrRd", title=label)
    fig.colorbar(sm, ax=ax, shrink=0.7)
fig.suptitle("Projected 2025-2026 annual-mean heat-stress conditions by district",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(FIG, "map_projected_means.png"), dpi=150, bbox_inches="tight")
plt.close(fig); print("wrote map_projected_means.png")

# ---- Figure: hazard-months per district (2x2 sequential, 0-12) ----
haz_specs = [
    ("heat_index", 32.0, "Heat index >= 32 C  (Extreme Caution)"),
    ("wbgt_sun", 27.8, "Sun WBGT >= 27.8 C  (Moderate)"),
    ("uv", 8.0, "UV index >= 8  (Very High)"),
    ("dew_point", 24.0, "Dew point >= 24 C  (Extremely Oppressive)"),
]
fig, axes = plt.subplots(2, 2, figsize=(14, 15))
for ax, (var, thr, label) in zip(axes.flat, haz_specs):
    sm = ch.choropleth(ax, geoms, hazard_months(var, thr), cmap="YlOrRd",
                       vmin=0, vmax=12, title=label)
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, ticks=range(0, 13, 2))
    cb.set_label("Months per year", fontsize=8)
fig.suptitle("Projected months per year in each hazard class, by district (2025)",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(FIG, "map_hazard_months.png"), dpi=150, bbox_inches="tight")
plt.close(fig); print("wrote map_hazard_months.png")

# ---- Figure: multi-condition zonal monthly heatmap (2x3) ----
hm_specs = [("heat_index", "Heat Index (degC)"), ("wbgt_sun", "WBGT Sun (degC)"),
            ("wet_bulb", "Wet-bulb (degC)"), ("dew_point", "Dew point (degC)"),
            ("cdd", "CDD (degC d/d)"), ("uv", "UV index")]
months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
zlabels = [ZSHORT[z] for z in range(1, 7)]
fig, axes = plt.subplots(2, 3, figsize=(16, 7))
for ax, (var, label) in zip(axes.flat, hm_specs):
    im = ax.imshow(zone_month_matrix(var), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(12)); ax.set_xticklabels(months, fontsize=7)
    ax.set_yticks(range(6)); ax.set_yticklabels(zlabels, fontsize=7)
    ax.set_title(label, fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.85)
fig.suptitle("Projected zonal monthly climatology by condition (2025-2026)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(FIG, "heatmap_multi.png"), dpi=150, bbox_inches="tight")
plt.close(fig); print("wrote heatmap_multi.png")
print("done")
