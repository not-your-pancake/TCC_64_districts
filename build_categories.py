#!/usr/bin/env python3
"""
Post-process the projected forecasts into operational risk categories.

Applies the interpretive band systems in config.py (heat-index comfort, WBGT
occupational, WHO UV, dew-point moisture stress) to the already-projected
monthly fields -- no retraining. Produces:
  outputs/categories/projected_category_distribution.csv   (overall + by zone)
  outputs/categories/category_validation_2025.csv          (predicted vs observed)
  outputs/figures/category_heatindex_by_zone.png
and prints the summary numbers used in the manuscript.

Run with:  py -3.14 build_categories.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tcc_pipeline import config as C
from tcc_pipeline import indices as ix
from tcc_pipeline.engine import load_monthly

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = os.path.join(HERE, "outputs")
FC = os.path.join(O, "forecasts")
CATDIR = os.path.join(O, "categories")
os.makedirs(CATDIR, exist_ok=True)

# Condition -> (label, band system, ordered category labels)
CONDITIONS = {
    "heat_index": ("Heat-index comfort", C.HEAT_INDEX_BANDS),
    "wbgt_sun":   ("WBGT (sun, outdoor)", C.WBGT_BANDS),
    "wbgt_shade": ("WBGT (shade, indoor)", C.WBGT_BANDS),
    "uv":         ("UV (WHO)", C.UV_BANDS),
    "dew_point":  ("Dew-point moisture stress", C.DEW_POINT_COMFORT_BANDS),
}


def band_labels(bands):
    return [b[2] for b in bands]


def categorize(values, bands):
    labels = band_labels(bands)
    out = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        lab = labels[-1]
        for lo, hi, name in bands:
            if lo <= v < hi:
                lab = name
                break
        out[i] = lab
    return out


def cat_index(series, bands):
    order = {lab: k for k, lab in enumerate(band_labels(bands))}
    return series.map(order)


# --- zone map ---
zones = pd.read_csv(os.path.join(HERE, "tcc_pipeline", "assets", "zones.csv"))
zone_of = dict(zip(zones["district"], zones["zone_name"]))

# --- 1. projected category distribution (all districts x 24 months) ---
records = []
zone_records = []
for f in sorted(os.listdir(FC)):
    d = f.replace("_forecast.csv", "")
    fc = pd.read_csv(os.path.join(FC, f))
    z = zone_of.get(d, "NA")
    for var, (label, bands) in CONDITIONS.items():
        if var not in fc.columns:
            continue
        cats = categorize(fc[var].values, bands)
        for c in cats:
            records.append({"condition": label, "category": c})
            zone_records.append({"zone": z, "condition": label, "category": c})

dist = pd.DataFrame(records)
rows = []
for var, (label, bands) in CONDITIONS.items():
    sub = dist[dist["condition"] == label]
    n = len(sub)
    for cat in band_labels(bands):
        cnt = int((sub["category"] == cat).sum())
        rows.append({"condition": label, "category": cat,
                     "n_district_months": cnt, "pct": round(100 * cnt / n, 1) if n else 0.0})
dist_tab = pd.DataFrame(rows)
dist_tab.to_csv(os.path.join(CATDIR, "projected_category_distribution.csv"), index=False)

print("===== PROJECTED RISK-CATEGORY DISTRIBUTION (2025-2026, 64 districts) =====")
for var, (label, bands) in CONDITIONS.items():
    sub = dist_tab[dist_tab["condition"] == label]
    parts = [f"{r.category} {r.pct}%" for r in sub.itertuples() if r.pct > 0]
    print(f"  {label:28s}: " + "  ".join(parts))

# by zone (heat index only, for the figure + a table)
zdf = pd.DataFrame(zone_records)
hi_bands = C.HEAT_INDEX_BANDS
hi_labels = band_labels(hi_bands)
zone_hi = (zdf[zdf["condition"] == "Heat-index comfort"]
           .groupby(["zone", "category"]).size().unstack(fill_value=0))
zone_hi = zone_hi.reindex(columns=hi_labels, fill_value=0)
zone_hi_pct = zone_hi.div(zone_hi.sum(axis=1), axis=0) * 100
zone_hi_pct.to_csv(os.path.join(CATDIR, "heatindex_category_by_zone.csv"))
print("\n===== HEAT-INDEX COMFORT CATEGORY BY ZONE (% of projected months) =====")
print(zone_hi_pct.round(1).to_string())

# --- 2. 2025 categorical validation: predicted vs observed ---
val_rows = []
for f in sorted(os.listdir(FC)):
    d = f.replace("_forecast.csv", "")
    obs_csv = os.path.join(HERE, "2025-dataset", f"{d}_historical_weather_2025.csv")
    if not os.path.exists(obs_csv):
        continue
    fc = pd.read_csv(os.path.join(FC, f)); fc["date"] = pd.to_datetime(fc["date"])
    p25 = fc[fc["date"].dt.year == 2025].copy(); p25["m"] = p25["date"].dt.month
    om = load_monthly(obs_csv, 2025, 2025)
    obs = ix.recompute_indices_from_base(om.reset_index())
    obs["m"] = pd.to_datetime(obs["date"]).dt.month
    for var, (label, bands) in CONDITIONS.items():
        if var not in p25.columns or var not in obs.columns:
            continue
        j = pd.merge(p25[["m", var]].rename(columns={var: "pred"}),
                     obs[["m", var]].rename(columns={var: "obs"}), on="m").dropna()
        if j.empty:
            continue
        pc = cat_index(pd.Series(categorize(j["pred"].values, bands)), bands)
        oc = cat_index(pd.Series(categorize(j["obs"].values, bands)), bands)
        val_rows.append({"district": d, "condition": label,
                         "n": len(j),
                         "exact": int((pc.values == oc.values).sum()),
                         "adj": int((np.abs(pc.values - oc.values) <= 1).sum())})

vdf = pd.DataFrame(val_rows)
vsum = (vdf.groupby("condition")[["n", "exact", "adj"]].sum())
vsum["exact_acc"] = (vsum["exact"] / vsum["n"] * 100).round(1)
vsum["adj_acc"] = (vsum["adj"] / vsum["n"] * 100).round(1)
vsum.to_csv(os.path.join(CATDIR, "category_validation_2025.csv"))
print("\n===== 2025 CATEGORICAL VALIDATION (predicted vs observed) =====")
print(vsum[["n", "exact_acc", "adj_acc"]].to_string())

# --- 3. figure: heat-index comfort category by zone (stacked bar) ---
colors = {"Comfortable": "#2ecc71", "Caution": "#f1c40f",
          "Extreme Caution": "#e67e22", "Danger": "#e74c3c", "Extreme Danger": "#7b241c"}
fig, ax = plt.subplots(figsize=(11, 5))
zorder = list(zone_hi_pct.index)
short = [z.split("(")[0].strip() for z in zorder]
left = np.zeros(len(zorder))
for cat in hi_labels:
    vals = zone_hi_pct[cat].values
    ax.barh(range(len(zorder)), vals, left=left, label=cat,
            color=colors.get(cat, "#999999"), edgecolor="white", height=0.7)
    left += vals
ax.set_yticks(range(len(zorder))); ax.set_yticklabels(short, fontsize=9)
ax.set_xlabel("Share of projected months (%)"); ax.set_xlim(0, 100)
ax.invert_yaxis()
ax.set_title("Projected heat-index comfort categories by climatic zone (2025-2026)",
             fontsize=12, fontweight="bold")
ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
figpath = os.path.join(O, "figures", "category_heatindex_by_zone.png")
fig.savefig(figpath, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"\nwrote {figpath}")
print("wrote categories/ tables")
