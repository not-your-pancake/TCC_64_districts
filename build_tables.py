#!/usr/bin/env python3
"""
Generate the LaTeX tabular blocks for the manuscript directly from outputs/, so
Tables 1-6 stay in sync with the pipeline after any re-run. Each output file
contains a complete \\begin{tabular}...\\end{tabular} (column spec, header,
rules, data rows) and is \\input at the table-float level in
manuscript/manuscript.tex -- NOT inside a tabular, because \\input inside a
tabular breaks under Tectonic ("Misplaced \\noalign"). Captions and labels stay
in the manuscript.

Run AFTER run_pipeline.py + build_categories.py:  py -3.14 build_tables.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(HERE, "outputs")
TDIR = os.path.join(HERE, "manuscript", "tables")
os.makedirs(TDIR, exist_ok=True)

VAR_LABEL = {
    "temperature": "Mean temperature", "tmax": "Maximum temperature",
    "tmin": "Minimum temperature", "humidity": "Relative humidity",
    "dew_point": "Dew point", "solar_radiation": "Solar radiation", "uv": "UV index",
    "heat_index": "Heat index", "wet_bulb": "Wet-bulb temperature",
    "wbgt_shade": "WBGT (shade)", "wbgt_sun": "WBGT (sun)",
    "cdd": "Cooling degree-days",
}
ZONE_SHORT = {
    1: "1. Northwest Extreme Heat", 2: "2. Central Urban Heat",
    3: "3. Southwest Hot--Dry", 4: "4. Coastal Humid Heat",
    5: "5. Haor/Wetland Humid", 6: "6. Hill Tract",
}


def write_tabular(name, colspec, header, body):
    """Write a complete tabular block. `body` is a list of row / \\midrule lines."""
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header, "\\midrule"]
    lines += body
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TDIR, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote tables/{name} ({len(body)} body lines)")


def pad(s, w):
    return s + " " * max(0, w - len(s))


# ---- Table 1: best-model selection frequency ----
bm = pd.read_csv(os.path.join(O, "metrics", "best_model_test_metrics.csv"))
ct = pd.crosstab(bm["variable"], bm["best_model"]).reindex(
    columns=["LightGBM", "RandomForest", "CatBoost"], fill_value=0)
body = []
for v in ["temperature", "tmax", "tmin", "humidity", "dew_point", "solar_radiation", "uv"]:
    r = ct.loc[v]
    body.append(f"{pad(VAR_LABEL[v], 21)} & {r['LightGBM']:>2d} & {r['RandomForest']:>2d} "
                f"& {r['CatBoost']:>2d} \\\\")
write_tabular("table_models.tex", "lccc",
              "Variable & LightGBM & Random Forest & CatBoost \\\\", body)

# ---- Table 2: external validation 2025 ----
ev = pd.read_csv(os.path.join(O, "metrics", "external_validation_2025.csv"))
g = ev.groupby("variable").agg(n=("district", "count"), R2=("val_R2", "median"),
                               RMSE=("val_RMSE", "median"), MAE=("val_MAE", "median"))
def vrow(v):
    r = g.loc[v]
    return f"{pad(VAR_LABEL[v], 19)} & {int(r['n'])} & {r['R2']:.3f} & {r['RMSE']:.3f} & {r['MAE']:.3f} \\\\"
body = [vrow(v) for v in ["temperature", "tmax", "tmin", "humidity", "dew_point",
                          "solar_radiation", "uv"]]
body += ["\\midrule"] + [vrow(v) for v in ["heat_index", "wet_bulb", "wbgt_shade",
                                           "wbgt_sun", "cdd"]]
write_tabular("table_validation.tex", "lcccc",
              "Variable & $n$ & Median $R^{2}$ & Median RMSE & Median MAE \\\\", body)

# ---- Table 3: observed trends 1980-2024 ----
tr = pd.read_csv(os.path.join(O, "trends", "mann_kendall_sen_1980_2024.csv"))
body = []
for v in ["tmax", "temperature", "tmin", "dew_point", "humidity",
          "heat_index", "wet_bulb", "wbgt_shade"]:
    s = tr[tr.variable == v]
    inc = int((s.trend == "increasing").sum())
    dec = int((s.trend == "decreasing").sum())
    no = int((s.trend == "no trend").sum())
    med = float(s["sen_slope_per_decade"].median())
    body.append(f"{pad(VAR_LABEL[v], 21)} & {inc:>2d} & {dec:>2d} & {no:>2d} & ${med:+.3f}$ \\\\")
write_tabular("table_trends.tex", "lcccc",
              "Variable & Increasing & Decreasing & No trend & Median slope/decade \\\\", body)

# ---- Table 4: zonal projected annual means ----
za = pd.read_csv(os.path.join(O, "summary", "zone_summary_annual.csv"))
za["zid"] = za["zone_name"].str.extract(r"^(\d+)").astype(int)
body = []
for zid in range(1, 7):
    sub = za[za["zid"] == zid]
    def m(var):
        return float(sub[sub.variable == var]["mean"].iloc[0])
    hi = sub[sub.variable == "heat_index"].iloc[0]
    hi_cell = f"{hi['mean']:.1f} ({hi['min']:.1f}--{hi['max']:.1f})"
    body.append(f"{pad(ZONE_SHORT[zid], 25)} & {hi_cell} & {m('wbgt_sun'):.1f} "
                f"& {m('wet_bulb'):.1f} & {m('cdd'):.1f} \\\\")
write_tabular("table_zonal.tex", "lcccc",
              "Zone & Heat index & WBGT (sun) & Wet-bulb & CDD \\\\", body)

# ---- Table 5: projected risk-category distribution ----
dist = pd.read_csv(os.path.join(O, "categories", "projected_category_distribution.csv"))
cond_order = ["Heat-index comfort", "WBGT (sun, outdoor)", "WBGT (shade, indoor)",
              "UV (WHO)", "Dew-point moisture stress"]
body = []
for i, cond in enumerate(cond_order):
    sub = dist[(dist.condition == cond) & (dist.pct > 0)]
    if i > 0:
        body.append("\\midrule")
    first = True
    for r in sub.itertuples():
        label = cond if first else ""
        body.append(f"{pad(label, 25)} & {pad(r.category, 23)} & {r.pct:.1f} \\\\")
        first = False
write_tabular("table_catdist.tex", "llr",
              "Condition & Category & Share (\\%) \\\\", body)

# ---- Table 6: categorical validation 2025 ----
cv = pd.read_csv(os.path.join(O, "categories", "category_validation_2025.csv")).set_index("condition")
body = []
for cond in ["Heat-index comfort", "WBGT (sun, outdoor)", "WBGT (shade, indoor)",
             "UV (WHO)", "Dew-point moisture stress"]:
    r = cv.loc[cond]
    body.append(f"{pad(cond, 26)} & {int(r['n'])} & {r['exact_acc']:.1f} & {r['adj_acc']:.1f} \\\\")
write_tabular("table_catval.tex", "lccc",
              "Condition & $n$ & Exact-class (\\%) & Within one class (\\%) \\\\", body)

print("done -- 6 complete tabular blocks written to manuscript/tables/")
