#!/usr/bin/env python3
"""
TCC Bangladesh - seasonal-projection pipeline (single entry point).
 
Runs the entire deterministic workflow end to end and writes every artifact the
manuscript and supplementary need. No notebooks, no manual spreadsheet editing.
 
Usage
-----
  python run_pipeline.py                 # all 64 districts
  python run_pipeline.py --districts Dhaka Rajshahi Sylhet   # subset (smoke test)
  python run_pipeline.py --no-trend --no-figures            # skip slow extras
  python run_pipeline.py --outdir outputs                    # output location
 
Outputs (under --outdir)
------------------------
  forecasts/<district>_forecast.csv        24-month base + index projections + CI
  metrics/model_selection_long.csv         every district x variable x model row
  metrics/best_model_test_metrics.csv      SAW winner per district x variable
  metrics/external_validation_2025.csv     projected-vs-observed 2025 accuracy
  metrics/skill_scores.csv                 skill vs seasonal-naive & climatology
  trends/mann_kendall_sen_1980_2024.csv    observational trend test
  summary/zone_summary_monthly.csv         zonal monthly index envelopes
  summary/zone_summary_annual.csv          zonal annual means
  supplementary/all_district_index_metrics.csv
  figures/*.png                            zone envelopes, validation overlays, heatmap
  TCC_metrics_workbook.xlsx                multi-sheet reviewer-facing workbook
  run_manifest.json                        provenance (versions, config, timings)
"""
from __future__ import annotations
 
import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
 
import numpy as np
import pandas as pd
 
warnings.filterwarnings("ignore")
 
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
 
from tcc_pipeline import config as C
from tcc_pipeline import indices as ix
from tcc_pipeline import figures as fig
from tcc_pipeline.engine import run_district
from tcc_pipeline.trend import district_trends
 
HIST_DIR = os.path.join(HERE, "1980-2024-dataset")
OBS2025_DIR = os.path.join(HERE, "2025-dataset")
ASSET_DIR = os.path.join(HERE, "tcc_pipeline", "assets")
 
INDEX_COLS = ["heat_index", "wet_bulb", "wbgt_shade", "wbgt_sun", "heatstroke_hazard", "cdd"]
 
 
def hist_path(d):    return os.path.join(HIST_DIR, f"{d}_historical_weather_1980_2024.csv")
def obs2025_path(d): return os.path.join(OBS2025_DIR, f"{d}_historical_weather_2025.csv")
 
 
def load_zones():
    z = pd.read_csv(os.path.join(ASSET_DIR, "zones.csv"))
    return z, dict(zip(z["district"], z["zone_name"]))
 
 
def add_uncertainty(forecast_indices: pd.DataFrame, rmse_map: dict) -> pd.DataFrame:
    """Attach 68%/95% bands to key indices via analytical error propagation."""
    out = forecast_indices.copy()
    s_hi = ix.sigma_heat_index(rmse_map.get("temperature", np.nan), rmse_map.get("humidity", np.nan))
    s_wb = ix.sigma_wet_bulb(rmse_map.get("temperature", np.nan), rmse_map.get("humidity", np.nan))
    s_wbgt = ix.sigma_wbgt(rmse_map.get("temperature", np.nan), rmse_map.get("humidity", np.nan),
                           rmse_map.get("solar_radiation", np.nan))
    sig = {"heat_index": s_hi, "wet_bulb": s_wb, "wbgt_shade": s_wbgt, "wbgt_sun": s_wbgt}
    for var, s in sig.items():
        if var in out.columns:
            out[f"{var}_lo95"] = out[var] - 1.96 * s
            out[f"{var}_hi95"] = out[var] + 1.96 * s
            out[f"{var}_lo68"] = out[var] - s
            out[f"{var}_hi68"] = out[var] + s
    return out
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--districts", nargs="*", default=None, help="Subset (default: all 64)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--no-trend", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
 
    t_start = time.time()
    zones_df, zone_map = load_zones()
    all_districts = sorted(zones_df["district"].tolist())
    districts = args.districts or all_districts
    exemplars = set(zones_df[zones_df["is_exemplar"]]["district"])
 
    for sub in ["forecasts", "metrics", "trends", "summary", "supplementary", "figures"]:
        os.makedirs(os.path.join(args.outdir, sub), exist_ok=True)
 
    print(f"[TCC] Running {len(districts)} districts -> {args.outdir}")
    model_rows, best_rows, val_rows, skill_rows, trend_frames = [], [], [], [], []
    zone_month_long, zone_annual_rows = [], []
    district_results = []
 
    for i, d in enumerate(districts, 1):
        if not os.path.exists(hist_path(d)):
            print(f"  ! [{i}/{len(districts)}] {d}: missing history CSV, skipped")
            continue
        t0 = time.time()
        obs = obs2025_path(d) if os.path.exists(obs2025_path(d)) else None
        dr = run_district(d, hist_path(d), obs2025_csv=obs, zone=zone_map.get(d))
        district_results.append(dr)
 
        rmse_map = {v: r.final_rmse_full for v, r in dr.variable_results.items()}
 
        # Per-variable metric rows.
        for var, vr in dr.variable_results.items():
            for _, mrow in vr.metrics_table.iterrows():
                model_rows.append({"district": d, "zone": dr.zone, "variable": var,
                                   **{k: mrow[k] for k in ["Model", "R2", "CV_R2", "RMSE",
                                                           "Gen_Gap", "Composite_Score", "Status"]}})
            best_rows.append({"district": d, "zone": dr.zone, "variable": var,
                              "best_model": vr.best_model, "test_R2": vr.test_r2,
                              "test_RMSE": vr.test_rmse, "CV_R2": vr.cv_r2, "gen_gap": vr.gen_gap,
                              **{f"acc_{t}": vr.tol_accuracy[t] for t in vr.tol_accuracy}})
            skill_rows.append({"district": d, "zone": dr.zone, "variable": var,
                               "skill_vs_naive": vr.skill_vs_naive, "skill_vs_clim": vr.skill_vs_clim,
                               **vr.baseline_metrics})
 
        # Forecast CSV with uncertainty.
        fwide = dr.forecast_base.merge(
            dr.forecast_indices[["date"] + [c for c in INDEX_COLS if c in dr.forecast_indices.columns]],
            on="date")
        fwide = add_uncertainty(fwide, rmse_map)
        fwide.insert(0, "district", d); fwide.insert(1, "zone", dr.zone)
        fwide.to_csv(os.path.join(args.outdir, "forecasts", f"{d}_forecast.csv"), index=False)
 
        # External validation.
        if dr.external_validation is not None and not dr.external_validation.empty:
            ev = dr.external_validation.copy(); ev["zone"] = dr.zone
            val_rows.append(ev)
 
        # Zone summary contributions (projected indices, monthly + annual).
        fi = dr.forecast_indices.copy()
        fi["month"] = pd.to_datetime(fi["date"]).dt.month
        for var in INDEX_COLS + ["temperature", "humidity"]:
            if var not in fi.columns:
                continue
            for m, val in fi.groupby("month")[var].mean().items():
                zone_month_long.append({"zone_name": dr.zone, "district": d,
                                        "variable": var, "month": int(m), "value": float(val)})
            zone_annual_rows.append({"zone_name": dr.zone, "district": d,
                                     "variable": var, "annual_mean": float(fi[var].mean())})
 
        tag = " (exemplar)" if d in exemplars else ""
        print(f"  + [{i}/{len(districts)}] {d}{tag}: {time.time()-t0:4.1f}s  "
              f"temp R2={dr.variable_results['temperature'].test_r2:.3f}")
 
    # ---- assemble tables ----
    model_df = pd.DataFrame(model_rows)
    best_df = pd.DataFrame(best_rows)
    skill_df = pd.DataFrame(skill_rows)
    val_df = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()
    zm_long = pd.DataFrame(zone_month_long)
    za_df = pd.DataFrame(zone_annual_rows)
 
    model_df.to_csv(os.path.join(args.outdir, "metrics", "model_selection_long.csv"), index=False)
    best_df.to_csv(os.path.join(args.outdir, "metrics", "best_model_test_metrics.csv"), index=False)
    skill_df.to_csv(os.path.join(args.outdir, "metrics", "skill_scores.csv"), index=False)
    if not val_df.empty:
        val_df.to_csv(os.path.join(args.outdir, "metrics", "external_validation_2025.csv"), index=False)
 
    # Zone summaries.
    zone_monthly = (zm_long.groupby(["zone_name", "variable", "month"])["value"]
                    .agg(["mean", "min", "max"]).reset_index()) if not zm_long.empty else pd.DataFrame()
    zone_annual = (za_df.groupby(["zone_name", "variable"])["annual_mean"]
                   .agg(["mean", "min", "max"]).reset_index()) if not za_df.empty else pd.DataFrame()
    if not zone_monthly.empty:
        zone_monthly.to_csv(os.path.join(args.outdir, "summary", "zone_summary_monthly.csv"), index=False)
        zm_long.to_csv(os.path.join(args.outdir, "summary", "zone_summary_monthly_long.csv"), index=False)
    if not zone_annual.empty:
        zone_annual.to_csv(os.path.join(args.outdir, "summary", "zone_summary_annual.csv"), index=False)
 
    # Supplementary: full per-district index validation.
    if not val_df.empty:
        val_df[val_df["kind"] == "index"].to_csv(
            os.path.join(args.outdir, "supplementary", "all_district_index_metrics.csv"), index=False)
 
    # ---- trend analysis ----
    if not args.no_trend:
        print("[TCC] Mann-Kendall / Sen's slope on 1980-2024 ...")
        for d in districts:
            if os.path.exists(hist_path(d)):
                trend_frames.append(district_trends(hist_path(d), d))
        if trend_frames:
            trend_df = pd.concat(trend_frames, ignore_index=True)
            trend_df["zone"] = trend_df["district"].map(zone_map)
            trend_df.to_csv(os.path.join(args.outdir, "trends",
                                         "mann_kendall_sen_1980_2024.csv"), index=False)
 
    # ---- figures ----
    if not args.no_figures and district_results:
        print("[TCC] Rendering figures ...")
        fdir = os.path.join(args.outdir, "figures")
        for zname in sorted({dr.zone for dr in district_results if dr.zone}):
            for var in ["heat_index", "wbgt_sun", "cdd"]:
                fig.zone_envelope_plot(district_results, zname, var,
                                       os.path.join(fdir, f"zone{zname.split('.')[0]}_{var}.png"))
        for dr in district_results:
            if dr.district in exemplars and dr.external_validation is not None:
                obs = None
                if os.path.exists(obs2025_path(dr.district)):
                    from tcc_pipeline.engine import load_monthly
                    om = load_monthly(obs2025_path(dr.district), 2025, 2025)
                    obs = ix.recompute_indices_from_base(om.reset_index())
                fig.validation_overlay_plot(dr, "heat_index",
                                            os.path.join(fdir, f"val_{dr.district}_heat_index.png"), obs)
                # Three-panel heat-index narrative (record / validation / projection).
                fc_csv = os.path.join(args.outdir, "forecasts", f"{dr.district}_forecast.csv")
                if os.path.exists(fc_csv) and os.path.exists(hist_path(dr.district)):
                    fig.heat_index_three_panel(
                        fc_csv, hist_path(dr.district),
                        obs2025_path(dr.district) if os.path.exists(obs2025_path(dr.district)) else None,
                        dr.district, os.path.join(fdir, f"threepanel_{dr.district}_heat_index.png"))
        if not zm_long.empty:
            fig.zone_month_heatmap(zm_long, "heat_index",
                                   os.path.join(fdir, "heatmap_zone_month_heat_index.png"))
 
    # ---- reviewer workbook ----
    xlsx = os.path.join(args.outdir, "TCC_metrics_workbook.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        _overview(districts, best_df, val_df).to_excel(xw, sheet_name="Overview", index=False)
        best_df.to_excel(xw, sheet_name="Test_Metrics", index=False)
        model_df.to_excel(xw, sheet_name="Model_Selection", index=False)
        if not val_df.empty:
            val_df.to_excel(xw, sheet_name="External_Validation_2025", index=False)
        skill_df.to_excel(xw, sheet_name="Skill_vs_Baselines", index=False)
        if not zone_monthly.empty:
            zone_monthly.to_excel(xw, sheet_name="Zone_Monthly", index=False)
        if not args.no_trend and trend_frames:
            trend_df.to_excel(xw, sheet_name="Trends_1980_2024", index=False)
 
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_districts": len(district_results),
        "districts": [dr.district for dr in district_results],
        "runtime_seconds": round(time.time() - t_start, 1),
        "config": {"train_years": C.TRAIN_YEARS, "test_years": C.TEST_YEARS,
                   "ml_window": [C.ML_START_YEAR, C.ML_END_YEAR],
                   "forecast_months": C.FORECAST_MONTHS, "fourier_harmonics": C.FOURIER_HARMONICS,
                   "models": C.MODEL_ORDER, "saw_weights": C.SAW_WEIGHTS},
        "methodology_notes": {
            "projection_trend": ("Robust 1980-2024 Sen slope per district/variable "
                                 "(solar_radiation, uv fall back to their 2014-2024 record); "
                                 "trees model the seasonal residual and the observed climate "
                                 "trend is re-added at projection time (future-known features "
                                 "only). Included for consistency with the Mann-Kendall trend "
                                 "analysis; it does not materially improve near-term test skill "
                                 "because the monthly seasonal cycle dominates."),
            "heatstroke_hazard": ("Excluded from validation/skill results: negative "
                                  "out-of-sample R2 (worse than a constant). Retained as a "
                                  "descriptive projected column only."),
        },
        "versions": _versions(),
    }
    with open(os.path.join(args.outdir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
 
    print(f"[TCC] Done in {manifest['runtime_seconds']}s. Workbook: {xlsx}")
 
 
def _overview(districts, best_df, val_df):
    rows = []
    if not best_df.empty:
        for var in best_df["variable"].unique():
            sub = best_df[best_df["variable"] == var]
            rows.append({"scope": "test", "variable": var,
                         "median_R2": round(sub["test_R2"].median(), 3),
                         "median_RMSE": round(sub["test_RMSE"].median(), 3),
                         "top_model": sub["best_model"].mode().iloc[0]})
    if val_df is not None and not val_df.empty:
        for var in val_df["variable"].unique():
            sub = val_df[val_df["variable"] == var]
            rows.append({"scope": "val2025", "variable": var,
                         "median_R2": round(sub["val_R2"].median(), 3),
                         "median_RMSE": round(sub["val_RMSE"].median(), 3), "top_model": ""})
    return pd.DataFrame(rows)
 
 
def _versions():
    import sklearn, lightgbm, xgboost, catboost
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
            "xgboost": xgboost.__version__, "catboost": catboost.__version__}
 
 
if __name__ == "__main__":
    main()