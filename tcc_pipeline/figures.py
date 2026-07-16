"""
Publication figure generators (matplotlib only; no geospatial dependency).
 
These produce the journal-page-limit-friendly figures the manuscript relies on
instead of dozens of per-district time series:
  * zone envelope plots  - forecast spread (min/mean/max across member districts)
  * validation overlays  - observed 2025 vs projected 2025 for exemplar districts
 
Choropleths live in choropleth.py because they require the district geojson.
"""
from __future__ import annotations
 
import os
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from . import config as C
from . import indices as ix
 
INDEX_LABELS = {
    "heat_index": "Heat Index (degC)",
    "wet_bulb": "Wet-Bulb Temperature (degC)",
    "wbgt_shade": "WBGT - Shade (degC)",
    "wbgt_sun": "WBGT - Sun (degC)",
    "cdd": "Cooling Degree Days",
    "heatstroke_hazard": "Composite Heatstroke Hazard Score",
    "temperature": "Temperature (degC)",
    "humidity": "Relative Humidity (%)",
}
 
 
def zone_envelope_plot(district_results, zone_name, variable, out_path):
    """Envelope of the projected `variable` across all districts in a zone."""
    frames = []
    for dr in district_results:
        if dr.zone != zone_name or dr.forecast_indices is None:
            continue
        src = dr.forecast_indices if variable in dr.forecast_indices.columns else dr.forecast_base
        if variable not in src.columns:
            continue
        frames.append(src.set_index("date")[variable].rename(dr.district))
    if not frames:
        return None
    mat = pd.concat(frames, axis=1)
    lo, mean, hi = mat.min(axis=1), mat.mean(axis=1), mat.max(axis=1)
 
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(mat.index, lo, hi, alpha=0.22, color="#c0392b",
                    label="District spread (min-max)")
    ax.plot(mat.index, mean, color="#c0392b", lw=2.2, label="Zone mean")
    for col in mat.columns:
        ax.plot(mat.index, mat[col], color="#7f8c8d", lw=0.7, alpha=0.5)
    ax.set_title(f"{zone_name}\nProjected {INDEX_LABELS.get(variable, variable)}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel(INDEX_LABELS.get(variable, variable))
    ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path
 
 
def validation_overlay_plot(district_result, variable, out_path, obs_2025=None):
    """Observed 2025 vs projected 2025 for one district and one index."""
    dr = district_result
    src = dr.forecast_indices if (dr.forecast_indices is not None
                                  and variable in dr.forecast_indices.columns) else dr.forecast_base
    if src is None or variable not in src.columns:
        return None
    proj = src.set_index("date")[variable]
    proj_2025 = proj[proj.index.year == 2025]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(proj_2025.index, proj_2025.values, "o-", color="#2980b9",
            lw=2, label="Projected 2025")
    if obs_2025 is not None and variable in obs_2025.columns:
        o = obs_2025.set_index("date")[variable]
        o = o[o.index.year == 2025]
        ax.plot(o.index, o.values, "s--", color="#e67e22", lw=2, label="Observed 2025")
    ax.set_title(f"{dr.district}: {INDEX_LABELS.get(variable, variable)} - "
                 f"projected vs observed 2025", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel(INDEX_LABELS.get(variable, variable))
    ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path
 
 
def zone_month_heatmap(zone_summary_long, variable, out_path):
    """Heatmap of a monthly index across zones (zone x month)."""
    sub = zone_summary_long[zone_summary_long["variable"] == variable]
    if sub.empty:
        return None
    pivot = sub.pivot_table(index="zone_name", columns="month", values="value", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([z.split(".")[0] + "." + z.split(".")[1].split("(")[0][:18]
                        for z in pivot.index], fontsize=8)
    ax.set_xlabel("Month"); ax.set_title(f"Zonal monthly {INDEX_LABELS.get(variable, variable)}",
                                         fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def _daily_heat_index(csv_path, year_lo, year_hi):
    """Daily heat-index series from a raw district CSV, restricted to a year span."""
    raw = pd.read_csv(csv_path)
    raw["date"] = pd.to_datetime(raw[["year", "month", "day"]])
    raw = raw[(raw["date"].dt.year >= year_lo) & (raw["date"].dt.year <= year_hi)].copy()
    t = raw[C.BASE_VARIABLES["temperature"]].values
    rh = raw[C.BASE_VARIABLES["humidity"]].values
    raw["hi"] = ix.compute_heat_index(t, rh)
    return raw[["date", "hi"]].dropna()


def heat_index_three_panel(forecast_csv, hist_csv, obs2025_csv, district, out_path):
    """Three-panel heat-index narrative for an exemplar district:

      1. 2014-2024 observed daily heat index (the noisy record the model learns from),
         with the monthly mean overlaid.
      2. 2025: observed daily heat index (solid) vs the monthly projection (dashed)
         with 68/95% confidence bands -- the external-validation year.
      3. 2026: pure monthly projection with 68/95% confidence cloud, tilted by the
         re-added 1980-2024 Sen slope.
    """
    fc = pd.read_csv(forecast_csv, parse_dates=["date"])
    need = ["heat_index", "heat_index_lo95", "heat_index_hi95",
            "heat_index_lo68", "heat_index_hi68"]
    if any(c not in fc.columns for c in need):
        return None
    p25 = fc[fc["date"].dt.year == 2025]
    p26 = fc[fc["date"].dt.year == 2026]

    hist = _daily_heat_index(hist_csv, 2014, 2024)
    hist_m = hist.set_index("date")["hi"].resample("MS").mean()
    obs = _daily_heat_index(obs2025_csv, 2025, 2025) if obs2025_csv else None

    fig, (a1, a2, a3) = plt.subplots(
        1, 3, figsize=(16, 5), sharey=True,
        gridspec_kw={"width_ratios": [4.0, 1.6, 1.6], "wspace": 0.08})

    proj_c, obs_c, band_c = "#c0392b", "#e67e22", "#c0392b"

    # Panel 1: 2014-2024 daily record + monthly mean.
    a1.plot(hist["date"], hist["hi"], color="#95a5a6", lw=0.4, alpha=0.7,
            label="Observed daily")
    a1.plot(hist_m.index, hist_m.values, color="#2c3e50", lw=1.4, label="Monthly mean")
    a1.set_title("2014-2024  observed (training record)", fontsize=11, fontweight="bold")
    a1.set_ylabel(INDEX_LABELS["heat_index"]); a1.set_xlabel("Year")
    a1.grid(alpha=0.3, ls="--"); a1.legend(fontsize=8, loc="upper left")

    # Panel 2: 2025 observed daily vs projected monthly + bands.
    a2.fill_between(p25["date"], p25["heat_index_lo95"], p25["heat_index_hi95"],
                    color=band_c, alpha=0.15, label="95% CI")
    a2.fill_between(p25["date"], p25["heat_index_lo68"], p25["heat_index_hi68"],
                    color=band_c, alpha=0.28, label="68% CI")
    if obs is not None and not obs.empty:
        a2.plot(obs["date"], obs["hi"], color=obs_c, lw=1.0, alpha=0.9,
                label="Observed daily")
    a2.plot(p25["date"], p25["heat_index"], "o--", color=proj_c, lw=1.8,
            ms=4, label="Projected monthly")
    a2.set_title("2025  validation", fontsize=11, fontweight="bold")
    a2.set_xlabel("Month"); a2.grid(alpha=0.3, ls="--"); a2.legend(fontsize=7, loc="upper left")

    # Panel 3: 2026 projection + confidence cloud.
    a3.fill_between(p26["date"], p26["heat_index_lo95"], p26["heat_index_hi95"],
                    color=band_c, alpha=0.15, label="95% CI")
    a3.fill_between(p26["date"], p26["heat_index_lo68"], p26["heat_index_hi68"],
                    color=band_c, alpha=0.28, label="68% CI")
    a3.plot(p26["date"], p26["heat_index"], "o--", color=proj_c, lw=1.8,
            ms=4, label="Projected monthly")
    a3.set_title("2026  projection", fontsize=11, fontweight="bold")
    a3.set_xlabel("Month"); a3.grid(alpha=0.3, ls="--"); a3.legend(fontsize=7, loc="upper left")

    # Month-abbreviation ticks (year is already in each panel title).
    for ax in (a2, a3):
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    fig.suptitle(f"{district}: heat-index record, validation and projection",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path