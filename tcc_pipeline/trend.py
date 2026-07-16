
"""
Observational trend analysis (Mann-Kendall + Sen's slope) on the full
1980-2024 record. This is descriptive statistics on observations, NOT
forecasting and NOT model training, so it introduces no target leakage. It is
computed on annual-mean series and strengthens the policy narrative
(long-run warming / humidification signals per district and zone).
 
Self-contained implementation (normal approximation with tie correction) so the
pipeline has no extra hard dependency.
"""
from __future__ import annotations
 
from dataclasses import dataclass
 
import numpy as np
import pandas as pd
 
from . import config as C
from . import indices as ix
 
 
@dataclass
class TrendResult:
    variable: str
    n: int
    tau: float
    s: float
    z: float
    p_value: float
    sen_slope: float       # units per year
    trend: str             # "increasing" / "decreasing" / "no trend"
 
 
def _mann_kendall(x: np.ndarray) -> tuple[float, float, float]:
    """Return (S, Z, tau) for the Mann-Kendall test with tie correction."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = 0.0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k + 1:] - x[k]))
 
    # Variance with tie correction.
    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
 
    if var_s <= 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
 
    tau = s / (0.5 * n * (n - 1)) if n > 1 else 0.0
    return s, z, tau
 
 
def _sen_slope(x: np.ndarray, t: np.ndarray) -> float:
    """Theil-Sen median slope (units of x per unit of t)."""
    n = len(x)
    slopes = []
    for i in range(n - 1):
        dt = t[i + 1:] - t[i]
        dx = x[i + 1:] - x[i]
        valid = dt != 0
        slopes.extend((dx[valid] / dt[valid]).tolist())
    return float(np.median(slopes)) if slopes else np.nan
 
 
def _norm_sf(z: float) -> float:
    """Two-sided p-value from a standard normal (erf-based, no scipy)."""
    from math import erf, sqrt
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
 
 
def mann_kendall_sen(series: pd.Series, alpha: float = 0.05, name: str = "") -> TrendResult:
    s_clean = series.dropna()
    years = s_clean.index.values.astype(float)
    vals = s_clean.values.astype(float)
    n = len(vals)
    if n < 4:
        return TrendResult(name, n, np.nan, np.nan, np.nan, np.nan, np.nan, "insufficient")
    s, z, tau = _mann_kendall(vals)
    p = _norm_sf(z)
    slope = _sen_slope(vals, years)
    if p < alpha and slope > 0:
        trend = "increasing"
    elif p < alpha and slope < 0:
        trend = "decreasing"
    else:
        trend = "no trend"
    return TrendResult(name, n, float(tau), float(s), float(z), float(p), slope, trend)
 
 
def annual_series_from_csv(csv_path: str) -> pd.DataFrame:
    """Annual-mean series (1980-2024) of base variables + recomputed HI/WB/WBGT.
    HI and WB need only temperature+humidity, both available from 1980, so their
    long trends are meaningful across the whole record."""
    raw = pd.read_csv(csv_path)
    raw["date"] = pd.to_datetime(raw[["year", "month", "day"]])
    inv = {v: k for k, v in C.BASE_VARIABLES.items()}
    raw = raw.rename(columns=inv)
    raw = raw[(raw["date"].dt.year >= C.TREND_START_YEAR) & (raw["date"].dt.year <= C.TREND_END_YEAR)]

    # Mask missing-as-zero early-record values so they cannot fabricate a trend.
    for _var, _floor in C.TREND_VALID_FLOORS.items():
        if _var in raw.columns:
            raw.loc[raw[_var] < _floor, _var] = np.nan
 
    # Recompute temperature-only-and-humidity indices at daily resolution.
    raw["heat_index"] = ix.heat_index(raw["temperature"], raw["humidity"])
    raw["wet_bulb"] = ix.wet_bulb(raw["temperature"], raw["humidity"])
    raw["wbgt_shade"] = ix.wbgt_shade(raw["temperature"], raw["wet_bulb"])
 
    cols = ["temperature", "tmax", "tmin", "humidity", "dew_point",
            "heat_index", "wet_bulb", "wbgt_shade"]
    grp = raw.groupby(raw["date"].dt.year)[cols]
    annual = grp.mean()
    # A year needs a minimum number of real daily observations to be trusted.
    annual = annual.where(grp.count() >= C.TREND_MIN_VALID_DAYS)
    annual.index.name = "year"
    return annual
 
 
def district_trends(csv_path: str, district: str) -> pd.DataFrame:
    annual = annual_series_from_csv(csv_path)
    rows = []
    for col in annual.columns:
        tr = mann_kendall_sen(annual[col], name=col)
        rows.append({
            "district": district,
            "variable": col,
            "n_years": tr.n,
            "MK_tau": tr.tau,
            "MK_Z": tr.z,
            "p_value": tr.p_value,
            "sen_slope_per_year": tr.sen_slope,
            "sen_slope_per_decade": tr.sen_slope * 10 if not np.isnan(tr.sen_slope) else np.nan,
            "trend": tr.trend,
        })
    return pd.DataFrame(rows)
 
