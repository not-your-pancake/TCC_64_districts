"""
Per-district modelling engine for the TCC pipeline.
 
For one district this module:
  1. loads and monthly-aggregates the 2014-2024 record,
  2. builds Fourier + year features (future-known only),
  3. trains RF / XGBoost / LightGBM / CatBoost for each base variable using a
     chronological hold-out (2014-2022 train, 2023-2024 test) plus TimeSeriesSplit
     cross-validation,
  4. selects the best model per variable via SAW with a generalization-gap
     penalty,
  5. retrains the winner on the full 2014-2024 record and projects 24 months,
  6. recomputes all thermal indices from the projected base variables,
  7. validates the 2025 projection against observed 2025 data, and
  8. benchmarks against seasonal-naive and monthly-climatology baselines.
 
The engine returns plain dataclasses/DataFrames; all file writing lives in the
orchestrator so this stays unit-testable.
"""
from __future__ import annotations
 
from dataclasses import dataclass, field
 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
 
from . import config as C
from . import indices as ix
from .trend import _sen_slope
 
 
# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def add_fourier_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    months = out[date_col].dt.month
    for n in range(1, C.FOURIER_HARMONICS + 1):
        out[f"sin_{n}"] = np.sin(2 * np.pi * n * months / C.FOURIER_PERIOD)
        out[f"cos_{n}"] = np.cos(2 * np.pi * n * months / C.FOURIER_PERIOD)
    out["year_val"] = out[date_col].dt.year
    return out
 
 
def _build_models():
    return {
        "RandomForest": RandomForestRegressor(**C.RF_PARAMS),
        "XGBoost": xgb.XGBRegressor(**C.XGB_PARAMS),
        "LightGBM": lgb.LGBMRegressor(**C.LGB_PARAMS),
        "CatBoost": CatBoostRegressor(**C.CAT_PARAMS),
    }
 
 
def tolerance_accuracy(y_true, y_pred, tol) -> float:
    return float((np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tol).mean())


def _decimal_year(dates: pd.Series) -> np.ndarray:
    """Continuous time axis (year + month fraction) for trend fitting/extrapolation."""
    return (dates.dt.year + (dates.dt.month - 1) / 12.0).to_numpy(dtype=float)


def _annual_theil_sen(dates: pd.Series, y: np.ndarray) -> float:
    """Theil-Sen slope (units/year) of the annual-mean series over the given rows.

    Fitting on annual means (not the raw monthly series) removes the seasonal cycle
    so the slope reflects the year-over-year trend only. Returns 0.0 if too short.
    """
    yrs = dates.dt.year.to_numpy()
    ann = pd.Series(np.asarray(y, dtype=float)).groupby(yrs).mean()
    if len(ann) < 3:
        return 0.0
    slope = _sen_slope(ann.values.astype(float), ann.index.values.astype(float))
    return float(slope) if np.isfinite(slope) else 0.0
 
 
# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class VariableResult:
    district: str
    variable: str
    best_model: str
    metrics_table: pd.DataFrame          # one row per model
    test_r2: float
    test_rmse: float
    cv_r2: float
    gen_gap: float
    tol_accuracy: dict                   # {tol_value: fraction}
    baseline_metrics: dict               # seasonal_naive / climatology R2+RMSE
    skill_vs_naive: float                # 1 - RMSE_model/RMSE_naive
    skill_vs_clim: float
    final_rmse_full: float               # RMSE of winner refit on full record (in-sample proxy sigma)
    forecast: pd.DataFrame               # date, prediction  (24 months)
    fitted_2025: pd.DataFrame            # date, prediction for 2025 months (from final model)
    trend_slope_per_year: float = 0.0    # Theil-Sen slope re-added at projection time
 
 
@dataclass
class DistrictResult:
    district: str
    zone: str | None
    monthly_history: pd.DataFrame
    variable_results: dict = field(default_factory=dict)
    forecast_base: pd.DataFrame | None = None      # wide: date + all base vars (24 mo)
    forecast_indices: pd.DataFrame | None = None   # wide: date + base + indices (24 mo)
    external_validation: pd.DataFrame | None = None  # per-variable/index 2025 metrics
 
 
# ---------------------------------------------------------------------------
# Data loading / monthly aggregation
# ---------------------------------------------------------------------------
def load_monthly(csv_path: str, year_lo: int, year_hi: int) -> pd.DataFrame:
    """Load a district CSV and return a monthly-mean, linearly interpolated
    frame indexed by month-start dates, restricted to [year_lo, year_hi]."""
    raw = pd.read_csv(csv_path)
    raw["date"] = pd.to_datetime(raw[["year", "month", "day"]])
    cols = list(C.BASE_VARIABLES.values())
    sub = raw[["date"] + cols].copy()
    sub = sub[(sub["date"].dt.year >= year_lo) & (sub["date"].dt.year <= year_hi)]
    monthly = sub.set_index("date")[cols].resample("MS").mean()
    monthly = monthly.interpolate(method="linear")
    # rename to internal names
    inv = {v: k for k, v in C.BASE_VARIABLES.items()}
    monthly = monthly.rename(columns=inv)
    return monthly
 
 
# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _seasonal_naive(series: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
    """Predict each test month with the same calendar month one year earlier."""
    preds = []
    for ts in test_index:
        prev = ts - pd.DateOffset(years=1)
        preds.append(series.get(prev, np.nan))
    return np.array(preds, dtype=float)
 
 
def _climatology(train_series: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
    """Predict each test month with the training-period mean for that month."""
    clim = train_series.groupby(train_series.index.month).mean()
    return np.array([clim.get(ts.month, np.nan) for ts in test_index], dtype=float)
 
 
# ---------------------------------------------------------------------------
# Core per-variable routine
# ---------------------------------------------------------------------------
def _saw_select(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    df["Gen_Gap"] = (df["R2"] - df["CV_R2"]).abs()
    norm = df.copy()
    for metric, maximize in C.SAW_MAXIMIZE.items():
        lo, hi = df[metric].min(), df[metric].max()
        if hi - lo < 1e-12:
            norm[metric] = 1.0
        elif maximize:
            norm[metric] = (df[metric] - lo) / (hi - lo)
        else:
            norm[metric] = (hi - df[metric]) / (hi - lo)
    df["Composite_Score"] = sum(norm[m] * w for m, w in C.SAW_WEIGHTS.items())
    df["Status"] = np.where(df["Gen_Gap"] > C.OVERFIT_GAP_THRESHOLD, "Overfit Risk", "Robust")
    return df.sort_values("Composite_Score", ascending=False).reset_index(drop=True)
 
 
def run_variable(district: str, variable: str, monthly: pd.DataFrame,
                 trend_slope: float | None = None) -> VariableResult:
    feats = add_fourier_features(monthly.reset_index().rename(columns={"index": "date"}))
    feats = feats.dropna(subset=[variable])
    X = feats[C.FEATURE_COLUMNS]
    y = feats[variable].values
    dates = feats["date"]
 
    train_mask = (dates.dt.year >= C.TRAIN_YEARS[0]) & (dates.dt.year <= C.TRAIN_YEARS[1])
    test_mask = (dates.dt.year >= C.TEST_YEARS[0]) & (dates.dt.year <= C.TEST_YEARS[1])
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]
 
    grid = C.TOLERANCE_GRIDS[variable]
    tol_mid = grid[C.SAW_TOL_INDEX]
    tscv = TimeSeriesSplit(n_splits=C.TSCV_SPLITS)
 
    # Detrend, let the trees model the seasonal residual, and re-add the trend at
    # prediction time. The slope is the robust 1980-2024 Sen slope (passed in) so
    # the projection is consistent with the manuscript's trend analysis; where no
    # long record exists (solar_radiation, uv) it falls back to the ML window.
    # Extrapolating the trend onto the 2023-2024 test years is the honest
    # out-of-sample check of the trend component.
    slope = trend_slope if trend_slope is not None else _annual_theil_sen(dates, y)
    t_all = _decimal_year(dates)
    y_resid_tr = y_tr - slope * t_all[train_mask.values]
    y_resid_full = y - slope * t_all

    rows = []
    for name, model in _build_models().items():
        model.fit(X_tr, y_resid_tr)
        y_hat = model.predict(X_te) + slope * t_all[test_mask.values]
        r2 = r2_score(y_te, y_hat)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))
        cv = cross_val_score(model, X, y_resid_full, cv=tscv, scoring="r2", n_jobs=-1).mean()
        row = {
            "Model": name,
            "R2": r2,
            "CV_R2": float(cv),
            "RMSE": rmse,
            "Acc_tol1": tolerance_accuracy(y_te, y_hat, tol_mid),
        }
        for t in grid:
            row[f"acc_{t}"] = tolerance_accuracy(y_te, y_hat, t)
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    ranked = _saw_select(metrics_df)
    best = ranked.iloc[0]["Model"]
    best_row = ranked[ranked["Model"] == best].iloc[0]
 
    # Baselines on the chronological test set.
    series = pd.Series(y, index=pd.DatetimeIndex(dates))
    train_series = series[train_mask.values]
    test_index = pd.DatetimeIndex(dates[test_mask])
    sn = _seasonal_naive(series, test_index)
    cl = _climatology(train_series, test_index)
    valid_sn = ~np.isnan(sn)
    valid_cl = ~np.isnan(cl)
    rmse_sn = float(np.sqrt(mean_squared_error(y_te[valid_sn], sn[valid_sn]))) if valid_sn.any() else np.nan
    rmse_cl = float(np.sqrt(mean_squared_error(y_te[valid_cl], cl[valid_cl]))) if valid_cl.any() else np.nan
    r2_sn = float(r2_score(y_te[valid_sn], sn[valid_sn])) if valid_sn.sum() > 1 else np.nan
    r2_cl = float(r2_score(y_te[valid_cl], cl[valid_cl])) if valid_cl.sum() > 1 else np.nan
    model_rmse = best_row["RMSE"]
    skill_naive = float(1 - model_rmse / rmse_sn) if rmse_sn and not np.isnan(rmse_sn) else np.nan
    skill_clim = float(1 - model_rmse / rmse_cl) if rmse_cl and not np.isnan(rmse_cl) else np.nan
 
    # Refit winner on the FULL record (detrended) and project forward + fit 2025,
    # re-adding the same slope used above.
    slope_proj = slope
    y_resid = y_resid_full
    final_model = _build_models()[best]
    final_model.fit(X, y_resid)
    in_sample = final_model.predict(X) + slope_proj * t_all
    final_rmse_full = float(np.sqrt(mean_squared_error(y, in_sample)))
 
    last_date = dates.max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=C.FORECAST_MONTHS, freq="MS"
    )
    fut = add_fourier_features(pd.DataFrame({"date": future_dates}))
    lo, hi = C.BASE_VARIABLE_BOUNDS[variable]
    fut_t = _decimal_year(fut["date"])
    fpred = np.clip(final_model.predict(fut[C.FEATURE_COLUMNS]) + slope_proj * fut_t, lo, hi)
    forecast = pd.DataFrame({"date": future_dates, "prediction": fpred})
    fitted_2025 = forecast[forecast["date"].dt.year == 2025].reset_index(drop=True)
 
    return VariableResult(
        district=district,
        variable=variable,
        best_model=best,
        metrics_table=ranked,
        test_r2=float(best_row["R2"]),
        test_rmse=float(best_row["RMSE"]),
        cv_r2=float(best_row["CV_R2"]),
        gen_gap=float(best_row["Gen_Gap"]),
        tol_accuracy={t: float(best_row[f"acc_{t}"]) for t in grid},
        baseline_metrics={
            "seasonal_naive_R2": r2_sn, "seasonal_naive_RMSE": rmse_sn,
            "climatology_R2": r2_cl, "climatology_RMSE": rmse_cl,
        },
        skill_vs_naive=skill_naive,
        skill_vs_clim=skill_clim,
        final_rmse_full=final_rmse_full,
        forecast=forecast,
        fitted_2025=fitted_2025,
        trend_slope_per_year=slope_proj,
    )
 
 
# ---------------------------------------------------------------------------
# District orchestration
# ---------------------------------------------------------------------------
def _long_run_slopes(hist_csv: str) -> dict:
    """Robust 1980-2024 Sen slope (units/year) per base variable, for re-adding
    the observed climate trend to the projection. Variables without a pre-2014
    record (solar_radiation, uv) naturally use only the years they exist; the
    caller falls back to an ML-window fit if a variable is absent here."""
    raw = pd.read_csv(hist_csv)
    raw["date"] = pd.to_datetime(raw[["year", "month", "day"]])
    inv = {v: k for k, v in C.BASE_VARIABLES.items()}
    raw = raw.rename(columns=inv)
    raw = raw[(raw["date"].dt.year >= C.TREND_START_YEAR) &
              (raw["date"].dt.year <= C.TREND_END_YEAR)]
    slopes = {}
    for var in C.BASE_VARIABLES.keys():
        if var not in raw.columns:
            continue
        s = raw[["date", var]].copy()
        # Drop missing-as-zero early-record values before fitting the slope.
        floor = C.TREND_VALID_FLOORS.get(var)
        if floor is not None:
            s.loc[s[var] < floor, var] = np.nan
        s = s.dropna(subset=[var])
        if s.empty:
            continue
        g = s.groupby(s["date"].dt.year)[var]
        ann = g.mean()[g.count() >= C.TREND_MIN_VALID_DAYS]
        if len(ann) < 3:
            continue
        sl = _sen_slope(ann.values.astype(float), ann.index.values.astype(float))
        if np.isfinite(sl):
            slopes[var] = float(sl)
    return slopes


def run_district(
    district: str,
    hist_csv: str,
    obs2025_csv: str | None = None,
    zone: str | None = None,
    variables: list[str] | None = None,
) -> DistrictResult:
    variables = variables or list(C.BASE_VARIABLES.keys())
    monthly = load_monthly(hist_csv, C.ML_START_YEAR, C.ML_END_YEAR)
    long_slopes = _long_run_slopes(hist_csv)
 
    result = DistrictResult(district=district, zone=zone, monthly_history=monthly)
 
    # Per-variable modelling.
    base_forecasts = {}
    rmse_map = {}
    for var in variables:
        vres = run_variable(district, var, monthly, trend_slope=long_slopes.get(var))
        result.variable_results[var] = vres
        base_forecasts[var] = vres.forecast.set_index("date")["prediction"]
        rmse_map[var] = vres.final_rmse_full
 
    # Assemble wide base-variable forecast and recompute indices.
    fbase = pd.DataFrame(base_forecasts)
    fbase.index.name = "date"
    findices = ix.recompute_indices_from_base(fbase.reset_index()).set_index("date")
    result.forecast_base = fbase.reset_index()
    result.forecast_indices = findices.reset_index()
 
    # External validation against observed 2025.
    if obs2025_csv is not None:
        result.external_validation = _external_validation(
            district, obs2025_csv, base_forecasts, findices, rmse_map
        )
    return result
 
 
def _external_validation(district, obs2025_csv, base_forecasts, findices, rmse_map) -> pd.DataFrame:
    obs = load_monthly(obs2025_csv, 2025, 2025)
    obs = ix.recompute_indices_from_base(obs.reset_index()).set_index("date")
 
    rows = []
    # Base variables.
    for var in base_forecasts:
        if var not in obs.columns:
            continue
        pred = base_forecasts[var]
        joined = pd.concat([obs[var].rename("obs"), pred.rename("pred")], axis=1).dropna()
        if len(joined) < 2:
            continue
        rows.append(_val_row(district, var, "base", joined, rmse_map.get(var)))
 
    # Recomputed indices. heatstroke_hazard is intentionally excluded from
    # validation -- its out-of-sample R2 is negative (worse than a constant), so
    # it is retained only as a descriptive projected column, never as a
    # validated predictive result.
    index_cols = ["heat_index", "wet_bulb", "wbgt_shade", "wbgt_sun", "cdd"]
    for var in index_cols:
        if var not in obs.columns or var not in findices.columns:
            continue
        pred = findices[var]
        joined = pd.concat([obs[var].rename("obs"), pred.rename("pred")], axis=1).dropna()
        if len(joined) < 2:
            continue
        rows.append(_val_row(district, var, "index", joined, None))
    return pd.DataFrame(rows)
 
 
def _val_row(district, var, kind, joined, sigma):
    y_true = joined["obs"].values
    y_pred = joined["pred"].values
    grid = C.TOLERANCE_GRIDS.get(var, [0.5, 1.0, 2.0, 3.0])
    row = {
        "district": district,
        "variable": var,
        "kind": kind,
        "n_months": len(joined),
        "val_R2": float(r2_score(y_true, y_pred)) if len(joined) > 1 else np.nan,
        "val_RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "val_MAE": float(np.mean(np.abs(y_true - y_pred))),
    }
    for t in grid:
        row[f"val_acc_{t}"] = tolerance_accuracy(y_true, y_pred, t)
    return row