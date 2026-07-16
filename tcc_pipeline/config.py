"""
Central configuration for the TCC Bangladesh seasonal-projection pipeline.
 
Everything the manuscript treats as a *locked* methodological decision lives here
so a reviewer (or a future run) can audit the choices in one place. Nothing in
this file is derived at run time; these are fixed specification constants.
"""
from __future__ import annotations
 
# ----------------------------------------------------------------------------
# Temporal specification (locked)
# ----------------------------------------------------------------------------
# ML training uses the high-quality recent window only. Older data (1980-2013)
# are noisy for humidity and lack solar/UV entirely, so they are reserved for
# the Mann-Kendall / Sen's slope observational trend analysis, NOT for training.
ML_START_YEAR = 2014
ML_END_YEAR = 2024
 
# Chronological hold-out for reported test metrics (replaces shuffled split).
TRAIN_YEARS = (2014, 2022)   # inclusive
TEST_YEARS = (2023, 2024)    # inclusive
 
# TimeSeriesSplit cross-validation folds.
TSCV_SPLITS = 5
 
# Forward projection horizon (months) beyond the last training month.
FORECAST_MONTHS = 24         # 2025-01 .. 2026-12
 
# External validation window: observed data available for 2025.
# November 2025 is partial in the source files; we note this in reporting.
EXTERNAL_VAL_START = "2025-01-01"
EXTERNAL_VAL_END = "2025-11-30"
 
# Full observational record used ONLY for Mann-Kendall + Sen's slope.
TREND_START_YEAR = 1980
TREND_END_YEAR = 2024

# Some districts have large blocks of early-record (pre-~2000) days where the
# source encoded MISSING data as 0. Bangladesh daily values never reach these
# floors, so anything below is treated as missing before the trend / long-run
# Sen slope is fit -- otherwise the 0 -> ~30 C jump fabricates an impossible
# warming trend. A year must also retain at least TREND_MIN_VALID_DAYS real days
# to contribute to the annual-mean series.
TREND_VALID_FLOORS = {
    "temperature": 5.0,
    "tmax": 5.0,
    "tmin": 2.0,
    "dew_point": 2.0,
    "humidity": 5.0,
}
TREND_MIN_VALID_DAYS = 60
 
RANDOM_STATE = 42
 
# ----------------------------------------------------------------------------
# Base (physically primary) variables that the models predict directly.
# All thermal indices are RECALCULATED from these afterwards to avoid target
# leakage. Keys are pipeline-internal names; values are the raw CSV columns.
# ----------------------------------------------------------------------------
BASE_VARIABLES = {
    "temperature": "temperature(degree C)",
    "humidity": "humidity",
    "solar_radiation": "solar_radiation",
    "uv": "UV",
    "tmax": "max_temperature(degree C)",
    "tmin": "minimum_temperature(degree C)",
    "dew_point": "dew_point",
}
 
# Physical/plausibility clamps applied to *predicted* base variables before
# indices are recomputed (prevents nonsensical extrapolations feeding physics).
BASE_VARIABLE_BOUNDS = {
    "temperature": (0.0, 50.0),
    "humidity": (1.0, 100.0),
    "solar_radiation": (0.0, 400.0),
    "uv": (0.0, 16.0),
    "tmax": (0.0, 55.0),
    "tmin": (-5.0, 45.0),
    "dew_point": (-10.0, 35.0),
}
 
# ----------------------------------------------------------------------------
# Fourier seasonal features (monthly period, 3 harmonics). Only future-known
# features are permitted: calendar year + Fourier terms. No lags, no rolling
# windows, no autoregressive inputs.
# ----------------------------------------------------------------------------
FOURIER_PERIOD = 12
FOURIER_HARMONICS = 3
 
def fourier_feature_names() -> list[str]:
    names = []
    for n in range(1, FOURIER_HARMONICS + 1):
        names += [f"sin_{n}", f"cos_{n}"]
    return names
 
FEATURE_COLUMNS = ["year_val"] + fourier_feature_names()
 
# ----------------------------------------------------------------------------
# Model zoo. Configs are the fixed tuned settings carried over verbatim from the
# original notebooks and applied uniformly to every district/variable so results
# are comparable and the pipeline is deterministic.
# ----------------------------------------------------------------------------
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    min_samples_split=2,
    min_samples_leaf=7,
    max_features=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
 
XGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    reg_alpha=0.1,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
 
LGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    num_leaves=10,
    min_child_samples=5,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=0.2,
    reg_lambda=0.2,
    importance_type="gain",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)
 
CAT_PARAMS = dict(
    iterations=1500,
    learning_rate=0.01,
    depth=4,
    l2_leaf_reg=10,
    bootstrap_type="Bayesian",
    bagging_temperature=0.5,
    random_strength=2,
    loss_function="RMSE",
    od_type="Iter",
    od_wait=50,
    random_seed=RANDOM_STATE,
    verbose=0,
)
 
MODEL_ORDER = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
 
# ----------------------------------------------------------------------------
# SAW (Simple Additive Weighting) model-selection scheme with a
# generalization-gap penalty. Higher composite score wins per variable.
# ----------------------------------------------------------------------------
SAW_WEIGHTS = {
    "R2": 0.25,
    "CV_R2": 0.25,
    "RMSE": 0.20,        # minimized
    "Gen_Gap": 0.20,     # minimized  (|R2 - CV_R2|)
    "Acc_tol1": 0.10,
}
SAW_MAXIMIZE = {"R2": True, "CV_R2": True, "RMSE": False, "Gen_Gap": False, "Acc_tol1": True}
OVERFIT_GAP_THRESHOLD = 0.10   # Gen_Gap above this is flagged "Overfit Risk"
 
# The tolerance level (in each variable's native unit) used as the SAW accuracy
# term. For temperature-like variables this is 1.0 degC.
SAW_TOL_KEY = "tol_mid"
 
# ----------------------------------------------------------------------------
# Operational tolerance grids per variable (approved). Reported as the fraction
# of test points whose absolute error falls within each band, so accuracy is
# expressed operationally rather than only as raw R2/RMSE.
# ----------------------------------------------------------------------------
TOLERANCE_GRIDS = {
    "temperature": [0.5, 1.0, 2.0, 3.0],          # degC
    "tmax": [0.5, 1.0, 2.0, 3.0],
    "tmin": [0.5, 1.0, 2.0, 3.0],
    "dew_point": [0.5, 1.0, 2.0, 3.0],
    "humidity": [1.0, 2.0, 3.0, 5.0],             # percentage points
    "solar_radiation": [10.0, 20.0, 30.0, 50.0],  # W/m^2
    "uv": [0.5, 1.0, 1.5, 2.0],                   # UV index units
    # Recomputed indices reuse a temperature-like grid.
    "heat_index": [0.5, 1.0, 2.0, 3.0],
    "wet_bulb": [0.5, 1.0, 2.0, 3.0],
    "wbgt_shade": [0.5, 1.0, 2.0, 3.0],
    "wbgt_sun": [0.5, 1.0, 2.0, 3.0],
    "cdd": [0.5, 1.0, 2.0, 3.0],                  # degC-days/day (monthly-mean daily rate)
    "heatstroke_hazard": [0.05, 0.10, 0.15, 0.20],
}
 
# Index of the "middle" tolerance used for the SAW accuracy term per variable.
SAW_TOL_INDEX = 1  # second entry (e.g. 1.0 degC, 2 pp, 20 W/m^2, 1.0 UV)
 
# ----------------------------------------------------------------------------
# Category band definitions for the recalculated indices (for choropleths and
# operational interpretation). (low_inclusive, high_exclusive, label).
# ----------------------------------------------------------------------------
HEAT_INDEX_BANDS = [
    (-1e9, 27, "Comfortable"),
    (27, 32, "Caution"),
    (32, 41, "Extreme Caution"),
    (41, 54, "Danger"),
    (54, 1e9, "Extreme Danger"),
]
 
WBGT_BANDS = [
    (-1e9, 27.8, "Low"),
    (27.8, 29.4, "Moderate"),
    (29.4, 31.0, "High"),
    (31.0, 1e9, "Extreme"),
]
 
# WHO UV Index exposure categories.
UV_BANDS = [
    (-1e9, 3, "Low"),
    (3, 6, "Moderate"),
    (6, 8, "High"),
    (8, 11, "Very High"),
    (11, 1e9, "Extreme"),
]
 
# Dew-point human-comfort categories (degC).
DEW_POINT_COMFORT_BANDS = [
    (-1e9, 10, "Dry"),
    (10, 16, "Comfortable"),
    (16, 18, "Slightly Humid"),
    (18, 21, "Humid"),
    (21, 24, "Very Humid / Oppressive"),
    (24, 1e9, "Extremely Oppressive"),
]
 
# Heatstroke hazard-score interpretive bands (0-1, comparative only).
HEATSTROKE_BANDS = [
    (-1e9, 0.2, "Low"),
    (0.2, 0.4, "Moderate"),
    (0.4, 0.6, "Elevated"),
    (0.6, 0.8, "High"),
    (0.8, 1e9, "Severe"),
]
 