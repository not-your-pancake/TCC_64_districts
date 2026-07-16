"""
Vectorized physical equations and thermal indices for the TCC pipeline.
Includes analytical error propagation for uncertainty bands.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Heat Index (NWS Rothfusz Regression)
# ---------------------------------------------------------------------------
def compute_heat_index(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Vectorized NWS Heat Index calculation with Rothfusz adjustments."""
    t_f = t_c * 9.0 / 5.0 + 32.0
    
    # Simple formula fallback for mild conditions
    hi_simple = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))
    
    # Full Rothfusz regression
    hi_full = (
        -42.379 + 2.04901523 * t_f + 10.14333127 * rh
        - 0.22475541 * t_f * rh - 6.83783e-3 * t_f**2
        - 5.481717e-2 * rh**2 + 1.22874e-3 * t_f**2 * rh
        + 8.5282e-4 * t_f * rh**2 - 1.99e-6 * t_f**2 * rh**2
    )
    
    # Adjustments
    adj1 = ((13.0 - rh) / 4.0) * np.sqrt((17.0 - np.abs(t_f - 95.0)) / 17.0)
    adj2 = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
    
    hi_full = np.where((rh < 13.0) & (t_f >= 80.0) & (t_f <= 112.0), hi_full - adj1, hi_full)
    hi_full = np.where((rh > 85.0) & (t_f >= 80.0) & (t_f <= 87.0), hi_full + adj2, hi_full)
    
    # Choose simple or full based on standard NWS threshold
    hi_f = np.where(hi_simple < 80.0, hi_simple, hi_full)
    
    # Return converted back to Celsius
    return (hi_f - 32.0) * 5.0 / 9.0

# ---------------------------------------------------------------------------
# 2. Wet-Bulb Temperature (Stull 2011 Formula)
# ---------------------------------------------------------------------------
def compute_wet_bulb(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Vectorized calculation of Wet-Bulb temperature using Stull (2011)."""
    tw = (
        t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * (rh**(1.5)) * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return tw

# ---------------------------------------------------------------------------
# 3. WBGT (Wet-Bulb Globe Temperature) Variants
# ---------------------------------------------------------------------------
def compute_wbgt(t_c: np.ndarray, tw_c: np.ndarray, sr: np.ndarray, sun: bool = True) -> np.ndarray:
    """Computes WBGT variants (Sun with radiation factor; Shade without)."""
    if sun:
        # IAENG/predicted reference formulation: 0.7*Tw + 0.2*(T + 0.025*SR) + 0.1*T
        return 0.7 * tw_c + 0.2 * (t_c + 0.025 * sr) + 0.1 * t_c
    else:
        # Shade variant standard formulation
        return 0.7 * tw_c + 0.3 * t_c

# ---------------------------------------------------------------------------
# 4. Composite Heatstroke Hazard Score
# ---------------------------------------------------------------------------
def compute_heatstroke_hazard(hi_c: np.ndarray, tw_c: np.ndarray) -> np.ndarray:
    """
    Applies a recalibrated sigmoid centered at a monthly-mean Heat Index
    Danger threshold (~41C / 105.8F) with scaled Wet-Bulb multipliers.
    """
    hi_f = hi_c * 9.0 / 5.0 + 32.0
    
    # Multiplier tiers scaled to the monthly-mean wet-bulb distribution.
    mult = np.ones_like(tw_c)
    mult = np.where(tw_c < 22.0, 0.2, mult)
    mult = np.where((tw_c >= 22.0) & (tw_c < 24.0), 0.5, mult)
    mult = np.where((tw_c >= 24.0) & (tw_c < 26.0), 0.8, mult)
    mult = np.where(tw_c >= 26.0, 1.0, mult)
    
    # Inflection at the sustained monthly-mean "Danger" threshold rather than
    # the daily-peak one; a daily-peak centre leaves monthly means in the tail.
    z = 0.35 * (hi_f - 105.8)
    score = (1.0 / (1.0 + np.exp(-z))) * mult
    return np.clip(score, 0.0, 1.0)

# ---------------------------------------------------------------------------
# 5. Cooling Degree Days (CDD)
# ---------------------------------------------------------------------------
def compute_cdd(tmax: np.ndarray, tmin: np.ndarray, base: float = 18.0) -> np.ndarray:
    """Calculates Cooling Degree Days based on a mean daily baseline of 18°C."""
    tmean = (tmax + tmin) / 2.0
    return np.maximum(tmean - base, 0.0)

# ---------------------------------------------------------------------------
# Core Pipeline Interface & Error Propagation
# ---------------------------------------------------------------------------
def recompute_indices_from_base(df: pd.DataFrame) -> pd.DataFrame:
    """Accepts a wide DataFrame containing predicted base variables and 
    recalculates all physical secondary indices to maintain physical coupling."""
    out = df.copy()
    
    t = out["temperature"].values
    rh = out["humidity"].values
    sr = out["solar_radiation"].values
    tmax = out["tmax"].values
    tmin = out["tmin"].values
    
    out["heat_index"] = compute_heat_index(t, rh)
    out["wet_bulb"] = compute_wet_bulb(t, rh)
    out["wbgt_shade"] = compute_wbgt(t, out["wet_bulb"].values, sr, sun=False)
    out["wbgt_sun"] = compute_wbgt(t, out["wet_bulb"].values, sr, sun=True)
    out["heatstroke_hazard"] = compute_heatstroke_hazard(out["heat_index"].values, out["wet_bulb"].values)
    out["cdd"] = compute_cdd(tmax, tmin)
    
    return out

def sigma_heat_index(s_t: float, s_rh: float) -> float:
    """Analytical error propagation proxy for Heat Index uncertainty bands."""
    if np.isnan(s_t) or np.isnan(s_rh):
        return 1.5  # default conservative backup sigma
    return float(np.sqrt((1.1 * s_t)**2 + (0.05 * s_rh)**2))

def sigma_wet_bulb(s_t: float, s_rh: float) -> float:
    """Analytical error propagation proxy for Wet Bulb uncertainty bands."""
    if np.isnan(s_t) or np.isnan(s_rh):
        return 1.2
    return float(np.sqrt((0.6 * s_t)**2 + (0.12 * s_rh)**2))

def sigma_wbgt(s_t: float, s_rh: float, s_sr: float) -> float:
    """Analytical error propagation proxy for WBGT uncertainty bands."""
    if np.isnan(s_t) or np.isnan(s_rh) or np.isnan(s_sr):
        return 1.3
    s_tw = sigma_wet_bulb(s_t, s_rh)
    return float(np.sqrt((0.7 * s_tw)**2 + (0.3 * s_t)**2 + (0.005 * s_sr)**2))

# ---------------------------------------------------------------------------
# Short-name API used by trend.py, which works on the 1980-2024 record where
# solar radiation is absent -- hence shade-only WBGT with no radiation term.
# ---------------------------------------------------------------------------
def heat_index(t_c, rh):
    return compute_heat_index(t_c, rh)

def wet_bulb(t_c, rh):
    return compute_wet_bulb(t_c, rh)

def wbgt_shade(t_c, tw_c):
    return compute_wbgt(t_c, tw_c, sr=0.0, sun=False)