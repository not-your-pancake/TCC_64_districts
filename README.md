# TCC Bangladesh — District-Level Heat-Stress Projection

Trend-aware machine-learning projection and validation of seven
**Temperature-Correlated Conditions (TCC)** heat-stress indicators across all
64 districts of Bangladesh, with a reviewer-facing manuscript targeted at
*Urban Climate* (Elsevier).

> **Private repository.** Contains raw meteorological data obtained from the
> Visual Crossing Weather API under a free/educational account — **do not
> redistribute the datasets publicly** (see *Data* below).

## What this does

1. Trains four gradient-boosting ensembles (RandomForest, XGBoost, LightGBM,
   CatBoost) per district × base variable on the 2014–2024 monthly record using
   future-known Fourier + calendar-year features.
2. Detrends each variable by its robust 1980–2024 Theil–Sen slope, models the
   seasonal residual, and re-adds the trend at projection time so projections
   evolve realistically (2025–2026).
3. Recomputes seven thermal-stress conditions from the *projected* base
   variables (no target leakage): heat index, wet-bulb, WBGT (shade/sun),
   heatstroke risk *(tested and excluded for poor skill)*, cooling degree-days,
   UV exposure, dew-point moisture stress.
4. Validates the projected 2025 fields against withheld 2025 observations,
   benchmarks skill vs seasonal-naive and climatology, and runs a
   Mann–Kendall/Sen trend analysis on 1980–2024.
5. Produces district choropleths, zonal figures, risk-category maps, a reviewer
   workbook, and the LaTeX manuscript.

## Layout

```
tcc_pipeline/        core package (config, engine, indices, trend, figures, choropleth)
  assets/            zones.csv, name_map.csv, bgd_adm2.geojson (geoBoundaries, CC BY 3.0 IGO)
1980-2024-dataset/   daily historical weather per district  (Visual Crossing — do not redistribute)
2025-dataset/        daily 2025 observations for validation  (Visual Crossing — do not redistribute)
run_pipeline.py      main entry point (train + project + validate + figures + workbook)
build_assets.py      regenerate zones.csv / name_map.csv
build_categories.py  risk-category post-processing from the projected forecasts
build_maps.py        district choropleths + multi-condition heatmap
build_tables.py      LaTeX table bodies for the manuscript (outputs -> manuscript/tables/)
dashboard/           Streamlit interactive dashboard (app.py)
manuscript/          elsarticle LaTeX source, references.bib, generated tables/
outputs/             forecasts, metrics, trends, summary, figures, workbook (regenerable)
```

## Reproduce

Use the real CPython (not the MSYS2 shell Python): invoke everything with `py -3.14`.

```bash
py -3.14 run_pipeline.py          # ~55 min: full 64-district train + project
py -3.14 build_categories.py      # risk categories
py -3.14 build_maps.py            # choropleths + heatmap
py -3.14 build_tables.py          # manuscript table bodies
```

Regenerate the whole figure/table set after any pipeline re-run by running the
four `build_*` scripts in that order.

### Manuscript

```bash
cd manuscript
tectonic -X compile manuscript.tex
```

### Dashboard

```bash
py -3.14 -m streamlit run dashboard/app.py
```

## Data

Meteorological data: **Visual Crossing Weather API** (https://www.visualcrossing.com),
collected under a free/educational account. Their terms restrict public
redistribution — keep this repository private and do not publish the CSVs.

District boundaries: **geoBoundaries** gbOpen Bangladesh ADM2
(CC BY 3.0 IGO; source HDX/BBS 2015).
