"""
District choropleth maps for the TCC pipeline (pure matplotlib; no geopandas).

Parses the CC BY 3.0 IGO geoBoundaries Bangladesh ADM2 GeoJSON directly and
draws filled district polygons coloured by a per-district value (continuous) or
class (categorical). District names are matched to the pipeline's dataset names
via assets/name_map.csv, so callers pass ordinary dataset district names.
"""
from __future__ import annotations
import json, os, re, difflib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def load_district_geometry():
    """Return {dataset_district_name: geojson_geometry_dict} for all matched districts."""
    with open(os.path.join(ASSET, "bgd_adm2.geojson"), encoding="utf-8") as f:
        gj = json.load(f)
    nm = pd.read_csv(os.path.join(ASSET, "name_map.csv"))
    zones = pd.read_csv(os.path.join(ASSET, "zones.csv"))
    map_to_geo = dict(zip(nm["dataset_name"], nm["geojson_adm2_en"]))

    geo = {_norm(feat["properties"]["shapeName"]): feat["geometry"] for feat in gj["features"]}
    keys = list(geo.keys())
    out = {}
    for d in zones["district"]:
        cand = map_to_geo.get(d, d)
        key = next((k for k in (_norm(cand), _norm(d)) if k in geo), None)
        if key is None:
            close = difflib.get_close_matches(_norm(cand), keys, n=1, cutoff=0.8)
            key = close[0] if close else None
        if key is not None:
            out[d] = geo[key]
    return out


def _rings(geom):
    t, cs = geom["type"], geom["coordinates"]
    if t == "Polygon":
        return [cs[0]]
    if t == "MultiPolygon":
        return [poly[0] for poly in cs]
    return []


def _set_bounds(ax, geoms):
    xs, ys = [], []
    for geom in geoms.values():
        for ring in _rings(geom):
            arr = np.asarray(ring)
            xs.extend([arr[:, 0].min(), arr[:, 0].max()])
            ys.extend([arr[:, 1].min(), arr[:, 1].max()])
    mx = (max(xs) - min(xs)) * 0.02
    my = (max(ys) - min(ys)) * 0.02
    ax.set_xlim(min(xs) - mx, max(xs) + mx)
    ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.set_aspect("equal")
    ax.axis("off")


def choropleth(ax, geoms, values, cmap="YlOrRd", vmin=None, vmax=None,
               diverging=False, title="", cbar_label="", label_districts=None):
    """Continuous choropleth on `ax`. `values` is {district: float}."""
    vals = [v for v in values.values() if v is not None and np.isfinite(v)]
    vmin = min(vals) if vmin is None else vmin
    vmax = max(vals) if vmax is None else vmax
    if diverging:
        m = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m)
    else:
        norm = Normalize(vmin, vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    patches, colors = [], []
    for d, geom in geoms.items():
        v = values.get(d)
        col = sm.to_rgba(v) if (v is not None and np.isfinite(v)) else (0.85, 0.85, 0.85, 1.0)
        for ring in _rings(geom):
            patches.append(MplPolygon(np.asarray(ring), closed=True))
            colors.append(col)
    ax.add_collection(PatchCollection(patches, facecolor=colors, edgecolor="white", linewidth=0.25))
    _set_bounds(ax, geoms)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    if label_districts:
        for d in label_districts:
            if d in geoms:
                arr = np.asarray(_rings(geoms[d])[0])
                ax.annotate(d, (arr[:, 0].mean(), arr[:, 1].mean()), fontsize=6,
                            ha="center", va="center", fontweight="bold")
    return sm


def choropleth_categorical(ax, geoms, cats, color_map, title="", label_districts=None):
    """Categorical choropleth. `cats` is {district: label}; `color_map` is {label: color}."""
    patches, colors = [], []
    for d, geom in geoms.items():
        lab = cats.get(d)
        col = color_map.get(lab, (0.85, 0.85, 0.85, 1.0))
        for ring in _rings(geom):
            patches.append(MplPolygon(np.asarray(ring), closed=True))
            colors.append(col)
    ax.add_collection(PatchCollection(patches, facecolor=colors, edgecolor="white", linewidth=0.25))
    _set_bounds(ax, geoms)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    if label_districts:
        for d in label_districts:
            if d in geoms:
                arr = np.asarray(_rings(geoms[d])[0])
                ax.annotate(d, (arr[:, 0].mean(), arr[:, 1].mean()), fontsize=6,
                            ha="center", va="center", fontweight="bold")
