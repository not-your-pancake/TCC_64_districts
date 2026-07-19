"""
Build a lightweight district GeoJSON for the interactive dashboard.

The source ADM2 file is ~46 MB — far too heavy to ship to the browser for
Plotly choropleths. This script matches districts to dataset names via the
pipeline's loader, keeps exterior rings only, simplifies them with
Douglas-Peucker, rounds coordinates, and writes dashboard/assets/districts_light.geojson.

Run once (and re-run only if the source geometry changes):
    py -3.14 dashboard/build_geo.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tcc_pipeline import choropleth as ch

TOL = 0.0025          # degrees (~280 m) — invisible at 64-district scale
PRECISION = 4         # coordinate decimals (~11 m)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "assets", "districts_light.geojson")


def rdp(points: np.ndarray, tol: float) -> np.ndarray:
    """Iterative Douglas-Peucker on an (n, 2) array."""
    n = len(points)
    if n < 3:
        return points
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = points[j] - points[i]
        seg_len = np.hypot(*seg)
        pts = points[i + 1:j]
        if seg_len == 0:
            d = np.hypot(*(pts - points[i]).T)
        else:
            rel = points[i] - pts
            d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / seg_len
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack.append((i, i + 1 + k))
            stack.append((i + 1 + k, j))
    return points[keep]


def ensure_cw(points: np.ndarray) -> np.ndarray:
    """Force clockwise winding (open ring, lon/lat).

    Plotly's geo engine (d3-geo) treats rings as spherical polygons: a ring
    wound the "wrong" way is interpreted as the whole globe minus the shape.
    The source ADM2 file has mixed winding, so normalize every exterior ring.
    """
    x, y = points[:, 0], points[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    return points[::-1] if area2 > 0 else points  # area2 > 0 == counter-clockwise


def simplify_ring(ring, tol):
    arr = np.asarray(ring, dtype=float)
    closed = np.array_equal(arr[0], arr[-1])
    out = rdp(arr if not closed else arr[:-1], tol)
    if len(out) < 3:
        return None
    out = ensure_cw(np.round(out, PRECISION))
    return np.vstack([out, out[:1]]).tolist()


def simplify_geom(geom):
    t, cs = geom["type"], geom["coordinates"]
    polys = [cs] if t == "Polygon" else cs
    keep = []
    for poly in polys:                       # exterior ring only per polygon
        ring = simplify_ring(poly[0], TOL)
        if ring:
            keep.append([ring])
    if not keep:
        return None
    if len(keep) == 1:
        return {"type": "Polygon", "coordinates": keep[0]}
    return {"type": "MultiPolygon", "coordinates": keep}


def main():
    geoms = ch.load_district_geometry()
    feats = []
    for d, g in geoms.items():
        sg = simplify_geom(g)
        if sg is None:
            print(f"  !! dropped {d}")
            continue
        feats.append({"type": "Feature", "id": d,
                      "properties": {"district": d}, "geometry": sg})
    fc = {"type": "FeatureCollection", "features": feats}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fc, f, separators=(",", ":"))
    print(f"{len(feats)} districts -> {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
