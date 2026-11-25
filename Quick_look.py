"""Quicklook PNG of a RAPID result, drawn at the end of the chain (after Post_masks/apply_masks.py).

Three panels on the same Gamma0 VV (dB) background:
  1. Gamma0 VV (dB)
  2. VV + water only: 1 permanent water, 2-5 flood
  3. VV + all classes: water as in 2, snow / ice (100, 106, 116), desert undetermined (200),
     cloud (250), each mask in one colour

  python Quick_look.py <map.tif> <Gamma0_VV.tif> <out.png> [--title TITLE] [--max-dim 2400]

<map.tif> is normally <prefix>_Morph_masked.tif; a plain Morph_final.tif (classes 0-5) also works,
panel 3 then shows water only. VV must be on the same grid (RAPID output grid = input grid).
"""
import argparse
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# water classes, same colours as the earlier RAPID quicklook
WATER = [
    (1, "Permanent water", (0.00, 0.20, 0.90)),
    (2, "Positive occurrence", (0.68, 0.85, 1.00)),
    (3, "High confidence", (0.52, 0.36, 0.86)),
    (4, "Moderate confidence", (0.95, 0.95, 0.70)),
    (5, "Low confidence", (0.93, 0.28, 0.12)),
]
# post-process masks, one colour each
MASKS = [
    ((100, 106, 116), "Snow / ice (100, 106, 116)", (0.00, 0.90, 1.00)),
    ((200,), "Desert undetermined (200)", (0.80, 0.55, 0.15)),
    ((250,), "Cloud (250)", (0.60, 0.60, 0.60)),
]


def _read(path, max_dim, resampling):
    with rasterio.open(path) as src:
        scale = min(max_dim / max(src.height, src.width), 1.0)
        shape = (max(1, int(src.height * scale)), max(1, int(src.width * scale)))
        data = src.read(1, out_shape=shape, resampling=resampling)
        b = src.bounds
        return data, src.nodata, src.crs, [b.left, b.right, b.bottom, b.top]


def _vv_db(vv, nodata):
    vv = vv.astype(np.float32)
    ok = np.isfinite(vv) & (vv > 0)
    if nodata is not None:
        ok &= vv != nodata
    db = np.full(vv.shape, np.nan, np.float32)
    db[ok] = 10.0 * np.log10(vv[ok])
    if ok.any():
        vmin, vmax = np.nanpercentile(db[ok], [2, 98])
        if np.isclose(vmin, vmax):
            vmin, vmax = np.nanmin(db[ok]), np.nanmax(db[ok])
    else:
        vmin, vmax = -25.0, 0.0
    return np.ma.masked_invalid(db), vmin, vmax


def _class_index(m, with_masks):
    """Map class codes to 0 (no overlay), 1-5 water, 6-8 masks."""
    idx = np.zeros(m.shape, np.uint8)
    for c, _, _ in WATER:
        idx[m == c] = c
    if with_masks:
        for i, (codes, _, _) in enumerate(MASKS):
            idx[np.isin(m, codes)] = 6 + i
    return idx


def quicklook(map_tif, vv_tif, out_png, title=None, max_dim=2400):
    m, _, crs, extent = _read(map_tif, max_dim, Resampling.nearest)
    vv, vv_nodata, _, _ = _read(vv_tif, max_dim, Resampling.average)
    if vv.shape != m.shape:
        raise ValueError(f"VV preview {vv.shape} and map preview {m.shape} differ: not the same grid")
    db, vmin, vmax = _vv_db(vv, vv_nodata)

    colors = [(0, 0, 0, 0)] + [c + (0.95,) for _, _, c in WATER] + [c + (0.95,) for _, _, c in MASKS]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), cmap.N)
    water_h = [Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=l) for _, l, c in WATER]
    mask_h = [Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=l) for _, l, c in MASKS]

    geo = crs is not None and crs.is_geographic
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True)
    panels = [("Gamma0 VV (dB)", None, None),
              ("VV + water (classes 1-5)", False, water_h),
              ("VV + all classes", True, water_h + mask_h)]
    for ax, (name, with_masks, handles) in zip(axes, panels):
        ax.imshow(db, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest", extent=extent)
        if with_masks is not None:
            idx = _class_index(m, with_masks)
            ax.imshow(np.ma.masked_equal(idx, 0), cmap=cmap, norm=norm, interpolation="nearest",
                      extent=extent)
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
        ax.set_title(name)
        ax.set_xlabel("Longitude" if geo else "X")
        ax.set_ylabel("Latitude" if geo else "Y")
    fig.suptitle(title or os.path.basename(map_tif), fontsize=14)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[quicklook] saved: {out_png}", flush=True)
    return out_png


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("map_tif")
    p.add_argument("vv_tif")
    p.add_argument("out_png")
    p.add_argument("--title")
    p.add_argument("--max-dim", type=int, default=2400)
    a = p.parse_args()
    quicklook(a.map_tif, a.vv_tif, a.out_png, a.title, a.max_dim)
