import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import rasterio


def _to_numpy_preview(arr, max_dim=1800):
    """
    Convert a dask/numpy array to a 2D numpy preview.
    If too large, downsample by simple stride for quick plotting.
    """
    if hasattr(arr, "compute"):
        arr = arr.compute()

    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")

    h, w = arr.shape
    step = max(1, int(np.ceil(max(h, w) / max_dim)))
    if step > 1:
        arr = arr[::step, ::step]

    return arr


def _extent_from_args(args, shape):
    """
    Build imshow extent from args.transform and array shape.
    """
    if not hasattr(args, "transform") or args.transform is None:
        return None

    transform = args.transform
    h, w = shape

    xmin = transform.c
    ymax = transform.f
    xmax = xmin + w * transform.a
    ymin = ymax + h * transform.e

    return [xmin, xmax, ymin, ymax]


def _plot_lcc(ax, lcc, extent=None):
    """
    Plot ESA WorldCover-style land cover.
    """
    # Common ESA WorldCover classes
    lcc_vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    lcc_colors = [
        "#6aa84f",  # Tree cover
        "#a4de6c",  # Shrubland
        "#8fce00",  # Grassland
        "#d9d24a",  # Cropland
        "#000000",  # Built-up
        "#c69c2d",  # Bare / sparse vegetation
        "#e6d5e8",  # Snow and ice
        "#3c78d8",  # Permanent water bodies
        "#76d7c4",  # Herbaceous wetland
        "#45818e",  # Mangroves
        "#999999",  # Moss and lichen
    ]

    cmap = ListedColormap(["#d9d9d9"] + lcc_colors)

    remap = np.zeros_like(lcc, dtype=np.int16)
    for i, v in enumerate(lcc_vals, start=1):
        remap[lcc == v] = i

    ax.imshow(remap, cmap=cmap, interpolation="nearest", extent=extent, origin="upper")
    ax.set_title("Landcover")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def _plot_sar_with_result(ax, sar, morph, extent=None):
    """
    Plot VV SAR as grayscale background and morph result as overlay.

    Morph classes:
      0 = background
      1 = Permanent
      2 = Positive/Occurrence
      3 = High Confidence
      4 = Moderate Confidence
      5 = Low Confidence
    """
    sar = sar.astype(np.float32)
    valid = np.isfinite(sar) & (sar > 0)

    if np.any(valid):
        vmin = np.nanpercentile(sar[valid], 2)
        vmax = np.nanpercentile(sar[valid], 98)
        if np.isclose(vmin, vmax):
            vmin = np.nanmin(sar[valid])
            vmax = np.nanmax(sar[valid])
    else:
        vmin, vmax = 0, 1

    ax.imshow(
        sar,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=extent,
        origin="upper",
    )

    # Result color scheme from your legend
    # 0 background transparent
    # 1 Permanent = blue
    # 2 Positive/Occurrence = purple
    # 3 High Confidence = light blue
    # 4 Moderate Confidence = light yellow
    # 5 Low Confidence = red-orange
    result_colors = [
        (0, 0, 0, 0.0),        # 0 transparent
        (0.00, 0.20, 0.90, 0.95),  # 1 Permanent
        (0.45, 0.30, 0.80, 0.95),  # 2 Positive/Occurrence
        (0.70, 0.85, 1.00, 0.95),  # 3 High Confidence
        (0.95, 0.95, 0.70, 0.95),  # 4 Moderate Confidence
        (0.95, 0.30, 0.15, 0.95),  # 5 Low Confidence
    ]
    cmap = ListedColormap(result_colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)

    morph_masked = np.ma.masked_where(morph == 0, morph)

    ax.imshow(
        morph_masked,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=extent,
        origin="upper",
    )

    ax.set_title("VV + Morph result")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    handles = [
        Patch(facecolor=result_colors[1], edgecolor="black", linewidth=0.5, label="Permanent"),
        Patch(facecolor=result_colors[2], edgecolor="black", linewidth=0.5, label="Positive/Occurrence"),
        Patch(facecolor=result_colors[3], edgecolor="black", linewidth=0.5, label="High Confidence"),
        Patch(facecolor=result_colors[4], edgecolor="black", linewidth=0.5, label="Moderate Confidence"),
        Patch(facecolor=result_colors[5], edgecolor="black", linewidth=0.5, label="Low Confidence"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)


def quicklook(fileMorphOut, ds, args, out_png=None, scene_name=None):
    """
    Create a quicklook PNG with:
      left  = land cover
      right = VV SAR background + morph overlay

    Parameters
    ----------
    fileMorphOut : str
        Path to final morph GeoTIFF.
    ds : object
        Dataset object from Load_Data, must contain:
          - ds.land_cover
          - ds.img_lp
    args : object
        Args object from Load_Data, should contain transform/refmeta.
    out_png : str or None
        Output png path. Default: same folder as fileMorphOut, named quicklook.png
    scene_name : str or None
        Optional title for the figure.
    """
    if out_png is None:
        out_png = os.path.join(os.path.dirname(fileMorphOut), "quicklook.png")

    # read morph from file to ensure final saved output is what we show
    with rasterio.open(fileMorphOut) as src:
        morph = src.read(1)
        transform = src.transform
        h, w = morph.shape

    # preview arrays
    lcc = _to_numpy_preview(ds.land_cover)
    sar = _to_numpy_preview(ds.img_lp)
    morph = _to_numpy_preview(morph)

    # ensure same preview shape
    min_h = min(lcc.shape[0], sar.shape[0], morph.shape[0])
    min_w = min(lcc.shape[1], sar.shape[1], morph.shape[1])

    lcc = lcc[:min_h, :min_w]
    sar = sar[:min_h, :min_w]
    morph = morph[:min_h, :min_w]

    # build extent using args if available, otherwise fallback to morph transform
    if hasattr(args, "transform") and args.transform is not None:
        extent = _extent_from_args(args, (min_h, min_w))
    else:
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + min_w * transform.a
        ymin = ymax + min_h * transform.e
        extent = [xmin, xmax, ymin, ymax]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    _plot_lcc(axes[0], lcc, extent=extent)
    _plot_sar_with_result(axes[1], sar, morph, extent=extent)

    title = scene_name if scene_name is not None else os.path.basename(os.path.dirname(fileMorphOut))
    fig.suptitle(title, fontsize=14)

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[quicklook] saved: {out_png}")
    return out_png
