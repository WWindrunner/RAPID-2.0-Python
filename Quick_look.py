import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import rasterio
import contextily as ctx
from rasterio.warp import calculate_default_transform, reproject, Resampling


def _reproject_array_to_3857(data, src_transform, src_crs):
    """
    Reproject a numpy array to EPSG:3857
    """
    dst_crs = "EPSG:3857"

    h, w = data.shape

    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, w, h,
        left=src_transform.c,
        bottom=src_transform.f + h * src_transform.e,
        right=src_transform.c + w * src_transform.a,
        top=src_transform.f
    )

    dst = np.zeros((height, width), dtype=data.dtype)

    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )

    return dst, transform


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
    Plot ESA WorldCover-style land cover with legend.
    """
    class_info = [
        (10, "Tree cover", "#6aa84f"),
        (20, "Shrubland", "#a4de6c"),
        (30, "Grassland", "#8fce00"),
        (40, "Cropland", "#d9d24a"),
        (50, "Built-up", "#000000"),
        (60, "Bare / sparse vegetation", "#c69c2d"),
        (70, "Snow and ice", "#e6d5e8"),
        (80, "Permanent water bodies", "#3c78d8"),
        (90, "Herbaceous wetland", "#76d7c4"),
        (95, "Mangroves", "#45818e"),
        (100, "Moss and lichen", "#999999"),
    ]

    cmap = ListedColormap(["#d9d9d9"] + [x[2] for x in class_info])

    remap = np.zeros_like(lcc, dtype=np.int16)
    for i, (val, _, _) in enumerate(class_info, start=1):
        remap[lcc == val] = i

    ax.imshow(
        remap,
        cmap=cmap,
        interpolation="nearest",
        extent=extent,
        origin="upper"
    )
    ax.set_title("Landcover")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=label)
        for _, label, color in class_info
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=8,
        frameon=True,
        title="Landcover"
    )


def _plot_sar_with_result(ax, sar, morph, extent=None):
    """
    Plot VV SAR as grayscale background and morph result as overlay.

    Morph classes:
      0 = background / no overlay
      1 = Permanent
      2 = Positive Occurrence
      3 = High Confidence
      4 = Moderate Confidence
      5 = Low Confidence
    """
    sar = sar.astype(np.float32)

    # SAR log10 transform for better contrast
    valid_sar = np.isfinite(sar) & (sar > 0)
    sar_log = np.full_like(sar, np.nan, dtype=np.float32)
    sar_log[valid_sar] = 10.0 * np.log10(sar[valid_sar])

    if np.any(valid_sar):
        vmin = np.nanpercentile(sar_log[valid_sar], 2)
        vmax = np.nanpercentile(sar_log[valid_sar], 98)
        if np.isclose(vmin, vmax):
            vmin = np.nanmin(sar_log[valid_sar])
            vmax = np.nanmax(sar_log[valid_sar])
    else:
        vmin, vmax = 0, 1

    # make nodata / invalid area transparent instead of filled
    sar_show = np.ma.masked_invalid(sar_log)

    ax.imshow(
        sar_show,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=extent,
        origin="upper",
    )

    # Only overlay result where both:
    # 1) morph is a valid class > 0
    # 2) SAR itself is valid
    valid_overlay = np.isfinite(sar_log) & (morph > 0)
    morph_masked = np.ma.masked_where(~valid_overlay, morph)

    # revised colors
    result_colors = [
        (0, 0, 0, 0.0),              # 0 transparent
        (0.00, 0.20, 0.90, 0.95),    # 1 Permanent
        (0.68, 0.85, 1.00, 0.95),    # 2 Positive Occurrence (light blue)
        (0.52, 0.36, 0.86, 0.95),    # 3 High Confidence (purple)
        (0.95, 0.95, 0.70, 0.95),    # 4 Moderate Confidence
        (0.93, 0.28, 0.12, 0.95),    # 5 Low Confidence
    ]
    cmap = ListedColormap(result_colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)

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
        Patch(facecolor=result_colors[2], edgecolor="black", linewidth=0.5, label="Positive Occurrence"),
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
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    _plot_lcc(axes[0], lcc, extent=extent)
    _plot_sar_with_result(axes[1], sar, morph, extent=extent)

    title = scene_name if scene_name is not None else os.path.basename(os.path.dirname(fileMorphOut))
    fig.suptitle(title, fontsize=14)

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[quicklook] saved: {out_png}")
    return out_png


def quicklook_map(fileMorphOut, ds, args, out_png=None, scene_name=None):
    """
    Quicklook with basemap:
      left  = LCC
      right = Google/OSM map + morph overlay
    """
    import os

    if out_png is None:
        out_png = os.path.join(os.path.dirname(fileMorphOut), "quicklook_map.png")

    with rasterio.open(fileMorphOut) as src:
        morph = src.read(1)
        transform = src.transform
        crs = src.crs

    # ===== preview =====
    lcc = _to_numpy_preview(ds.land_cover)
    morph = _to_numpy_preview(morph)

    min_h = min(lcc.shape[0], morph.shape[0])
    min_w = min(lcc.shape[1], morph.shape[1])

    lcc = lcc[:min_h, :min_w]
    morph = morph[:min_h, :min_w]

    morph_3857, transform_3857 = _reproject_array_to_3857(
        morph,
        transform,
        crs
    )

    # extent
    h, w = morph_3857.shape
    xmin = transform_3857.c
    ymax = transform_3857.f
    xmax = xmin + w * transform_3857.a
    ymin = ymax + h * transform_3857.e
    extent_3857 = [xmin, xmax, ymin, ymax]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # LCC
    extent_lcc = _extent_from_args(args, (min_h, min_w))
    _plot_lcc(axes[0], lcc, extent=extent_lcc)

    ax = axes[1]

    # morph overlay
    valid_overlay = morph_3857 > 0
    morph_masked = np.ma.masked_where(~valid_overlay, morph_3857)

    colors = [
        (0, 0, 0, 0.0),
        (0.00, 0.20, 0.90, 0.95),   # Permanent
        (0.68, 0.85, 1.00, 0.95),   # Positive Occurrence
        (0.52, 0.36, 0.86, 0.95),   # High Confidence
        (0.95, 0.95, 0.70, 0.95),   # Moderate
        (0.93, 0.28, 0.12, 0.95),   # Low
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)

    ax.imshow(
        morph_masked,
        cmap=cmap,
        norm=norm,
        extent=extent_3857,
        origin="upper"
    )

    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=ctx.providers.CartoDB.Positron
        )
    except Exception as e:
        print(f"Basemap failed, fallback to no basemap: {e}")

    ax.set_title("Flood map (basemap)")
    ax.set_axis_off()

    handles = [
        Patch(facecolor=colors[1], edgecolor="black", label="Permanent"),
        Patch(facecolor=colors[2], edgecolor="black", label="Positive Occurrence"),
        Patch(facecolor=colors[3], edgecolor="black", label="High Confidence"),
        Patch(facecolor=colors[4], edgecolor="black", label="Moderate Confidence"),
        Patch(facecolor=colors[5], edgecolor="black", label="Low Confidence"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)

    title = scene_name if scene_name else os.path.basename(os.path.dirname(fileMorphOut))
    fig.suptitle(title, fontsize=14)

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[quicklook_map] saved: {out_png}")
    return out_png
