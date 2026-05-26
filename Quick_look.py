import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, NoNorm
from matplotlib.patches import Patch
import rasterio
import contextily as ctx
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import zoom


def _read_raster_preview_with_transform(src, max_dim=1800):
    """
    Read raster preview and return adjusted transform.
    """
    h, w = src.height, src.width
    scale = min(max_dim / max(h, w), 1.0)

    out_h = max(1, int(h * scale))
    out_w = max(1, int(w * scale))

    data = src.read(
        1,
        out_shape=(out_h, out_w),
        resampling=Resampling.nearest
    )

    transform = src.transform * src.transform.scale(
        src.width / out_w,
        src.height / out_h
    )

    return data, transform


def _reproject_array_to_3857(data, src_transform, src_crs):
    dst_crs = "EPSG:3857"

    h, w = data.shape

    left = src_transform.c
    top = src_transform.f
    right = left + w * src_transform.a
    bottom = top + h * src_transform.e

    transform, width, height = calculate_default_transform(
        src_crs,
        dst_crs,
        w,
        h,
        left=left,
        bottom=bottom,
        right=right,
        top=top
    )

    dst = np.zeros((height, width), dtype=data.dtype)

    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0
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


def _resize_to_shape(arr, target_shape, order=0):
    zoom_y = target_shape[0] / arr.shape[0]
    zoom_x = target_shape[1] / arr.shape[1]
    out = zoom(arr, (zoom_y, zoom_x), order=order)
    return out[:target_shape[0], :target_shape[1]]


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
    Plot ESA WorldCover-style land cover with fixed class-color mapping.
    """
    class_info = [
        (10, "Tree cover", "#006400"),
        (20, "Shrubland", "#ffbb22"),
        (30, "Grassland", "#ffff4c"),
        (40, "Cropland", "#f096ff"),
        (50, "Built-up", "#fa0000"),
        (60, "Bare / sparse vegetation", "#b4b4b4"),
        (70, "Snow and ice", "#f0f0f0"),
        (80, "Permanent water bodies", "#0064c8"),
        (90, "Herbaceous wetland", "#0096a0"),
        (95, "Mangroves", "#00cf75"),
        (100, "Moss and lichen", "#fae6a0"),
    ]

    colors = ["#d9d9d9"] + [x[2] for x in class_info]
    cmap = ListedColormap(colors)

    remap = np.zeros_like(lcc, dtype=np.int16)
    for i, (val, _, _) in enumerate(class_info, start=1):
        remap[lcc == val] = i

    ax.imshow(
        remap,
        cmap=cmap,
        norm=NoNorm(),
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

    ax.set_title("SAR + RAPID v2.0 result")
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
      right = SAR + RAPID result

    Spatial extent is based on fileMorphOut.
    """
    if out_png is None:
        out_png = os.path.join(os.path.dirname(fileMorphOut), "quicklook.png")

    # read morph with preview transform
    with rasterio.open(fileMorphOut) as src:
        morph, morph_transform = _read_raster_preview_with_transform(src, max_dim=1800)
        morph_crs = src.crs
        morph_nodata = src.nodata

    if morph_nodata is not None:
        morph = np.where(morph == morph_nodata, 0, morph)

    morph = np.where(np.isin(morph, [1, 2, 3, 4, 5]), morph, 0).astype(np.uint8)

    # preview arrays
    target_shape = morph.shape

    lcc = _to_numpy_preview(ds.land_cover, max_dim=max(target_shape))
    sar = _to_numpy_preview(ds.img_lp, max_dim=max(target_shape))

    # force LCC / SAR to exactly match morph preview shape
    if lcc.shape != target_shape:
        lcc = _resize_to_shape(lcc, target_shape, order=0).astype(lcc.dtype)

    if sar.shape != target_shape:
        sar = _resize_to_shape(sar, target_shape, order=1).astype(np.float32)

    # use morph output transform as the only spatial reference
    h, w = morph.shape
    xmin = morph_transform.c
    ymax = morph_transform.f
    xmax = xmin + w * morph_transform.a
    ymin = ymax + h * morph_transform.e
    extent = [xmin, xmax, ymin, ymax]

    # optional CRS-based axis labels
    if morph_crs is not None and morph_crs.is_geographic:
        xlabel = "Longitude"
        ylabel = "Latitude"
    else:
        xlabel = "X"
        ylabel = "Y"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    _plot_lcc(axes[0], lcc, extent=extent)
    _plot_sar_with_result(axes[1], sar, morph, extent=extent)

    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    title = scene_name if scene_name is not None else os.path.basename(os.path.dirname(fileMorphOut))
    fig.suptitle(title, fontsize=14)

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[quicklook] crs: {morph_crs}")
    print(f"[quicklook] extent: {extent}")
    print(f"[quicklook] saved: {out_png}")

    return out_png


def quicklook_map(fileMorphOut, ds, args, out_png=None, scene_name=None):
    """
    Quicklook with basemap:
      left  = LCC
      right = basemap + morph overlay
    """

    if out_png is None:
        out_png = os.path.join(os.path.dirname(fileMorphOut), "quicklook_map.png")

    # =========================
    # Read morph with correct preview transform
    # =========================
    with rasterio.open(fileMorphOut) as src:
        morph, morph_transform = _read_raster_preview_with_transform(src, max_dim=1800)
        morph_crs = src.crs
        morph_nodata = src.nodata

    if morph_nodata is not None:
        morph = np.where(morph == morph_nodata, 0, morph)

    morph = np.where(np.isin(morph, [1, 2, 3, 4, 5]), morph, 0).astype(np.uint8)

    if morph_crs is None:
        print("[quicklook_map] morph CRS is None, skip basemap quicklook.")
        return None

    # =========================
    # Reproject morph to EPSG:3857
    # =========================
    morph_3857, transform_3857 = _reproject_array_to_3857(
        morph,
        morph_transform,
        morph_crs
    )

    h, w = morph_3857.shape
    xmin = transform_3857.c
    ymax = transform_3857.f
    xmax = xmin + w * transform_3857.a
    ymin = ymax + h * transform_3857.e
    extent_3857 = [xmin, xmax, ymin, ymax]

    # =========================
    # LCC preview
    # =========================
    target_shape = morph.shape

    lcc = _to_numpy_preview(ds.land_cover, max_dim=max(target_shape))

    if lcc.shape != target_shape:
        lcc = _resize_to_shape(lcc, target_shape, order=0).astype(lcc.dtype)

    # use morph transform as the ONLY spatial reference
    h, w = morph.shape
    xmin_lcc = morph_transform.c
    ymax_lcc = morph_transform.f
    xmax_lcc = xmin_lcc + w * morph_transform.a
    ymin_lcc = ymax_lcc + h * morph_transform.e
    extent_lcc = [xmin_lcc, xmax_lcc, ymin_lcc, ymax_lcc]

    # =========================
    # Plot
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    _plot_lcc(axes[0], lcc, extent=extent_lcc)

    ax = axes[1]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=ctx.providers.Esri.WorldImagery,
            zoom="auto"
        )
    except Exception as e:
        print(f"[quicklook_map] basemap failed: {e}")

    morph_3857 = np.asarray(morph_3857)
    morph_3857 = np.where(np.isin(morph_3857, [1, 2, 3, 4, 5]), morph_3857, 0)

    valid_overlay = np.isin(morph_3857, [1, 2, 3, 4, 5])
    morph_masked = np.ma.masked_where(~valid_overlay, morph_3857)

    colors = [
        (0, 0, 0, 0.0),
        (0.00, 0.20, 0.90, 0.90),   # Permanent
        (0.68, 0.85, 1.00, 0.80),   # Positive Occurrence
        (0.52, 0.36, 0.86, 0.85),   # High Confidence
        (0.95, 0.95, 0.70, 0.80),   # Moderate
        (0.93, 0.28, 0.12, 0.85),   # Low
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)

    ax.imshow(
        morph_masked,
        cmap=cmap,
        norm=norm,
        extent=extent_3857,
        origin="upper",
        interpolation="nearest",
        zorder=10
    )

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
