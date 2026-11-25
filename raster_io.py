"""
Raster loading, alignment, and value extraction at points.
"""
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol
from pyproj import Transformer


def load_multiple_rasters_aligned(folder_path):
    """
    Load all .tif files in `folder_path` and align them to the
    first raster (grid, CRS, resolution). Bilinear resampling.

    Returns
    -------
    stacked : (n_bands, H, W) float32
    transform, crs, bounds : reference grid
    band_names : list[str]
    """
    files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".tif"))
    if not files:
        raise FileNotFoundError(f"No .tif files in {folder_path}")

    ref_path = os.path.join(folder_path, files[0])
    with rasterio.open(ref_path) as ref:
        ref_arr       = ref.read(1)
        ref_transform = ref.transform
        ref_crs       = ref.crs
        ref_bounds    = ref.bounds

    rasters   = [ref_arr.astype(np.float32)]
    band_names = [files[0]]

    for f in files[1:]:
        path = os.path.join(folder_path, f)
        with rasterio.open(path) as src:
            dst = np.empty(ref_arr.shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
            rasters.append(dst)
            band_names.append(f)

    return np.stack(rasters, axis=0), ref_transform, ref_crs, ref_bounds, band_names


def extract_values_at_points(raster_data, transform, raster_crs,
                             points_df, points_crs="EPSG:4326"):
    """
    Sample raster stack at each (lon, lat) point.
    Points outside the raster get NaN rows.
    """
    transformer = Transformer.from_crs(points_crs, raster_crs, always_xy=True)
    n_bands = raster_data.shape[0]
    values = []
    for _, row in points_df.iterrows():
        lon, lat = float(row["lon"]), float(row["lat"])
        x, y = transformer.transform(lon, lat)
        try:
            r, c = rowcol(transform, x, y)
            if 0 <= r < raster_data.shape[1] and 0 <= c < raster_data.shape[2]:
                vals = raster_data[:, r, c].astype(np.float32)
            else:
                vals = np.full(n_bands, np.nan)
        except Exception:
            vals = np.full(n_bands, np.nan)
        values.append(vals)
    return np.array(values)