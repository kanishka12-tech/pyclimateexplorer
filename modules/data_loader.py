"""
modules/data_loader.py
──────────────────────
Handles all data-access concerns:
  • Loading the NetCDF file into an xarray Dataset
  • Auto-generating the sample dataset if it doesn't exist
  • Helper utilities consumed by spatial_view.py and temporal_view.py
"""

import os
import importlib
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st


# ── Constants ─────────────────────────────────────────────────────────────
NC_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_climate_data.nc")

# Human-readable labels and display metadata for each NetCDF variable
VARIABLE_META: dict[str, dict] = {
    "temperature": {
        "label":      "Surface Temperature",
        "unit":       "K",
        "color_scale": "RdBu_r",
        "icon":       "🌡️",
    },
    "precipitation": {
        "label":      "Precipitation",
        "unit":       "mm/day",
        "color_scale": "Blues",
        "icon":       "🌧️",
    },
    "wind_speed": {
        "label":      "Wind Speed",
        "unit":       "m/s",
        "color_scale": "Viridis",
        "icon":       "🌬️",
    },
}

# Preset city locations for the geographic selector
CITY_LOCATIONS: dict[str, tuple[float, float]] = {
    "Delhi, India":            ( 28.6,  77.2),
    "New York, USA":           ( 40.7, -74.0),
    "London, UK":              ( 51.5,  -0.1),
    "Tokyo, Japan":            ( 35.7, 139.7),
    "Sydney, Australia":       (-33.9, 151.2),
    "São Paulo, Brazil":       (-23.5, -46.6),
    "Cairo, Egypt":            ( 30.0,  31.2),
    "Moscow, Russia":          ( 55.8,  37.6),
    "Beijing, China":          ( 39.9, 116.4),
    "Lagos, Nigeria":          (  6.5,   3.4),
    "Buenos Aires, Argentina": (-34.6, -58.4),
    "Singapore":               (  1.4, 103.8),
    "Toronto, Canada":         ( 43.7, -79.4),
    "Dubai, UAE":              ( 25.2,  55.3),
    "Paris, France":           ( 48.9,   2.4),
    "Reykjavik, Iceland":      ( 64.1, -21.8),
    "Mumbai, India":           ( 19.1,  72.9),
    "Istanbul, Turkey":        ( 41.0,  29.0),
}


# ── Data Loading ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading climate dataset…")
def load_dataset() -> xr.Dataset:
    """
    Load the NetCDF dataset. If the file doesn't exist,
    auto-generate it using generate_sample_data.py.
    Returns an xarray.Dataset with dims (time, lat, lon).
    """
    nc_path = os.path.normpath(NC_PATH)

    if not os.path.exists(nc_path):
        st.info("📦 Sample dataset not found — generating it now (takes ~2 s)…")
        # Dynamically import and run the generator
        spec = importlib.util.spec_from_file_location(
            "generate_sample_data",
            os.path.join(os.path.dirname(__file__), "..", "generate_sample_data.py"),
        )
        gen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_module)
        gen_module.generate_sample_nc(nc_path)

    ds = xr.open_dataset(nc_path, engine="netcdf4")
    return ds


# ── Accessor Helpers ──────────────────────────────────────────────────────

def get_years(ds: xr.Dataset) -> list[int]:
    """Return sorted list of integer years available in the dataset."""
    return sorted(int(t.dt.year) for t in ds["time"])


def get_variable_names(ds: xr.Dataset) -> list[str]:
    """Return list of data variable keys present in the dataset."""
    return [v for v in ds.data_vars if v in VARIABLE_META]


def slice_by_year(ds: xr.Dataset, year: int) -> xr.Dataset:
    """Return a dataset subset for a single year."""
    return ds.sel(time=str(year))


def slice_by_year_range(ds: xr.Dataset, y_start: int, y_end: int) -> xr.Dataset:
    """Return a dataset subset covering [y_start, y_end] inclusive."""
    return ds.sel(time=slice(str(y_start), str(y_end)))


def nearest_grid_point(
    ds: xr.Dataset, lat: float, lon: float
) -> tuple[float, float]:
    """
    Snap an arbitrary (lat, lon) to the nearest grid cell
    in the dataset and return the actual grid coordinates.
    """
    lat_idx = int(np.argmin(np.abs(ds["lat"].values - lat)))
    lon_idx = int(np.argmin(np.abs(ds["lon"].values - lon)))
    return float(ds["lat"].values[lat_idx]), float(ds["lon"].values[lon_idx])


def extract_time_series(
    ds: xr.Dataset, variable: str, lat: float, lon: float
) -> pd.DataFrame:
    """
    Extract a time-series DataFrame for `variable` at the grid cell
    nearest to (lat, lon).

    Returns a DataFrame with columns: year, value.
    """
    grid_lat, grid_lon = nearest_grid_point(ds, lat, lon)
    da = ds[variable].sel(lat=grid_lat, lon=grid_lon, method="nearest")
    df = da.to_dataframe().reset_index()
    df["year"] = pd.to_datetime(df["time"]).dt.year
    df = df.rename(columns={variable: "value"})
    return df[["year", "value"]].sort_values("year").reset_index(drop=True)


def spatial_slice(
    ds: xr.Dataset, variable: str, year: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (lats, lons, values) for `variable` at `year`
    as flat 1-D numpy arrays suitable for Plotly scatter_geo.
    """
    snapshot = slice_by_year(ds, year)[variable]
    lats_2d, lons_2d = np.meshgrid(ds["lat"].values, ds["lon"].values, indexing="ij")
    return (
        lats_2d.ravel(),
        lons_2d.ravel(),
        snapshot.values.ravel(),
    )


def compute_stats(ds: xr.Dataset, variable: str, y_start: int, y_end: int) -> dict:
    """
    Compute global statistics for `variable` over the selected time window.
    Returns dict: mean, min, max, std.
    """
    subset = slice_by_year_range(ds, y_start, y_end)[variable]
    vals   = subset.values
    valid  = vals[~np.isnan(vals)]
    return {
        "mean": float(np.mean(valid)),
        "min":  float(np.min(valid)),
        "max":  float(np.max(valid)),
        "std":  float(np.std(valid)),
        "unit": VARIABLE_META[variable]["unit"],
    }