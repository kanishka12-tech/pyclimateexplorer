"""
generate_sample_data.py
Run once to generate the sample NetCDF climate dataset.
Usage: python generate_sample_data.py
"""

import numpy as np
import xarray as xr
import pandas as pd

def generate_sample_nc(output_path: str = "data/sample_climate_data.nc"):
    """
    Generates a realistic synthetic NetCDF dataset covering:
      - Surface Temperature  (K)
      - Precipitation        (mm/day)
      - Wind Speed           (m/s)
    across a global lat/lon grid for years 2000-2023.
    """

    # ── Grid Definition ──────────────────────────────────────────────────
    lats  = np.arange(-87.5, 90.0, 2.5)   # 72 latitude points
    lons  = np.arange(-177.5, 180.0, 2.5) # 144 longitude points
    times = pd.date_range("2000-01-01", "2023-01-01", freq="YS")  # 24 annual steps

    n_time = len(times)
    n_lat  = len(lats)
    n_lon  = len(lons)

    rng = np.random.default_rng(seed=42)

    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    # ── Surface Temperature (K) ──────────────────────────────────────────
    # Base field: warm equator (~300 K), cold poles (~240 K), realistic gradient
    base_temp = (
        285.0
        - 35.0 * np.abs(LAT) / 90.0           # latitude gradient
        + 8.0  * np.cos(np.deg2rad(LON * 2))   # slight zonal wave
    )
    # Ocean warmth boost near dateline / land cooling near continents
    base_temp += 5.0 * np.cos(np.deg2rad(LAT * 1.5))

    temp_data = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        year_trend = t * 0.04                  # +0.04 K / year warming
        noise      = rng.normal(0, 1.2, (n_lat, n_lon))
        temp_data[t] = base_temp + year_trend + noise

    # ── Precipitation (mm/day) ────────────────────────────────────────────
    # High in tropics, low in subtropics (Hadley cell), moderate mid-lats
    base_precip = (
        8.0  * np.exp(-(LAT / 12.0) ** 2)     # ITCZ peak ~equator
        + 3.0 * np.exp(-((np.abs(LAT) - 50) / 15.0) ** 2)  # mid-lat storms
    )
    base_precip = np.clip(base_precip, 0.1, None)

    precip_data = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        noise = rng.exponential(0.5, (n_lat, n_lon))
        precip_data[t] = np.clip(base_precip + noise, 0, None)

    # ── Wind Speed (m/s) ──────────────────────────────────────────────────
    # Trade winds (~10°–30° lat), jet streams (~50°–60° lat), calm tropics
    base_wind = (
        6.0
        + 3.0 * np.abs(np.sin(np.deg2rad(LAT * 2)))   # jets at ~45°
        - 2.0 * np.exp(-(LAT / 8.0) ** 2)             # calms near equator
    )

    wind_data = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        noise = rng.normal(0, 1.5, (n_lat, n_lon))
        wind_data[t] = np.clip(base_wind + noise, 0, None)

    # ── Assemble xarray Dataset ───────────────────────────────────────────
    ds = xr.Dataset(
        {
            "temperature": xr.DataArray(
                temp_data,
                dims=["time", "lat", "lon"],
                attrs={
                    "long_name": "Surface Temperature",
                    "units":     "K",
                    "standard_name": "air_temperature",
                },
            ),
            "precipitation": xr.DataArray(
                precip_data,
                dims=["time", "lat", "lon"],
                attrs={
                    "long_name": "Total Precipitation",
                    "units":     "mm/day",
                    "standard_name": "precipitation_flux",
                },
            ),
            "wind_speed": xr.DataArray(
                wind_data,
                dims=["time", "lat", "lon"],
                attrs={
                    "long_name": "Wind Speed",
                    "units":     "m/s",
                    "standard_name": "wind_speed",
                },
            ),
        },
        coords={
            "lat":  xr.DataArray(lats,  dims=["lat"],  attrs={"units": "degrees_north"}),
            "lon":  xr.DataArray(lons,  dims=["lon"],  attrs={"units": "degrees_east"}),
            "time": xr.DataArray(times, dims=["time"]),
        },
        attrs={
            "title":       "PyClimaExplorer Sample Climate Dataset",
            "source":      "Synthetic data generated for demonstration",
            "institution": "PyClimaExplorer",
            "creation_date": str(pd.Timestamp.now().date()),
        },
    )

    ds.to_netcdf(output_path)
    print(f"✅  Dataset saved → {output_path}")
    print(ds)
    return ds


if __name__ == "__main__":
    generate_sample_nc()