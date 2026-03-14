"""
modules/spatial_view.py
────────────────────────
Builds all global-map (spatial) visualisations using Plotly.

Public API
──────────
  build_spatial_map(ds, variable, year)  →  plotly Figure
  build_choropleth_map(ds, variable, year) →  plotly Figure  (density heatmap)
  build_anomaly_map(ds, variable, year, baseline_start, baseline_end) → plotly Figure
"""

import numpy as np
import xarray as xr
import plotly.graph_objects as go

from modules.data_loader import (
    spatial_slice,
    VARIABLE_META,
)


# ── Shared styling ────────────────────────────────────────────────────────

_GEO_LAYOUT = dict(
    showframe=False,
    showcoastlines=True,
    coastlinecolor="rgba(180,200,220,0.6)",
    showland=True,
    landcolor="rgba(30,40,55,0.95)",
    showocean=True,
    oceancolor="rgba(10,20,40,0.85)",
    showlakes=True,
    lakecolor="rgba(10,20,40,0.85)",
    showcountries=True,
    countrycolor="rgba(100,120,140,0.4)",
    projection_type="natural earth",
    bgcolor="rgba(0,0,0,0)",
)

_PAPER_BGCOLOR = "rgba(15,22,38,0.0)"
_PLOT_BGCOLOR  = "rgba(15,22,38,0.0)"

_FONT = dict(family="'Space Grotesk', 'Segoe UI', sans-serif", color="#c8d6e8", size=12)


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color="#e0eaf5"), x=0.01),
        paper_bgcolor=_PAPER_BGCOLOR,
        plot_bgcolor=_PLOT_BGCOLOR,
        font=_FONT,
        margin=dict(l=0, r=0, t=40, b=0),
        geo=_GEO_LAYOUT,
    )


# ── Sub-sampling helper ───────────────────────────────────────────────────

def _subsample(lats, lons, vals, max_points: int = 3000):
    """
    Randomly sub-sample to `max_points` to keep Plotly snappy.
    Preserves spatial coverage by using a regular stride first.
    """
    n = len(vals)
    if n <= max_points:
        return lats, lons, vals
    stride = max(1, n // max_points)
    idx    = np.arange(0, n, stride)
    return lats[idx], lons[idx], vals[idx]


# ── 1. Scatter-geo map ────────────────────────────────────────────────────

def build_spatial_map(
    ds: xr.Dataset,
    variable: str,
    year: int,
    max_points: int = 2500,
) -> go.Figure:
    """
    Scatter-geo plot: each grid cell is a coloured dot.
    Hover shows latitude, longitude, and the variable value.
    """
    meta = VARIABLE_META[variable]
    lats, lons, vals = spatial_slice(ds, variable, year)
    lats, lons, vals = _subsample(lats, lons, vals, max_points)

    # Build hover text
    hover = [
        f"<b>{meta['label']}</b><br>"
        f"Lat: {la:.1f}°  Lon: {lo:.1f}°<br>"
        f"Value: <b>{v:.2f} {meta['unit']}</b>"
        for la, lo, v in zip(lats, lons, vals)
    ]

    fig = go.Figure(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="markers",
            marker=dict(
                size=4,
                color=vals,
                colorscale=meta["color_scale"],
                showscale=True,
                colorbar=dict(
                    title=dict(text=meta["unit"], font=dict(color="#8a9bb5", size=11)),
                    tickfont=dict(color="#8a9bb5", size=10),
                    thickness=12,
                    len=0.75,
                    bgcolor="rgba(15,22,38,0.6)",
                    bordercolor="rgba(80,100,130,0.3)",
                ),
                opacity=0.85,
            ),
            text=hover,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        **_base_layout(f"{meta['icon']} {meta['label']} — {year}")
    )
    return fig


# ── 2. Density heatmap map ────────────────────────────────────────────────

def build_density_map(
    ds: xr.Dataset,
    variable: str,
    year: int,
) -> go.Figure:
    """
    Plotly Densitymapbox-style rendering using a regular grid.
    Gives a smooth heatmap appearance over the globe.
    """
    meta   = VARIABLE_META[variable]
    lats, lons, vals = spatial_slice(ds, variable, year)

    # Normalise values 0–1 for the radius weighting
    v_min, v_max = np.nanmin(vals), np.nanmax(vals)
    norm = (vals - v_min) / (v_max - v_min + 1e-9)

    fig = go.Figure(
        go.Densitymapbox(
            lat=lats,
            lon=lons,
            z=vals,
            radius=8,
            colorscale=meta["color_scale"],
            showscale=True,
            colorbar=dict(
                title=dict(text=meta["unit"], font=dict(color="#8a9bb5", size=11)),
                tickfont=dict(color="#8a9bb5", size=10),
                thickness=12,
                bgcolor="rgba(15,22,38,0.6)",
                bordercolor="rgba(80,100,130,0.3)",
            ),
            hovertemplate=(
                f"<b>{meta['label']}</b><br>"
                "Lat: %{lat:.1f}°<br>"
                "Lon: %{lon:.1f}°<br>"
                f"Value: %{{z:.2f}} {meta['unit']}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_zoom=1,
        mapbox_center={"lat": 20, "lon": 0},
        paper_bgcolor=_PAPER_BGCOLOR,
        plot_bgcolor=_PAPER_BGCOLOR,
        font=_FONT,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text=f"{meta['icon']} {meta['label']} Density — {year}",
            font=dict(size=15, color="#e0eaf5"),
            x=0.01,
        ),
    )
    return fig


# ── 3. Anomaly map ────────────────────────────────────────────────────────

def build_anomaly_map(
    ds: xr.Dataset,
    variable: str,
    year: int,
    baseline_start: int,
    baseline_end: int,
    max_points: int = 2500,
) -> go.Figure:
    """
    Shows departure from a reference-period baseline (default 2000–2010).
    Blue = cooler/drier/calmer than baseline; Red = warmer/wetter/windier.
    """
    meta = VARIABLE_META[variable]

    # Current year snapshot
    lats_c, lons_c, vals_c = spatial_slice(ds, variable, year)

    # Baseline: mean over all years in [baseline_start, baseline_end]
    baseline_years = [
        y for y in range(baseline_start, baseline_end + 1)
        if str(y) in [str(t.dt.year.values) for t in ds["time"]]
    ]
    if not baseline_years:
        baseline_years = [int(t.dt.year) for t in ds["time"]]

    stacked = np.stack(
        [spatial_slice(ds, variable, y)[2] for y in baseline_years], axis=0
    )
    baseline_vals = np.nanmean(stacked, axis=0)

    anomaly = vals_c - baseline_vals
    lats_c, lons_c, anomaly = _subsample(lats_c, lons_c, anomaly, max_points)

    hover = [
        f"<b>Anomaly: {meta['label']}</b><br>"
        f"Lat: {la:.1f}°  Lon: {lo:.1f}°<br>"
        f"Δ = <b>{a:+.2f} {meta['unit']}</b>"
        for la, lo, a in zip(lats_c, lons_c, anomaly)
    ]

    abs_max = max(abs(np.nanmin(anomaly)), abs(np.nanmax(anomaly)))

    fig = go.Figure(
        go.Scattergeo(
            lat=lats_c,
            lon=lons_c,
            mode="markers",
            marker=dict(
                size=4,
                color=anomaly,
                colorscale="RdBu_r",
                cmin=-abs_max,
                cmax=abs_max,
                showscale=True,
                colorbar=dict(
                    title=dict(text=f"Δ {meta['unit']}", font=dict(color="#8a9bb5", size=11)),
                    tickfont=dict(color="#8a9bb5", size=10),
                    thickness=12,
                    len=0.75,
                    bgcolor="rgba(15,22,38,0.6)",
                    bordercolor="rgba(80,100,130,0.3)",
                ),
                opacity=0.85,
            ),
            text=hover,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        **_base_layout(
            f"{meta['icon']} {meta['label']} Anomaly vs "
            f"{baseline_start}–{baseline_end} baseline ({year})"
        )
    )
    return fig


# ── 4. Multi-variable snapshot (small multiples) ──────────────────────────

def build_multi_variable_maps(
    ds: xr.Dataset,
    year: int,
    variables: list[str],
) -> go.Figure:
    """
    Side-by-side scatter-geo maps (subplot grid) for multiple variables
    at the same year – useful for the 'Overview' tab.
    """
    from plotly.subplots import make_subplots

    n     = len(variables)
    cols  = min(n, 3)
    rows  = (n + cols - 1) // cols

    subplot_specs = [[{"type": "geo"}] * cols for _ in range(rows)]
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=subplot_specs,
        subplot_titles=[VARIABLE_META[v]["label"] for v in variables],
    )

    for idx, variable in enumerate(variables):
        row = idx // cols + 1
        col = idx % cols + 1
        meta = VARIABLE_META[variable]
        lats, lons, vals = spatial_slice(ds, variable, year)
        lats, lons, vals = _subsample(lats, lons, vals, 800)

        fig.add_trace(
            go.Scattergeo(
                lat=lats,
                lon=lons,
                mode="markers",
                marker=dict(
                    size=3,
                    color=vals,
                    colorscale=meta["color_scale"],
                    showscale=False,
                    opacity=0.8,
                ),
                name=meta["label"],
                hovertemplate=(
                    f"<b>{meta['label']}</b><br>"
                    "Lat: %{lat:.1f}°  Lon: %{lon:.1f}°<br>"
                    f"Value: %{{marker.color:.2f}} {meta['unit']}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )

        geo_key = "geo" if idx == 0 else f"geo{idx + 1}"
        fig.update_layout(**{geo_key: _GEO_LAYOUT})

    fig.update_layout(
        title=dict(
            text=f"🌍 Global Climate Snapshot — {year}",
            font=dict(size=16, color="#e0eaf5"),
            x=0.01,
        ),
        paper_bgcolor=_PAPER_BGCOLOR,
        font=_FONT,
        height=360 * rows,
        showlegend=False,
        margin=dict(l=0, r=0, t=60, b=10),
    )
    return fig