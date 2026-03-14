"""
modules/temporal_view.py
────────────────────────
All fill colours use rgba() strings — Plotly does NOT accept 8-digit hex (#rrggbbaa).
"""

import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.data_loader import (
    extract_time_series,
    VARIABLE_META,
    CITY_LOCATIONS,
)


# ── Colour helper ─────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' — the only format Plotly accepts for transparency."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Shared styling ────────────────────────────────────────────────────────

_PAPER_BG  = "rgba(15,22,38,0.0)"
_PLOT_BG   = "rgba(15,22,38,0.0)"
_GRID_CLR  = "rgba(50,70,100,0.35)"
_FONT      = dict(family="'Space Grotesk','Segoe UI',sans-serif", color="#c8d6e8", size=12)
_AXIS_STYLE = dict(
    color="#4a5a75",
    gridcolor=_GRID_CLR,
    linecolor="rgba(50,70,100,0.5)",
    zerolinecolor=_GRID_CLR,
)
_PALETTE = [
    "#2b7fff", "#ff3b3b", "#00d68f", "#f59e0b",
    "#a855f7", "#ec4899", "#06b6d4", "#84cc16",
    "#f97316", "#6366f1",
]


def _base_layout(title: str, yaxis_title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color="#e0eaf5"), x=0.01),
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=_FONT,
        margin=dict(l=60, r=20, t=50, b=50),
        xaxis=dict(title="Year", **_AXIS_STYLE),
        yaxis=dict(title=yaxis_title, **_AXIS_STYLE),
        hovermode="x unified",
    )


# ── 1. Simple time-series ─────────────────────────────────────────────────

def build_time_series(
    df: pd.DataFrame,
    variable: str,
    city_name: str,
    accent_color: str = "#2b7fff",
) -> go.Figure:
    meta  = VARIABLE_META[variable]
    x_num = np.arange(len(df))
    z     = np.polyfit(x_num, df["value"], 1)
    trend = np.polyval(z, x_num)

    fig = go.Figure()

    # Shaded fill — rgba() only
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["value"],
        mode="none",
        fill="tozeroy",
        fillcolor=_hex_to_rgba(accent_color, 0.08),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Main line
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["value"],
        mode="lines+markers",
        line=dict(color=accent_color, width=2.5, shape="spline"),
        marker=dict(size=6, color=accent_color, line=dict(color="#0a0f1a", width=1.5)),
        name=meta["label"],
        hovertemplate=(
            f"<b>{city_name}</b><br>Year: %{{x}}<br>"
            f"{meta['label']}: %{{y:.2f}} {meta['unit']}<extra></extra>"
        ),
    ))

    # Trend line
    slope_dir = "▲" if z[0] > 0 else "▼"
    slope_txt = f"{slope_dir} {abs(z[0]):.3f} {meta['unit']}/yr"
    fig.add_trace(go.Scatter(
        x=df["year"], y=trend,
        mode="lines",
        line=dict(color="#f59e0b", width=1.5, dash="dash"),
        name="Linear trend",
        hovertemplate=f"Trend: %{{y:.2f}} {meta['unit']}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(
            f"{meta['icon']} {meta['label']} at {city_name}  |  trend {slope_txt}",
            f"{meta['label']} ({meta['unit']})",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11, color="#8a9bb5"), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── 2. Rolling average ────────────────────────────────────────────────────

def build_rolling_average(
    df: pd.DataFrame,
    variable: str,
    city_name: str,
    window: int = 5,
    accent_color: str = "#00d68f",
) -> go.Figure:
    meta    = VARIABLE_META[variable]
    rolling = df["value"].rolling(window, center=True).mean()

    fig = go.Figure()

    # Raw values — use rgba for faded colour
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["value"],
        mode="lines",
        line=dict(color=_hex_to_rgba(accent_color, 0.35), width=1),
        name="Annual",
        hovertemplate=f"Annual: %{{y:.2f}} {meta['unit']}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df["year"], y=rolling,
        mode="lines+markers",
        line=dict(color=accent_color, width=3, shape="spline"),
        marker=dict(size=5, color=accent_color),
        name=f"{window}-yr average",
        hovertemplate=f"{window}-yr avg: %{{y:.2f}} {meta['unit']}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(
            f"{meta['icon']} {window}-Year Rolling Average — {city_name}",
            f"{meta['label']} ({meta['unit']})",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11, color="#8a9bb5"), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── 3. Year-over-year delta ───────────────────────────────────────────────

def build_yoy_delta(
    df: pd.DataFrame,
    variable: str,
    city_name: str,
) -> go.Figure:
    meta   = VARIABLE_META[variable]
    deltas = df["value"].diff().dropna()
    years  = df["year"].iloc[1:].values
    colors = ["#ff3b3b" if d >= 0 else "#2b7fff" for d in deltas]

    fig = go.Figure(go.Bar(
        x=years, y=deltas.values,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate=f"Year: %{{x}}<br>Δ: %{{y:+.2f}} {meta['unit']}<extra></extra>",
        name="YoY Δ",
    ))
    fig.add_hline(y=0, line_color="rgba(180,200,220,0.3)", line_width=1)
    fig.update_layout(
        **_base_layout(
            f"📊 Year-over-Year Change — {meta['label']} at {city_name}",
            f"Δ {meta['label']} ({meta['unit']})",
        ),
        bargap=0.25,
    )
    return fig


# ── 4. Multi-city comparison ──────────────────────────────────────────────

def build_multi_city_series(
    ds: xr.Dataset,
    variable: str,
    city_names: list,
) -> go.Figure:
    meta = VARIABLE_META[variable]
    fig  = go.Figure()

    for i, city in enumerate(city_names):
        if city not in CITY_LOCATIONS:
            continue
        lat, lon = CITY_LOCATIONS[city]
        df       = extract_time_series(ds, variable, lat, lon)
        color    = _PALETTE[i % len(_PALETTE)]

        # Faint fill — rgba() only
        fig.add_trace(go.Scatter(
            x=df["year"], y=df["value"],
            mode="none",
            fill="tozeroy",
            fillcolor=_hex_to_rgba(color, 0.05),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=df["year"], y=df["value"],
            mode="lines+markers",
            line=dict(color=color, width=2, shape="spline"),
            marker=dict(size=5, color=color, line=dict(color="#0a0f1a", width=1)),
            name=city,
            hovertemplate=(
                f"<b>{city}</b><br>Year: %{{x}}<br>"
                f"{meta['label']}: %{{y:.2f}} {meta['unit']}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_base_layout(
            f"{meta['icon']} {meta['label']} — City Comparison",
            f"{meta['label']} ({meta['unit']})",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11, color="#8a9bb5"), bgcolor="rgba(0,0,0,0)",
                    itemsizing="constant"),
    )
    return fig


# ── 5. Trend decomposition ────────────────────────────────────────────────

def build_trend_decomposition(
    df: pd.DataFrame,
    variable: str,
    city_name: str,
    accent_color: str = "#a855f7",
) -> go.Figure:
    meta  = VARIABLE_META[variable]
    x_num = np.arange(len(df))
    poly  = np.polyfit(x_num, df["value"], 2)
    trend = np.polyval(poly, x_num)
    resid = df["value"].values - trend

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Original signal", "Long-term trend", "Interannual residuals"],
    )

    fig.add_trace(go.Scatter(
        x=df["year"], y=df["value"],
        mode="lines+markers",
        line=dict(color=accent_color, width=2),
        marker=dict(size=4),
        name="Original",
        hovertemplate=f"Year: %{{x}}<br>Value: %{{y:.2f}} {meta['unit']}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["year"], y=trend,
        mode="lines",
        line=dict(color="#f59e0b", width=2.5),
        name="Trend",
        hovertemplate=f"Trend: %{{y:.2f}} {meta['unit']}<extra></extra>",
    ), row=2, col=1)

    res_colors = ["#ff3b3b" if r > 0 else "#2b7fff" for r in resid]
    fig.add_trace(go.Bar(
        x=df["year"], y=resid,
        marker_color=res_colors,
        marker_line_width=0,
        name="Residuals",
        hovertemplate=f"Year: %{{x}}<br>Residual: %{{y:+.2f}} {meta['unit']}<extra></extra>",
    ), row=3, col=1)

    for row_n in [1, 2, 3]:
        fig.update_xaxes(gridcolor=_GRID_CLR, linecolor="rgba(50,70,100,0.5)",
                         tickfont_color="#4a5a75", row=row_n, col=1)
        fig.update_yaxes(gridcolor=_GRID_CLR, linecolor="rgba(50,70,100,0.5)",
                         tickfont_color="#4a5a75", title_text=meta["unit"],
                         title_font_color="#4a5a75", row=row_n, col=1)

    fig.update_layout(
        title=dict(text=f"📉 Trend Decomposition — {meta['label']} at {city_name}",
                   font=dict(size=15, color="#e0eaf5"), x=0.01),
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=_FONT,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(size=11, color="#8a9bb5"), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        height=520,
        margin=dict(l=60, r=20, t=60, b=40),
        bargap=0.2,
    )
    return fig