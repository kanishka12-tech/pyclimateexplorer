"""
app.py  –  Climate-Explorer
Run with:  python -m streamlit run app.py
"""
1
import os
import sys
import importlib.util

import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.data_loader import (
    load_dataset, get_years, get_variable_names,
    extract_time_series, compute_stats,
    VARIABLE_META, CITY_LOCATIONS,
)
from modules.spatial_view import (
    build_spatial_map, build_anomaly_map, build_multi_variable_maps,
)
from modules.temporal_view import (
    build_time_series, build_rolling_average, build_yoy_delta,
    build_multi_city_series, build_trend_decomposition,
)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClimateExplorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Orbitron:wght@400;700&display=swap');
  html, body, [class*="css"] { font-family: 'Space Grotesk','Segoe UI',sans-serif; }
  .stApp { background: linear-gradient(160deg,#080e1c 0%,#0d1526 40%,#0a1020 100%); }
  section[data-testid="stSidebar"] {
    background: rgba(10,16,32,0.97);
    border-right: 1px solid rgba(43,127,255,0.18);
  }
  [data-testid="metric-container"] {
    background: rgba(20,30,55,0.85);
    border: 1px solid rgba(43,127,255,0.18);
    border-radius: 12px;
    padding: 14px 18px;
  }
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(10,16,32,0.6); border-radius:10px; gap:4px;
  }
  .stTabs [data-baseweb="tab"] {
    color:#6a80a0; border-radius:8px; padding:8px 18px;
    font-size:13px; font-weight:500;
  }
  .stTabs [aria-selected="true"] {
    background:rgba(43,127,255,0.15)!important;
    color:#a8c8ff!important;
    border-bottom:2px solid #2b7fff!important;
  }
  .banner-title {
    font-family:'Orbitron',monospace;
    font-size:clamp(22px,4vw,42px); font-weight:700;
    background:linear-gradient(90deg,#2b7fff,#00d68f 60%,#f59e0b);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:8px;
  }
  .banner-sub { font-size:15px; color:#6a88aa; line-height:1.6; max-width:680px; }
  .sect-divider {
    height:1px;
    background:linear-gradient(90deg,rgba(43,127,255,0.3),transparent);
    margin:16px 0;
  }
  .stPlotlyChart {
    border-radius:12px; overflow:hidden;
    border:1px solid rgba(43,127,255,0.12);
    background:rgba(12,18,35,0.6);
  }
  label[data-testid="stWidgetLabel"] {
    color:#5a7090!important; font-size:11px!important;
    font-weight:600!important; text-transform:uppercase; letter-spacing:1.1px;
  }
  #MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────
ds    = load_dataset()
years = get_years(ds)
vars_ = get_variable_names(ds)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Orbitron,monospace;font-size:16px;"
        "background:linear-gradient(90deg,#2b7fff,#00d68f);-webkit-background-clip:text;"
        "-webkit-text-fill-color:transparent;font-weight:700;padding:6px 0'>🌍 ClimaExplorer</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sect-divider'></div>", unsafe_allow_html=True)

    st.markdown("**Climate Variable**")
    var_labels = {k: f"{VARIABLE_META[k]['icon']}  {VARIABLE_META[k]['label']}" for k in vars_}
    variable = st.selectbox(
        "Select variable", options=vars_,
        format_func=lambda k: var_labels[k],
        label_visibility="collapsed",
    )
    meta = VARIABLE_META[variable]

    st.markdown("<div class='sect-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Time Range**")
    y_min, y_max = min(years), max(years)
    year_range = st.slider(
        "Year range", min_value=y_min, max_value=y_max,
        value=(y_min, y_max), step=1, label_visibility="collapsed",
    )
    y_start, y_end = year_range

    snapshot_year = st.slider(
        "Snapshot year (spatial map)",
        min_value=y_start, max_value=y_end, value=y_end, step=1,
    )

    st.markdown("<div class='sect-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Location**")
    loc_mode = st.radio(
        "Selection mode", ["City preset", "Manual lat/lon"],
        horizontal=True, label_visibility="collapsed",
    )

    if loc_mode == "City preset":
        city_name = st.selectbox(
            "Choose city", options=list(CITY_LOCATIONS.keys()),
            label_visibility="collapsed",
        )
        sel_lat, sel_lon = CITY_LOCATIONS[city_name]
        st.caption(f"📍 {sel_lat:.1f}°N, {sel_lon:.1f}°E")
    else:
        city_name = "Custom location"
        col_lat, col_lon = st.columns(2)
        with col_lat:
            sel_lat = st.number_input("Latitude",  min_value=-90.0,  max_value=90.0,  value=28.6, step=0.5)
        with col_lon:
            sel_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.2, step=0.5)

    st.markdown("<div class='sect-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Compare Cities** *(Temporal tab)*")
    compare_cities = st.multiselect(
        "Select cities to compare",
        options=list(CITY_LOCATIONS.keys()),
        default=list(CITY_LOCATIONS.keys())[:4],
        label_visibility="collapsed",
        max_selections=8,
    )

    st.markdown("<div class='sect-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Anomaly Baseline**")
    baseline_range = st.slider(
        "Baseline period", min_value=y_min, max_value=y_max,
        value=(y_min, min(y_min + 10, y_max)), step=1,
        label_visibility="collapsed",
    )
    b_start, b_end = baseline_range

    st.markdown("**Smoothing Window**")
    roll_win = st.slider(
        "Rolling average window (years)", 2, 10, 5,
        label_visibility="collapsed",
    )

# ── Landing banner ────────────────────────────────────────────────────────
banner_path = os.path.join(ROOT, "assets", "earth_banner.png")
if os.path.exists(banner_path):
    from PIL import Image
    st.image(Image.open(banner_path), use_container_width=True)

st.markdown(
    "<div class='banner-title'>ClimaExplorer</div>"
    "<div class='banner-sub'>"
    "An interactive platform for researchers and the general public to explore "
    "global climate trends — surface temperature, precipitation, and wind speed — "
    "through spatial maps, temporal time-series, anomaly analysis, and multi-city comparisons."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='sect-divider'></div>", unsafe_allow_html=True)

# ── Statistics summary ────────────────────────────────────────────────────
stats = compute_stats(ds, variable, y_start, y_end)
unit  = stats["unit"]

st.markdown(f"#### {meta['icon']} {meta['label']}  ·  Global Stats  ({y_start}–{y_end})")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Global Mean",    f"{stats['mean']:.2f} {unit}")
with c2:
    st.metric("Global Maximum", f"{stats['max']:.2f} {unit}")
with c3:
    st.metric("Global Minimum", f"{stats['min']:.2f} {unit}")
with c4:
    st.metric("Std Deviation",  f"{stats['std']:.2f} {unit}")

loc_df         = extract_time_series(ds, variable, sel_lat, sel_lon)
loc_mean       = loc_df["value"].mean()
loc_max        = loc_df["value"].max()
loc_min        = loc_df["value"].min()
loc_trend_slope = float(np.polyfit(np.arange(len(loc_df)), loc_df["value"], 1)[0])

st.markdown(f"**📍 {city_name}  ·  Local Stats  ({y_start}–{y_end})**")
lc1, lc2, lc3, lc4 = st.columns(4)
with lc1:
    st.metric("Local Mean", f"{loc_mean:.2f} {unit}")
with lc2:
    st.metric("Local Max",  f"{loc_max:.2f} {unit}")
with lc3:
    st.metric("Local Min",  f"{loc_min:.2f} {unit}")
with lc4:
    direction = "▲ Rising" if loc_trend_slope > 0 else "▼ Falling"
    st.metric("Trend slope", f"{loc_trend_slope:+.3f} {unit}/yr", delta=direction)

st.markdown("<div class='sect-divider'></div>", unsafe_allow_html=True)

# ── Main tabs ─────────────────────────────────────────────────────────────
tab_overview, tab_spatial, tab_temporal, tab_anomaly, tab_compare = st.tabs([
    "🌍 Overview", "🗺️ Spatial Map", "📈 Temporal Trends", "🔴 Anomaly", "🏙️ City Compare",
])

# ── TAB 1 · OVERVIEW ─────────────────────────────────────────────────────
with tab_overview:
    st.subheader(f"Global Climate Snapshot — {snapshot_year}")
    st.caption("All three climate variables shown simultaneously for the selected snapshot year.")
    with st.spinner("Rendering overview maps…"):
        fig_overview = build_multi_variable_maps(ds, snapshot_year, vars_)
    st.plotly_chart(fig_overview, use_container_width=True, key="overview_multi")

    st.markdown("<div class='sect-divider'></div>", unsafe_allow_html=True)
    st.subheader(f"🏙️ {meta['label']} at {city_name}")
    with st.spinner("Rendering time-series…"):
        loc_df_range = loc_df[(loc_df["year"] >= y_start) & (loc_df["year"] <= y_end)].copy()
        fig_overview_ts = build_time_series(loc_df_range, variable, city_name)
    st.plotly_chart(fig_overview_ts, use_container_width=True, key="overview_ts")

# ── TAB 2 · SPATIAL MAP ──────────────────────────────────────────────────
with tab_spatial:
    st.subheader(f"🗺️ Spatial Distribution — {meta['label']} ({snapshot_year})")
    st.caption(
        "Global scatter-geo map: each dot is a ~2.5° × 2.5° grid cell. "
        "Hover for exact coordinates and values."
    )
    with st.spinner("Rendering spatial map…"):
        fig_spatial = build_spatial_map(ds, variable, snapshot_year)
    st.plotly_chart(fig_spatial, use_container_width=True, key="spatial_map")

    try:
        point_val = ds[variable].sel(
            lat=sel_lat, lon=sel_lon, time=str(snapshot_year), method="nearest"
        ).item()
        st.info(
            f"📍 **{city_name}** ({sel_lat:.1f}°N, {sel_lon:.1f}°E)  — "
            f"value: **{point_val:.2f} {unit}**"
        )
    except Exception:
        pass

# ── TAB 3 · TEMPORAL TRENDS ──────────────────────────────────────────────
with tab_temporal:
    st.subheader(f"📈 Temporal Analysis — {meta['label']} at {city_name}")
    loc_df_t = loc_df[(loc_df["year"] >= y_start) & (loc_df["year"] <= y_end)].copy()

    sub_ts, sub_roll, sub_yoy, sub_decomp = st.tabs([
        "Time Series", f"{roll_win}-Year Average", "Year-over-Year Δ", "Decomposition",
    ])

    with sub_ts:
        with st.spinner("Rendering time-series…"):
            fig_ts = build_time_series(loc_df_t, variable, city_name)
        st.plotly_chart(fig_ts, use_container_width=True, key="temporal_ts")

    with sub_roll:
        with st.spinner("Rendering rolling average…"):
            fig_roll = build_rolling_average(loc_df_t, variable, city_name, window=roll_win)
        st.plotly_chart(fig_roll, use_container_width=True, key="temporal_roll")

    with sub_yoy:
        with st.spinner("Rendering YoY delta…"):
            fig_yoy = build_yoy_delta(loc_df_t, variable, city_name)
        st.plotly_chart(fig_yoy, use_container_width=True, key="temporal_yoy")

    with sub_decomp:
        with st.spinner("Rendering decomposition…"):
            fig_decomp = build_trend_decomposition(loc_df_t, variable, city_name)
        st.plotly_chart(fig_decomp, use_container_width=True, key="temporal_decomp")

# ── TAB 4 · ANOMALY ──────────────────────────────────────────────────────
with tab_anomaly:
    st.subheader(f"🔴 Anomaly Map — {meta['label']}")
    st.caption(
        f"Departure from the **{b_start}–{b_end}** baseline.  "
        "**Red** = above average  |  **Blue** = below average."
    )
    if b_end <= b_start:
        st.warning("⚠️ Baseline end year must be greater than start year.")
    else:
        with st.spinner("Computing anomaly…"):
            fig_anomaly = build_anomaly_map(ds, variable, snapshot_year, b_start, b_end)
        st.plotly_chart(fig_anomaly, use_container_width=True, key="anomaly_map")

        loc_baseline = extract_time_series(ds, variable, sel_lat, sel_lon)
        bl_mean = loc_baseline[
            (loc_baseline["year"] >= b_start) & (loc_baseline["year"] <= b_end)
        ]["value"].mean()
        current_val = loc_baseline[loc_baseline["year"] == snapshot_year]["value"].values
        if len(current_val):
            anom_val  = current_val[0] - bl_mean
            direction = "above" if anom_val > 0 else "below"
            color_ic  = "🔴" if anom_val > 0 else "🔵"
            st.info(
                f"{color_ic} **{city_name}** in {snapshot_year}: "
                f"**{anom_val:+.2f} {unit}** {direction} the "
                f"{b_start}–{b_end} baseline (mean = {bl_mean:.2f} {unit})"
            )

# ── TAB 5 · CITY COMPARE ─────────────────────────────────────────────────
with tab_compare:
    st.subheader(f"🏙️ City Comparison — {meta['label']}")
    st.caption("Overlay time-series for multiple cities. Adjust the Compare Cities selection in the sidebar.")

    if not compare_cities:
        st.warning("Select at least two cities in the sidebar to enable comparison.")
    else:
        with st.spinner("Rendering city comparison…"):
            fig_multi = build_multi_city_series(ds, variable, compare_cities)
        st.plotly_chart(fig_multi, use_container_width=True, key="compare_multi")

        import pandas as pd
        st.markdown("**Summary Statistics**")
        rows = []
        for city in compare_cities:
            if city not in CITY_LOCATIONS:
                continue
            lat, lon = CITY_LOCATIONS[city]
            cdf = extract_time_series(ds, variable, lat, lon)
            cdf = cdf[(cdf["year"] >= y_start) & (cdf["year"] <= y_end)]
            slope = float(np.polyfit(np.arange(len(cdf)), cdf["value"], 1)[0])
            rows.append({
                "City":      city,
                "Mean":      f"{cdf['value'].mean():.2f} {unit}",
                "Max":       f"{cdf['value'].max():.2f} {unit}",
                "Min":       f"{cdf['value'].min():.2f} {unit}",
                "Trend":     f"{slope:+.3f} {unit}/yr",
                "Direction": "▲ Rising" if slope > 0 else "▼ Falling",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("City"), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("<div class='sect-divider'></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#2a3a55;font-size:12px;padding:8px 0'>"
    "ClimaExplorer &nbsp;·&nbsp; Synthetic climate dataset for demonstration "
    "&nbsp;·&nbsp; Built with Streamlit · Plotly · Xarray"
    "</div>",
    unsafe_allow_html=True,
)