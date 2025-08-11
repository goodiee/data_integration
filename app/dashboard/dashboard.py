import time
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
import sys, os

# --- optional: ETS for forecasting
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # ETS
    _ETS_AVAILABLE = True
except Exception:
    _ETS_AVAILABLE = False

# --- optional: sci-kit learn for clustering
try:
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    _SK_AVAILABLE = True
except Exception:
    _SK_AVAILABLE = False

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.core.constants import GDP_METRICS
from app.core.snowflake_conn import have_sf_config, get_sf_conn, db_schema

# -------------------------------
# Config & constants
# -------------------------------
st.set_page_config(page_title="COVID-19 Dashboard", page_icon="🧭", layout="wide")

API_BASE = "http://localhost:8000"

DATA_MIN = "2020-01-01"
DATA_MAX = "2023-12-31"

DATA_MIN_D = datetime.strptime(DATA_MIN, "%Y-%m-%d").date()
DATA_MAX_D = datetime.strptime(DATA_MAX, "%Y-%m-%d").date()

# --- session state init ---
if "chart_fig" not in st.session_state:
    st.session_state.chart_fig = None
if "chart_params" not in st.session_state:
    st.session_state.chart_params = {}
if "countries_default" not in st.session_state:
    st.session_state.countries_default = ["Lithuania"]  # seed; will be sanitized against options

# -------------------------------
# Styles & header
# -------------------------------
st.markdown(
    """
    <style>
    .topbar {background:#3f87c0; color:white; padding:12px 16px; border-radius:10px;}
    .badge {display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; margin-left:6px;}
    .badge-success {background:#198754; color:white;}
    .badge-secondary {background:#6c757d; color:white;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="topbar"><span>🧭</span> <strong>COVID-19 Interactive Dashboard</strong></div>',
    unsafe_allow_html=True
)

# -------------------------------
# Helpers
# -------------------------------

@st.cache_data(ttl=12*60*60, show_spinner=False)
def _mart_global_date_span():
    """
    Returns (min_date, max_date) that actually exist in MART_COUNTRY_DAY.
    Used to cap clustering windows and date picker defaults.
    """
    import pandas as pd
    if not have_sf_config():
        # fallback to known bounds if offline
        return pd.to_datetime("2020-01-01"), pd.to_datetime("2023-03-09")

    db, sch = db_schema()
    conn = get_sf_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute(f"""
                SELECT MIN(CAST(D AS DATE)) AS mn, MAX(CAST(D AS DATE)) AS mx
                FROM {db}.{sch}.MART_COUNTRY_DAY
            """)
            mn, mx = cur.fetchone()
            return pd.to_datetime(mn), pd.to_datetime(mx)
        finally:
            cur.close()
    finally:
        conn.close()

def _series_diagnostics(s: pd.Series) -> dict:
    if s is None or s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return {"ok": False, "reason": "empty-or-non-datetime", "len": 0, "nnz": 0, "first": None, "last": None}
    snnz = s.dropna()
    return {
        "ok": True,
        "reason": "",
        "len": int(len(s)),
        "nnz": int(len(snnz)),
        "first": s.index.min().date().isoformat() if len(s) else None,
        "last": s.index.max().date().isoformat() if len(s) else None,
    }

def _has_usable_weekly_series(s: pd.Series, min_week_points: int = 4) -> bool:
    # Try to ensure daily continuity, then weekly resample safety
    if s is None or s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return False
    s = s.sort_index()
    # Reindex to daily and interpolate for stability
    try:
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        s = s.reindex(full_idx).interpolate("time")
    except Exception:
        return False
    try:
        w = s.resample("W").mean()
    except Exception:
        return False
    return w.dropna().shape[0] >= min_week_points

def is_forecastable(metric: str) -> bool:
    # Only non-GDP daily time series
    return metric not in GDP_METRICS

def clamp_date(date_str: str) -> str:
    if not date_str:
        return DATA_MIN
    return max(DATA_MIN, min(date_str, DATA_MAX))

@st.cache_data(ttl=12*60*60, show_spinner=False)
def _load_alias_map_cached() -> dict:
    """Map alias -> canonical from COUNTRY_ALIAS."""
    if not have_sf_config():
        return {}
    db, sch = db_schema()
    conn = get_sf_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                SELECT UPPER(alias) AS alias_u, canonical
                FROM {db}.{sch}.COUNTRY_ALIAS
                WHERE alias IS NOT NULL AND canonical IS NOT NULL
                """
            )
            rows = cur.fetchall()
            return {alias_u: canonical for alias_u, canonical in rows}
        finally:
            cur.close()
    finally:
        conn.close()

@st.cache_data(ttl=12*60*60, show_spinner=False)
def _load_allowed_gdp_canonical_cached() -> set:
    """Canonical country names that exist in GDP_PPP_LONG (ONLY for GDP availability badge)."""
    if not have_sf_config():
        return set()
    db, sch = db_schema()
    conn = get_sf_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                SELECT DISTINCT
                       COALESCE(a.canonical, g.country) AS country_norm
                FROM {db}.{sch}.GDP_PPP_LONG g
                LEFT JOIN {db}.{sch}.COUNTRY_ALIAS a
                  ON UPPER(a.alias) = UPPER(g.country)
                WHERE g.country IS NOT NULL
                """
            )
            return {r[0] for r in cur.fetchall() if r[0]}
        finally:
            cur.close()
    finally:
        conn.close()

@st.cache_data(ttl=12*60*60, show_spinner=False)
def _load_countries_from_mart_cached() -> list[str]:
    """
    Countries come directly from MART_COUNTRY_DAY, canonicalized via COUNTRY_ALIAS if present.
    No dependency on GDP tables for selection or clustering.
    """
    fallback = ["Lithuania", "Latvia", "Estonia", "Poland", "Germany"]
    if not have_sf_config():
        return fallback

    db, sch = db_schema()
    conn = get_sf_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                SELECT DISTINCT
                       COALESCE(a.canonical, m.country) AS country_norm
                FROM {db}.{sch}.MART_COUNTRY_DAY m
                LEFT JOIN {db}.{sch}.COUNTRY_ALIAS a
                  ON UPPER(a.alias) = UPPER(m.country)
                WHERE m.country IS NOT NULL
                ORDER BY country_norm
                """
            )
            rows = [r[0] for r in cur.fetchall() if r[0]]
            return rows if rows else fallback
        finally:
            cur.close()
    finally:
        conn.close()

def to_canonical(country: str) -> str:
    if not country:
        return ""
    c = country.strip()
    alias_map = _load_alias_map_cached()
    canon = alias_map.get(c.upper())
    return canon if canon else c

def gdp_allowed(country: str) -> bool:
    allowed_set = _load_allowed_gdp_canonical_cached()
    return to_canonical(country) in allowed_set

def build_metric_choices(allow_gdp: bool):
    choices = []
    for label, value in _base_metric_options:
        if value in GDP_METRICS and not allow_gdp:
            continue
        choices.append((label, value))
    return choices

_base_metric_options = [
    ("New Cases", "NEW_CASES"),
    ("New Cases per 100k", "NEW_CASES_PER_100K"),
    ("New Deaths", "NEW_DEATHS"),
    ("New Deaths per 100k", "NEW_DEATHS_PER_100K"),
    ("Total Vaccinations", "TOTAL_VACCINATIONS"),
    ("Daily Vaccinations", "DAILY_VACCINATIONS"),
    ("People Vaccinated", "PEOPLE_VACCINATED"),
    ("People Fully Vaccinated", "PEOPLE_FULLY_VACCINATED"),
    ("Total Vaccinations per 100", "TOTAL_VACCINATIONS_PER_HUNDRED"),
    ("People Vaccinated per 100", "PEOPLE_VACCINATED_PER_HUNDRED"),
    ("People Fully Vaccinated per 100", "PEOPLE_FULLY_VACCINATED_PER_HUNDRED"),
    ("GDP PPP per Capita", "GDP_PPP_PER_CAPITA"),
    ("GDP vs Cases per 100k (Year)", "GDP_VS_CASES_PER100K_YEAR"),
]

# -------------------------------
# Forecast utilities
# -------------------------------
def _make_forecast_df(df: pd.DataFrame, horizon: int, damped: bool = True) -> pd.DataFrame:
    """
    df: ['date','value'] daily (may have gaps)
    returns: ['date','value','kind'] where kind in {'history','forecast','conf_low','conf_high'}
    """
    if df.empty or df["value"].dropna().shape[0] < 10:
        return df.assign(kind="history")

    ts = (df.set_index("date")["value"]
            .asfreq("D")
            .interpolate("time")).clip(lower=0)

    try:
        model = ExponentialSmoothing(
            ts, trend="add", seasonal="add", seasonal_periods=7, damped_trend=damped
        ).fit(optimized=True, use_brute=True)
    except Exception:
        model = ExponentialSmoothing(
            ts, trend="add", seasonal=None, damped_trend=damped
        ).fit(optimized=True)

    fcast = model.forecast(horizon)

    resid = (ts - model.fittedvalues.reindex(ts.index)).dropna()
    s = resid.std(ddof=1) if resid.size > 3 else 0.0
    lo = (fcast - 1.96*s).clip(lower=0)
    hi = (fcast + 1.96*s)

    out = [
        df.assign(kind="history"),
        pd.DataFrame({"date": fcast.index, "value": fcast.values}).assign(kind="forecast"),
        pd.DataFrame({"date": lo.index,     "value": lo.values}).assign(kind="conf_low"),
        pd.DataFrame({"date": hi.index,     "value": hi.values}).assign(kind="conf_high"),
    ]
    return pd.concat(out, ignore_index=True)

def build_empty_fig(msg: str) -> go.Figure:
    fig = go.Figure().add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig

def render_chart(country: str, metric: str, start_date: str, end_date: str):
    """Overview chart only (no forecast overlay)."""
    # Display with canonical, but send RAW selected country to API
    display_name = to_canonical(country) or country

    if metric in GDP_METRICS and not gdp_allowed(country):
        return build_empty_fig(f"GDP metrics are not available for '{display_name}'. Choose a different metric or country.")

    payload = {"country": country, "metric": metric, "start": start_date, "end": end_date}

    try:
        resp = requests.post(f"{API_BASE}/metrics", json=payload, timeout=20)
    except requests.exceptions.RequestException as e:
        return build_empty_fig(f"Error connecting to API: {e}")

    if resp.status_code != 200:
        return build_empty_fig(f"API returned {resp.status_code}: {resp.text}")

    data = resp.json().get("data", [])
    if not data:
        return build_empty_fig("No data available")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # GDP scatter (bubble)
    if metric.upper() == "GDP_VS_CASES_PER100K_YEAR":
        df_pivot = df.pivot(index="date", columns="metric", values="value").reset_index()
        needed = {"GDP_PPP_PER_CAPITA", "NEW_CASES_PER_100K"}
        if not needed.issubset(df_pivot.columns):
            return build_empty_fig("Required columns not returned by API.")
        df_pivot["year"] = df_pivot["date"].dt.year
        df_pivot = df_pivot.dropna(subset=["GDP_PPP_PER_CAPITA", "NEW_CASES_PER_100K"])
        df_pivot = df_pivot[df_pivot["GDP_PPP_PER_CAPITA"] > 0]
        df_pivot = df_pivot[df_pivot["NEW_CASES_PER_100K"] >= 0]
        if df_pivot.empty:
            return build_empty_fig("No valid GDP/cases rows to plot.")

        max_cases = df_pivot["NEW_CASES_PER_100K"].max()
        df_pivot["bubble_size"] = (df_pivot["NEW_CASES_PER_100K"] / max_cases) * 40 + 6

        fig = go.Figure(go.Scatter(
            x=df_pivot["GDP_PPP_PER_CAPITA"],
            y=df_pivot["NEW_CASES_PER_100K"],
            mode="markers",
            marker=dict(
                size=df_pivot["bubble_size"],
                color=df_pivot["year"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=dict(text="Year", side="top"), tickvals=[2020, 2021, 2022, 2023]),
                line=dict(width=0.5, color="DarkSlateGrey"),
                opacity=0.8,
            ),
            hovertemplate=(
                "GDP PPP per Capita: %{x:,.0f}<br>"
                "New Cases per 100k: %{y:,.1f}<br>"
                "Year: %{marker.color}<br>"
                "Date: %{text}<extra></extra>"
            ),
            text=df_pivot["date"].dt.strftime("%Y-%m-%d"),
            name="GDP vs Cases per 100k",
        ))
        fig.update_layout(
            title=f"GDP vs Cases per 100k in {display_name} ({start_date} → {end_date})",
            xaxis_title="GDP PPP per Capita (USD)",
            yaxis_title="New Cases per 100k",
            xaxis=dict(tickformat=","),
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=40, r=30, t=60, b=40),
        )
        return fig

    # GDP bar (yearly)
    if metric.upper() == "GDP_PPP_PER_CAPITA":
        df_year = df.set_index("date").resample("YE")["value"].last().reset_index()
        if df_year.empty:
            return build_empty_fig("No GDP data to plot.")
        df_year["year"] = df_year["date"].dt.year
        fig = go.Figure(go.Bar(
            x=df_year["year"],
            y=df_year["value"],
            name="GDP PPP per Capita",
            hovertemplate="Year: %{x}<br>GDP PPP per Capita: %{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            title=f"GDP PPP per Capita in {display_name} ({start_date} → {end_date})",
            xaxis_title="Year",
            yaxis_title="USD",
            xaxis=dict(type="category"),
            bargap=0.2,
            margin=dict(l=40, r=30, t=60, b=40),
        )
        return fig

    # Non-GDP default time-series
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["value"],
            mode="lines+markers",
            name="History",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{metric} in {display_name} ({start_date} → {end_date})",
        xaxis_title="Date",
        yaxis_title="Value",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig

def render_forecast_chart(country: str, metric: str, start_date: str, end_date: str, horizon: int = 14) -> go.Figure:
    display_name = to_canonical(country) or country
    if not is_forecastable(metric):
        return build_empty_fig("Forecast is unavailable for the selected metric.")
    if not _ETS_AVAILABLE:
        return build_empty_fig("Install statsmodels to enable forecasting (pip install statsmodels).")

    # Send RAW selected country to API
    payload = {"country": country, "metric": metric, "start": start_date, "end": end_date}
    try:
        resp = requests.post(f"{API_BASE}/metrics", json=payload, timeout=20)
    except requests.exceptions.RequestException as e:
        return build_empty_fig(f"Error connecting to API: {e}")
    if resp.status_code != 200:
        return build_empty_fig(f"API returned {resp.status_code}: {resp.text}")

    data = resp.json().get("data", [])
    if not data:
        return build_empty_fig("No data available")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df_fc = _make_forecast_df(df[["date", "value"]], horizon=horizon, damped=True)

    hist = df_fc[df_fc["kind"] == "history"]
    fc   = df_fc[df_fc["kind"] == "forecast"]
    lo   = df_fc[df_fc["kind"] == "conf_low"]
    hi   = df_fc[df_fc["kind"] == "conf_high"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["value"], mode="lines+markers", name="History"))
    if not fc.empty:
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["value"], mode="lines", name=f"Forecast (+{horizon}d)"))
        if not lo.empty and not hi.empty:
            fig.add_trace(go.Scatter(
                x=pd.concat([lo["date"], hi["date"][::-1]]),
                y=pd.concat([lo["value"], hi["value"][::-1]]),
                fill="toself", opacity=0.2, line=dict(width=0),
                name="95% interval", hoverinfo="skip"
            ))

    fig.update_layout(
        title=f"{metric} in {display_name} ({start_date} → {end_date})  + {horizon}d forecast",
        xaxis_title="Date", yaxis_title="Value",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig

# -------------------------------
# Filters (LIVE / reactive)
# -------------------------------
st.write("### Filters")
g1, g2, g3, g4 = st.columns([1.8, 1.5, 2, 0.8])

with g1:
    # Populate from MART_COUNTRY_DAY (canonicalized), no GDP dependency
    all_countries = _load_countries_from_mart_cached()

    # --- SAFE DEFAULTS to avoid StreamlitAPIException
    prev_defaults = st.session_state.countries_default or []
    safe_defaults = [c for c in prev_defaults if c in all_countries]
    if not safe_defaults and all_countries:
        safe_defaults = [all_countries[0]]

    country_selection = st.multiselect(
        "Countries",
        options=all_countries,
        default=safe_defaults,
        help="Select one or more countries."
    )
    # Persist only values that exist in options
    st.session_state.countries_default = [c for c in country_selection if c in all_countries] or safe_defaults

# Primary country (used by Overview/Forecast)
primary_country = country_selection[0] if country_selection else (all_countries[0] if all_countries else "")

allowed = gdp_allowed(primary_country) if primary_country else False
if allowed:
    st.markdown('<span class="badge badge-success">GDP metrics available (for primary country)</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="badge badge-secondary">GDP metrics unavailable for this country</span>', unsafe_allow_html=True)

with g2:
    metric_choices = build_metric_choices(allowed)
    metric_labels = [l for l, _ in metric_choices]
    metric_values = {l: v for l, v in metric_choices}
    default_label = "New Cases per 100k"
    default_index = metric_labels.index(default_label) if default_label in metric_labels else 0

    sel_label = st.selectbox(
        "Metric",
        options=metric_labels,
        index=default_index,
        key="metric_label",
        help="Choose what to visualize."
    )
    metric = metric_values[sel_label]
    st.session_state["metric_value"] = metric

# Use DB span to cap the default date range shown to user
db_min_dt, db_max_dt = _mart_global_date_span()
default_start = max(DATA_MIN_D, db_min_dt.date())
default_end   = min(DATA_MAX_D, db_max_dt.date())

with g3:
    dr = st.date_input(
        "Date range",
        key="date_range",
        value=st.session_state.get("date_range", (default_start, default_end)),
        min_value=DATA_MIN_D,
        max_value=DATA_MAX_D,
        help="Pick your date window."
    )
    if isinstance(dr, tuple):
        start_d, end_d = dr
    else:
        start_d = end_d = dr

with g4:
    update_clicked = st.button("Update chart")

# Dates for renderers
start_str = clamp_date(start_d.strftime("%Y-%m-%d"))
end_str   = clamp_date(end_d.strftime("%Y-%m-%d"))

# -------------------------------
# Tabs: Overview | (Forecast) | Clustering | Comments
# -------------------------------
tab_labels = ["Overview"]
# Forecast shown only if metric is forecastable AND a single country selected
show_forecast_tab = is_forecastable(metric) and len(country_selection) == 1
if show_forecast_tab:
    tab_labels.append("Forecast")
tab_labels.append("Clustering")
tab_labels.append("Comments")
tabs = st.tabs(tab_labels)

# --- Overview
with tabs[0]:
    if not country_selection:
        st.info("Select at least one country to render the chart.")
    else:
        if len(country_selection) > 1:
            st.caption(f"Showing Overview for primary country: **{to_canonical(primary_country) or primary_country}** (you selected {len(country_selection)} countries).")
        if update_clicked or st.session_state.chart_fig is None:
            with st.spinner("Rendering chart…"):
                fig = render_chart(primary_country, metric, start_str, end_str)
            st.session_state.chart_fig = fig
            st.session_state.chart_params = {"country": primary_country, "metric": metric, "start": start_str, "end": end_str}

        st.plotly_chart(st.session_state.chart_fig, use_container_width=True, theme=None)

# --- Forecast (only when applicable)
index_shift = 1 if show_forecast_tab else 0
if show_forecast_tab:
    with tabs[1]:
        st.markdown("#### Forecast")
        c1, c2 = st.columns([1, 1])
        with c1:
            horizon = st.number_input("Horizon (days)", 7, 90, 14, step=1, key="forecast_horizon_tab")
        with c2:
            st.caption("Model: ETS (weekly seasonality, damped trend)")
        if st.button("Run forecast", key="run_fc_btn"):
            with st.spinner("Fitting model…"):
                fc_fig = render_forecast_chart(primary_country, metric, start_str, end_str, horizon=int(horizon))
            st.plotly_chart(fc_fig, use_container_width=True, theme=None)
else:
    st.caption("Forecast tab is hidden for GDP metrics or when multiple countries are selected.")

# -------------------------------
# CLUSTERING (uses selected countries)
# -------------------------------
def _non_gdp_metric_labels():
    return [l for (l, v) in _base_metric_options if v not in GDP_METRICS]

@st.cache_data(ttl=60*30, show_spinner=False)
def _fetch_series(country: str, metric_value: str, start: str, end: str) -> pd.Series:
    """Fetch a single country's series and return daily pd.Series indexed by DatetimeIndex."""
    def _empty_dt_series():
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    # Send RAW selected country to API
    payload = {"country": country, "metric": metric_value, "start": start, "end": end}
    try:
        r = requests.post(f"{API_BASE}/metrics", json=payload, timeout=20)
    except requests.exceptions.RequestException:
        return _empty_dt_series()

    if r.status_code != 200:
        return _empty_dt_series()

    d = r.json().get("data", [])
    if not d:
        return _empty_dt_series()

    df = pd.DataFrame(d)
    if df.empty or "date" not in df or "value" not in df:
        return _empty_dt_series()

    idx = pd.DatetimeIndex(pd.to_datetime(df["date"], errors="coerce")).tz_localize(None)

    s = pd.Series(df["value"].values, index=idx).sort_index()
    # Ensure DatetimeIndex and daily frequency before returning
    s = s[~s.index.isna()]
    if s.empty:
        return _empty_dt_series()

    # Fill to daily frequency to stabilize downstream resampling
    try:
        s = s.asfreq("D").interpolate("time")
    except Exception:
        pass

    return s

def _build_features(series: pd.Series) -> dict:
    """Compact, robust features from weekly-mean series."""
    if series is None or len(series) == 0:
        return {}

    # Ensure DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            coerced_idx = pd.to_datetime(series.index, errors="coerce")
            series = pd.Series(series.values, index=coerced_idx)
            series = series[~series.index.isna()]
        except Exception:
            return {}

    if series.empty:
        return {}

    # Ensure sorted, monotonic
    series = series.sort_index()

    # Make sure we have (at least) daily freq for a clean weekly resample
    try:
        if series.index.freq is None:
            series = series.asfreq("D")
    except Exception:
        try:
            full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
            series = series.reindex(full_idx).interpolate("time")
        except Exception:
            return {}

    # Weekly aggregation
    try:
        w = series.resample("W").mean()
    except Exception:
        return {}

    # Need minimum length of weekly points to compute stable stats
    if w.dropna().shape[0] < 4:
        return {}

    # Features
    try:
        growth = (w.iloc[-1] - w.iloc[0]) / (abs(w.iloc[0]) + 1e-6)
    except Exception:
        growth = 0.0

    try:
        vol = w.pct_change().rolling(3).std().iloc[-8:].mean(skipna=True)
    except Exception:
        vol = 0.0

    try:
        peak_level = float(w.max())
        time_to_peak = int((w.idxmax() - w.index[0]).days)
    except Exception:
        peak_level = 0.0
        time_to_peak = 0

    try:
        recent_mean = float(w.iloc[-4:].mean())
    except Exception:
        recent_mean = 0.0

    try:
        recent_trend = (w.iloc[-1] - w.iloc[-4]) / (abs(w.iloc[-4]) + 1e-6) if w.shape[0] >= 5 else 0.0
    except Exception:
        recent_trend = 0.0

    # Clean NaNs
    growth = float(growth) if pd.notna(growth) else 0.0
    vol = float(vol) if pd.notna(vol) else 0.0
    peak_level = float(peak_level) if pd.notna(peak_level) else 0.0
    recent_mean = float(recent_mean) if pd.notna(recent_mean) else 0.0
    recent_trend = float(recent_trend) if pd.notna(recent_trend) else 0.0

    return {
        "growth": growth,
        "volatility": vol,
        "peak_level": peak_level,
        "time_to_peak": time_to_peak,
        "recent_mean": recent_mean,
        "recent_trend": recent_trend,
    }

with tabs[index_shift + 1]:
    st.markdown("#### Clustering (regions)")

    cl_metric_label = st.selectbox("Metric for clustering", _non_gdp_metric_labels(), key="cl_metric_label")
    cl_metric = dict(_base_metric_options)[cl_metric_label]

    c_top1, c_top2 = st.columns([1, 1])
    with c_top1:
        st.caption("Clustering uses exactly the countries selected above.")
    with c_top2:
        st.caption("Country list is loaded from MART_COUNTRY_DAY (canonicalized).")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        window_days = st.number_input("Look-back window (days)", min_value=30, max_value=400, value=180, step=10)
    with c2:
        n_clusters = st.number_input("Clusters (k)", min_value=2, max_value=10, value=4, step=1)
    with c3:
        normalize_switch = st.checkbox("Standardize features", value=True, help="Recommended for fair clustering.")

    run_clusters = st.button("Run clustering")

if run_clusters:
    countries_for_cluster = country_selection

    if not countries_for_cluster:
        st.warning("No countries selected. Pick at least one in the Filters section.")
    elif not _SK_AVAILABLE:
        st.warning("Install scikit-learn to enable clustering: `pip install scikit-learn`")
    else:
        # Cap clustering window to what actually exists in MART_COUNTRY_DAY
        db_min, db_max = _mart_global_date_span()
        end_c_dt = min(pd.to_datetime(end_str), db_max)
        start_c_dt = max(pd.to_datetime(DATA_MIN), end_c_dt - pd.Timedelta(days=int(window_days)))
        start_c = clamp_date(start_c_dt.strftime("%Y-%m-%d"))
        end_c   = clamp_date(end_c_dt.strftime("%Y-%m-%d"))

        # --- Scan countries & collect diagnostics
        diag_rows = []
        usable_rows, usable_names = [], []
        for ctry in countries_for_cluster[:80]:  # guard
            s = _fetch_series(ctry, cl_metric, start_c, end_c)
            d = _series_diagnostics(s)
            d["country"] = to_canonical(ctry) or ctry

            if d["ok"] and _has_usable_weekly_series(s, min_week_points=4):
                feats = _build_features(s)
                if feats:
                    usable_rows.append(feats)
                    usable_names.append(d["country"])
                    d["reason"] = ""
                else:
                    d["ok"] = False
                    d["reason"] = "insufficient-features"
            else:
                if d["reason"] == "":
                    d["reason"] = "insufficient-weekly-points"
                d["ok"] = False

            diag_rows.append(d)

        # --- Show diagnostics table for transparency
        diag_df = pd.DataFrame(diag_rows)[
            ["country", "ok", "reason", "len", "nnz", "first", "last"]
        ].sort_values(["ok", "country"], ascending=[False, True])
        with st.expander("Why some countries were skipped? (diagnostics)", expanded=False):
            st.dataframe(diag_df, use_container_width=True)

        if not usable_rows:
            st.warning(
                "No usable data for the selected inputs after filtering. "
                "Tips: try a different metric, increase the look-back window, "
                "or hand-pick a few countries with richer data."
            )
        else:
           
            X = pd.DataFrame(usable_rows, index=usable_names).fillna(0.0)
            Z = X.values
            if normalize_switch:
                Z = StandardScaler().fit_transform(Z)

            # ---- safety: tiny datasets
            n_samples = Z.shape[0]
            if n_samples < 2:
                st.warning("Not enough countries with usable data to cluster (need at least 2).")
                st.dataframe(X.round(3), use_container_width=True)
                st.stop()

            # cap clusters to a valid range
            n_clusters = int(n_clusters)
            n_clusters = max(2, min(n_clusters, n_samples))
            if n_clusters != int(st.session_state.get("last_n_clusters", n_clusters)):
                st.session_state["last_n_clusters"] = n_clusters  # optional UI memory
            if n_clusters > n_samples:
                st.warning(
                    f"Reducing clusters to {n_clusters} because only {n_samples} "
                    f"countries have data in the selected range."
                )

            # fit kmeans
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
            labels = kmeans.fit_predict(Z)

            # silhouette only makes sense if >1 label and >2 samples
            try:
                sil = silhouette_score(Z, labels) if (len(set(labels)) > 1 and n_samples > n_clusters) else float("nan")
            except Exception:
                sil = float("nan")

            result = X.copy()
            result["cluster"] = labels
            st.caption(f"Silhouette score: {sil:.3f}" if sil == sil else "Silhouette score: n/a")
            st.dataframe(result.sort_values(["cluster"] + list(X.columns)).round(3), use_container_width=True)

            # ---- PCA plot (robust n_components)
            try:
                pca_components = min(2, n_samples, Z.shape[1])
                if pca_components >= 2:
                    p = PCA(n_components=2, random_state=42).fit_transform(Z)
                    figc = go.Figure()
                    for k in sorted(set(labels)):
                        mask = labels == k
                        figc.add_trace(go.Scatter(
                            x=p[mask, 0], y=p[mask, 1],
                            mode="markers+text",
                            text=[n for n, m in zip(usable_names, mask) if m],
                            textposition="top center",
                            name=f"Cluster {k}",
                            opacity=0.9
                        ))
                    figc.update_layout(
                        title=f"Clusters (PCA view) — {cl_metric_label} — last {int(window_days)} days",
                        xaxis_title="PC1", yaxis_title="PC2",
                        margin=dict(l=40, r=30, t=60, b=40),
                        legend=dict(orientation="h")
                    )
                    st.plotly_chart(figc, use_container_width=True, theme=None)
                else:
                    st.info("Not enough samples to draw a 2D PCA plot.")
            except Exception as e:
                st.warning(f"PCA visualization failed: {e}")

# --- Comments (always last tab)
with tabs[-1]:
    # -------------------------------
    # Comments
    # -------------------------------
    if "comment_seed" not in st.session_state:
        st.session_state.comment_seed = 0

    st.markdown("### Add annotation")

    def fetch_comments(country: str, metric: str):
        try:
            r = requests.get(f"{API_BASE}/comments/{country}/{metric}", timeout=10)
            if r.status_code != 200:
                return False, [f"Error loading comments: {r.text}"]
            comments = r.json().get("comments", [])
            if not comments:
                return True, ["No comments yet."]
            return True, [f"{c['timestamp'][:10]} — {c['user']}: {c['text']}" for c in comments]
        except Exception as e:
            return False, [f"Error fetching comments: {e}"]

    seed = st.session_state.comment_seed
    user_key = f"comment_user_{seed}"
    text_key = f"comment_text_{seed}"
    btn_key  = f"comment_submit_{seed}"

    box = st.container(border=True)
    with box:
        c1, c2, c3 = st.columns([1, 3, 0.6])
        with c1:
            st.text_input("Your name", key=user_key)
        with c2:
            st.text_input("Comment", key=text_key)
        with c3:
            submit = st.button("Submit", key=btn_key)

    if submit:
        name = st.session_state.get(user_key, "").strip()
        text = st.session_state.get(text_key, "").strip()
        if not (name and text):
            st.warning("Please provide your name and a comment.")
        else:
            canon = to_canonical(primary_country or "")
            end_for_comment = clamp_date(
                (st.session_state.chart_params.get("end", DATA_MAX))
                if st.session_state.get("chart_params") else DATA_MAX
            )
            payload = {
                "country": canon,
                "date": end_for_comment,
                "metric": st.session_state.get("metric_value"),
                "user": name,
                "comment": text,
                "value": None,
            }
            try:
                r = requests.post(f"{API_BASE}/comments/add", json=payload, timeout=15)
                if r.status_code != 200:
                    st.error(f"Failed to add comment: {r.status_code} {r.text}")
                else:
                    st.toast("Comment added!", icon="✅")
                    st.session_state.comment_seed += 1
                    st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")

    canon = to_canonical(primary_country or "")
    metric_for_list = st.session_state.get("metric_value")
    ok, items = fetch_comments(canon, metric_for_list)
    with st.expander("Comments", expanded=True):
        if ok:
            for line in items:
                st.write(line)
        else:
            for line in items:
                st.warning(line)

    st.write("")
    st.write("")
