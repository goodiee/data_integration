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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.core.constants import GDP_METRICS
from app.core.snowflake_conn import have_sf_config, get_sf_conn

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
def is_forecastable(metric: str) -> bool:
    # Only non-GDP daily time series
    return metric not in GDP_METRICS

def clamp_date(date_str: str) -> str:
    if not date_str:
        return DATA_MIN
    return max(DATA_MIN, min(date_str, DATA_MAX))

@st.cache_data(ttl=12*60*60, show_spinner=False)
def _load_alias_map_cached() -> dict:
    if not have_sf_config():
        return {}
    conn = get_sf_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT UPPER(alias) AS alias_u, canonical
                FROM COVID_DB.PUBLIC.COUNTRY_ALIAS
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
    if not have_sf_config():
        return set()
    conn = get_sf_conn()
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT DISTINCT
                       COALESCE(a.canonical, g.country) AS country_norm
                FROM COVID_DB.PUBLIC.GDP_PPP_LONG g
                LEFT JOIN COVID_DB.PUBLIC.COUNTRY_ALIAS a
                  ON UPPER(a.alias) = UPPER(g.country)
                WHERE g.country IS NOT NULL
                """
            )
            return {r[0] for r in cur.fetchall() if r[0]}
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

def build_metric_choices(allow_gdp: bool):
    choices = []
    for label, value in _base_metric_options:
        if value in GDP_METRICS and not allow_gdp:
            continue
        choices.append((label, value))
    return choices

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
    canon = to_canonical(country) or country

    if metric in GDP_METRICS and not gdp_allowed(country):
        return build_empty_fig(f"GDP metrics are not available for '{canon}'. Choose a different metric or country.")

    payload = {"country": canon, "metric": metric, "start": start_date, "end": end_date}

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
            title=f"GDP vs Cases per 100k in {canon} ({start_date} → {end_date})",
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
            title=f"GDP PPP per Capita in {canon} ({start_date} → {end_date})",
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
        title=f"{metric} in {canon} ({start_date} → {end_date})",
        xaxis_title="Date",
        yaxis_title="Value",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig

def render_forecast_chart(country: str, metric: str, start_date: str, end_date: str, horizon: int = 14) -> go.Figure:
    canon = to_canonical(country) or country
    if not is_forecastable(metric):
        return build_empty_fig("Forecast is unavailable for the selected metric.")
    if not _ETS_AVAILABLE:
        return build_empty_fig("Install statsmodels to enable forecasting (pip install statsmodels).")

    payload = {"country": canon, "metric": metric, "start": start_date, "end": end_date}
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
        title=f"{metric} in {canon} ({start_date} → {end_date})  + {horizon}d forecast",
        xaxis_title="Date", yaxis_title="Value",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig

# -------------------------------
# Filters (LIVE / reactive)
# -------------------------------
st.write("### Filters")
g1, g2, g3, g4 = st.columns([1.3, 1.5, 2, 0.8])

with g1:
    country = st.text_input(
        "Country",
        value=st.session_state.get("country", "Lithuania"),
        key="country",
        help="Type a country name (e.g., Lithuania, Germany)."
    )

allowed = gdp_allowed(country)
if allowed:
    st.markdown('<span class="badge badge-success">GDP metrics available</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="badge badge-secondary">GDP metrics unavailable for this country</span>', unsafe_allow_html=True)

with g2:
    metric_choices = build_metric_choices(allowed)
    metric_labels = [l for l, _ in metric_choices]
    metric_values = {l: v for l, v in metric_choices}
    default_label = "New Cases per 100k"

    sel_label = st.selectbox(
        "Metric",
        options=metric_labels,
        index=metric_labels.index(default_label) if default_label in metric_labels else 0,
        key="metric_label",
        help="Choose what to visualize."
    )
    metric = metric_values[sel_label]
    st.session_state["metric_value"] = metric

with g3:
    dr = st.date_input(
        "Date range",
        key="date_range",
        value=st.session_state.get("date_range", (DATA_MIN_D, DATA_MAX_D)),
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
# Tabs: Overview | (Forecast) | Comments
# -------------------------------
tab_labels = ["Overview"]
if is_forecastable(metric):
    tab_labels.append("Forecast")   # hidden automatically for GDP metrics
tab_labels.append("Comments")
tabs = st.tabs(tab_labels)

# --- Overview
with tabs[0]:
    if update_clicked:
        with st.spinner("Rendering chart…"):
            fig = render_chart(country, metric, start_str, end_str)
        st.session_state.chart_fig = fig
        st.session_state.chart_params = {"country": country, "metric": metric, "start": start_str, "end": end_str}

    if st.session_state.chart_fig is not None:
        st.plotly_chart(st.session_state.chart_fig, use_container_width=True, theme=None)
    else:
        st.info("Adjust filters and click **Update chart** to render.")

# --- Forecast (only for non-GDP)
if is_forecastable(metric):
    with tabs[1]:
        st.markdown("#### Forecast")
        c1, c2 = st.columns([1, 1])
        with c1:
            horizon = st.number_input("Horizon (days)", 7, 90, 14, step=1, key="forecast_horizon_tab")
        with c2:
            st.caption("Model: ETS (weekly seasonality, damped trend)")

        if st.button("Run forecast", key="run_fc_btn"):
            with st.spinner("Fitting model…"):
                fc_fig = render_forecast_chart(country, metric, start_str, end_str, horizon=int(horizon))
            st.plotly_chart(fc_fig, use_container_width=True, theme=None)

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
            canon = to_canonical(st.session_state.get("country") or "") or (st.session_state.get("country") or "")
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

    canon = to_canonical(st.session_state.get("country") or "") or (st.session_state.get("country") or "")
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
