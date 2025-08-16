import pandas as pd
import plotly.graph_objects as go
import requests
from config import API_BASE
from utils import to_canonical, gdp_allowed
from app.core.constants import GDP_METRICS

# NEW: import MR helpers
from app.dashboard.data_access import (
    mr_rising_streaks,
    mr_spike_days,
    mr_vax_surge,
)

# ---------------------------
# Base helpers (existing)
# ---------------------------
def build_empty_fig(msg: str) -> go.Figure:
    fig = go.Figure().add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig

def render_chart(country: str, metric: str, start_date: str, end_date: str) -> go.Figure:
    display_name = to_canonical(country) or country

    if metric in GDP_METRICS and not gdp_allowed(country):
        return build_empty_fig(f"GDP metrics are not available for '{display_name}'. Choose a different metric or country.")

    try:
        resp = requests.post(
            f"{API_BASE}/metrics",
            json={"country": country, "metric": metric, "start": start_date, "end": end_date},
            timeout=20,
        )
    except requests.exceptions.RequestException as e:
        return build_empty_fig(f"Error connecting to API: {e}")

    if resp.status_code != 200:
        return build_empty_fig(f"API returned {resp.status_code}: {resp.text}")

    data = resp.json().get("data", [])
    if not data:
        return build_empty_fig("No data available")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    if metric.upper() == "GDP_VS_CASES_PER100K_YEAR":
        dfp = df.pivot(index="date", columns="metric", values="value").reset_index()
        needed = {"GDP_PPP_PER_CAPITA", "NEW_CASES_PER_100K"}
        if not needed.issubset(dfp.columns):
            return build_empty_fig("Required columns not returned by API.")
        dfp["year"] = dfp["date"].dt.year
        dfp = dfp.dropna(subset=["GDP_PPP_PER_CAPITA", "NEW_CASES_PER_100K"])
        dfp = dfp[(dfp["GDP_PPP_PER_CAPITA"] > 0) & (dfp["NEW_CASES_PER_100K"] >= 0)]
        if dfp.empty:
            return build_empty_fig("No valid GDP/cases rows to plot.")

        max_cases = dfp["NEW_CASES_PER_100K"].max()
        dfp["bubble_size"] = (dfp["NEW_CASES_PER_100K"] / max_cases) * 40 + 6

        fig = go.Figure(go.Scatter(
            x=dfp["GDP_PPP_PER_CAPITA"],
            y=dfp["NEW_CASES_PER_100K"],
            mode="markers",
            marker=dict(
                size=dfp["bubble_size"],
                color=dfp["year"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=dict(text="Year", side="top"), tickvals=[2020, 2021, 2022, 2023]),
                line=dict(width=0.5, color="DarkSlateGrey"),
                opacity=0.8,
            ),
            hovertemplate=("GDP PPP per Capita: %{x:,.0f}<br>"
                           "New Cases per 100k: %{y:,.1f}<br>"
                           "Year: %{marker.color}<br>"
                           "Date: %{text}<extra></extra>"),
            text=dfp["date"].dt.strftime("%Y-%m-%d"),
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

    fig = go.Figure(go.Scatter(
        x=df["date"], y=df["value"], mode="lines+markers", name="History",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{metric} in {display_name} ({start_date} → {end_date})",
        xaxis_title="Date", yaxis_title="Value",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig


# ============================================
# NEW: match-recognition overlays
# ============================================

def _fetch_series(country: str, metric: str, start_date: str, end_date: str):
    """Fetch a single metric series from your API and return (df or None, err str or None)."""
    try:
        resp = requests.post(
            f"{API_BASE}/metrics",
            json={"country": country, "metric": metric, "start": start_date, "end": end_date},
            timeout=20,
        )
    except requests.exceptions.RequestException as e:
        return None, f"Error connecting to API: {e}"
    if resp.status_code != 200:
        return None, f"API returned {resp.status_code}: {resp.text}"
    data = resp.json().get("data", [])
    if not data:
        return None, "No data available"
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df, None


def render_mr_rising_streaks_chart(country: str, start_date: str, end_date: str, min_len: int = 3) -> go.Figure:
    """Line of NEW_CASES with translucent bands for each rising streak."""
    display = to_canonical(country) or country

    # series
    df, err = _fetch_series(country, "NEW_CASES", start_date, end_date)
    if err:
        return build_empty_fig(err)

    # matches
    mr_df = mr_rising_streaks(country, start_date, end_date, int(min_len))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["value"], mode="lines+markers", name="New cases",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>New cases: %{y:.0f}<extra></extra>",
    ))

    if not mr_df.empty:
        # add shaded spans per streak (yref='paper' covers full height)
        for _, row in mr_df.iterrows():
            x0 = pd.to_datetime(row["MR_START_DAY"])
            x1 = pd.to_datetime(row["MR_END_DAY"])
            fig.add_shape(
                type="rect",
                xref="x", yref="paper",
                x0=x0, x1=x1,
                y0=0, y1=1,
                fillcolor="rgba(0, 200, 0, 0.12)",
                line=dict(width=0),
                layer="below",
            )
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(mr_df["MR_START_DAY"]),
            y=[None]*len(mr_df),  # legend handle only
            mode="markers",
            marker=dict(size=10, symbol="square", color="rgba(0,200,0,0.6)"),
            name=f"Rising streaks (≥ {min_len} days)",
            hoverinfo="skip",
            showlegend=True,
        ))
    else:
        fig.add_annotation(text="No rising streaks found", xref="paper", yref="paper", x=0.5, y=0.92, showarrow=False)

    fig.update_layout(
        title=f"Rising streaks of new cases in {display} ({start_date} → {end_date})",
        xaxis_title="Date", yaxis_title="New cases",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig


def render_mr_spike_days_chart(country: str, start_date: str, end_date: str, spike_mult: float = 1.5) -> go.Figure:
    """Line of NEW_CASES with spike-day markers."""
    display = to_canonical(country) or country

    df, err = _fetch_series(country, "NEW_CASES", start_date, end_date)
    if err:
        return build_empty_fig(err)

    mr_df = mr_spike_days(country, start_date, end_date, float(spike_mult))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["value"], mode="lines+markers", name="New cases",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>New cases: %{y:.0f}<extra></extra>",
    ))

    if not mr_df.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(mr_df["MR_SPIKE_DAY"]),
            y=mr_df["MR_SPIKE_CASES"],
            mode="markers",
            name=f"Spikes (≥ {spike_mult}× prev day)",
            marker=dict(size=10, symbol="diamond", color="rgba(220, 20, 60, 0.85)", line=dict(width=1)),
            hovertemplate=("Spike day: %{x|%Y-%m-%d}<br>"
                           "Prev cases: %{customdata[0]:,.0f}<br>"
                           "Spike cases: %{y:,.0f}<br>"
                           "Jump: %{customdata[1]:.1f}%<extra></extra>"),
            customdata=mr_df[["MR_PREV_CASES", "MR_PCT_JUMP"]].values,
        ))
    else:
        fig.add_annotation(text="No spike days found", xref="paper", yref="paper", x=0.5, y=0.92, showarrow=False)

    fig.update_layout(
        title=f"Spike days in new cases for {display} ({start_date} → {end_date})",
        xaxis_title="Date", yaxis_title="New cases",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig


def render_mr_vax_surge_chart(country: str, start_date: str, end_date: str, min_len: int = 5) -> go.Figure:
    """Line of DAILY_VACCINATIONS with translucent bands for each surge."""
    display = to_canonical(country) or country

    df, err = _fetch_series(country, "DAILY_VACCINATIONS", start_date, end_date)
    if err:
        return build_empty_fig(err)

    mr_df = mr_vax_surge(country, start_date, end_date, int(min_len))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["value"], mode="lines+markers", name="Daily vaccinations",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Doses: %{y:,.0f}<extra></extra>",
    ))

    if not mr_df.empty:
        for _, row in mr_df.iterrows():
            x0 = pd.to_datetime(row["MR_START_DAY"])
            x1 = pd.to_datetime(row["MR_END_DAY"])
            fig.add_shape(
                type="rect",
                xref="x", yref="paper",
                x0=x0, x1=x1,
                y0=0, y1=1,
                fillcolor="rgba(30, 144, 255, 0.12)",
                line=dict(width=0),
                layer="below",
            )
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(mr_df["MR_START_DAY"]),
            y=[None]*len(mr_df),
            mode="markers",
            marker=dict(size=10, symbol="square", color="rgba(30,144,255,0.6)"),
            name=f"Vaccination surges (≥ {min_len} days)",
            hoverinfo="skip",
            showlegend=True,
        ))
    else:
        fig.add_annotation(text="No vaccination surges found", xref="paper", yref="paper", x=0.5, y=0.92, showarrow=False)

    fig.update_layout(
        title=f"Vaccination surges in {display} ({start_date} → {end_date})",
        xaxis_title="Date", yaxis_title="Doses",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig


# Dispatcher if you want a single entry point from the UI:
def render_match_recognition_chart(
    pattern: str,
    country: str,
    start_date: str,
    end_date: str,
    min_len: int | None = None,
    spike_mult: float | None = None,
    vax_min_len: int | None = None,
) -> go.Figure:
    """
    pattern: "Rising streaks (cases)" | "Spike day (cases)" | "Vaccination surge"
    Other params are optional; sensible defaults used if omitted.
    """
    if pattern == "Rising streaks (cases)":
        return render_mr_rising_streaks_chart(country, start_date, end_date, min_len or 3)
    if pattern == "Spike day (cases)":
        return render_mr_spike_days_chart(country, start_date, end_date, spike_mult or 1.5)
    if pattern == "Vaccination surge":
        return render_mr_vax_surge_chart(country, start_date, end_date, vax_min_len or 5)
    return build_empty_fig("Unknown pattern")
