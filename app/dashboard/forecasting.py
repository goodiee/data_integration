from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import requests

from app.dashboard.config import API_BASE
from app.dashboard.charts import build_empty_fig
from app.dashboard.metrics import is_forecastable
from app.dashboard.utils import to_canonical

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _ETS_AVAILABLE = True
except Exception:
    _ETS_AVAILABLE = False


def _make_forecast_df(df: pd.DataFrame, horizon: int, damped: bool = True) -> pd.DataFrame:
    """
    This fun takes a history dataframe with columns ['date','value'] and builds a
    concatenated dataframe with segments: history, forecast, conf_low, conf_high.
    """
    if df.empty or df["value"].dropna().shape[0] < 10:
        # Not enough data to fit a model; just return history
        return df.assign(kind="history")

    # daily continuity then better seasonality
    ts = (
        df.set_index("date")["value"]
          .asfreq("D")
          .interpolate("time")
          .clip(lower=0)
    )

    # Try seasonal weekly model; fall back to non-seasonal if it fails
    try:
        model = ExponentialSmoothing(
            ts, trend="add", seasonal="add", seasonal_periods=7, damped_trend=damped
        ).fit(optimized=True, use_brute=True)
    except Exception:
        model = ExponentialSmoothing(
            ts, trend="add", seasonal=None, damped_trend=damped
        ).fit(optimized=True)

    fcast = model.forecast(horizon)

    # Simple normal-ish band from residuals std
    resid = (ts - model.fittedvalues.reindex(ts.index)).dropna()
    s = resid.std(ddof=1) if resid.size > 3 else 0.0
    lo = (fcast - 1.96 * s).clip(lower=0)
    hi = (fcast + 1.96 * s)

    out_parts = [
        df.assign(kind="history"),
        pd.DataFrame({"date": fcast.index, "value": fcast.values}).assign(kind="forecast"),
        pd.DataFrame({"date": lo.index,     "value": lo.values}).assign(kind="conf_low"),
        pd.DataFrame({"date": hi.index,     "value": hi.values}).assign(kind="conf_high"),
    ]
    return pd.concat(out_parts, ignore_index=True)


def render_forecast_chart(country: str, metric: str, start_date: str, end_date: str, horizon: int = 14) -> go.Figure:
    """
    It fetches history from the API and renders a chart with ETS forecast + 95% interval.
    """
    display_name = to_canonical(country) or country

    if not is_forecastable(metric):
        return build_empty_fig("Forecast is unavailable for the selected metric.")
    if not _ETS_AVAILABLE:
        return build_empty_fig("Install statsmodels to enable forecasting (pip install statsmodels).")

    # Load history via API
    try:
        resp = requests.post(
            f"{API_BASE}/metrics",
            json={"country": country, "metric": metric, "start": start_date, "end": end_date},
            timeout=20,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return build_empty_fig(f"Error connecting to API: {e}")
    except Exception as e:
        return build_empty_fig(f"Unexpected error: {e}")

    payload = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    data = payload.get("data", [])
    if not data:
        return build_empty_fig("No data available")

    # Prepare dataframe
    df = pd.DataFrame(data)
    if "date" not in df or "value" not in df:
        return build_empty_fig("API response missing required fields ('date', 'value').")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")

    df_fc = _make_forecast_df(df[["date", "value"]], horizon=horizon, damped=True)

    hist = df_fc[df_fc["kind"] == "history"]
    fc   = df_fc[df_fc["kind"] == "forecast"]
    lo   = df_fc[df_fc["kind"] == "conf_low"]
    hi   = df_fc[df_fc["kind"] == "conf_high"]

    # Build chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["value"], mode="lines+markers", name="History",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>",
    ))
    if not fc.empty:
        fig.add_trace(go.Scatter(
            x=fc["date"], y=fc["value"], mode="lines", name=f"Forecast (+{horizon}d)",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: %{y:.2f}<extra></extra>",
        ))
        if not lo.empty and not hi.empty:
            fig.add_trace(go.Scatter(
                x=pd.concat([lo["date"], hi["date"][::-1]]),
                y=pd.concat([lo["value"], hi["value"][::-1]]),
                fill="toself", opacity=0.2, line=dict(width=0),
                name="95% interval", hoverinfo="skip",
            ))

    fig.update_layout(
        title=f"{metric} in {display_name} ({start_date} → {end_date})  + {horizon}d forecast",
        xaxis_title="Date", yaxis_title="Value",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig
