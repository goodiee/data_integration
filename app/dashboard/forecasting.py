import pandas as pd
import plotly.graph_objects as go
import requests
from config import API_BASE
from charts import build_empty_fig
from metrics import is_forecastable
from utils import to_canonical

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _ETS_AVAILABLE = True
except Exception:
    _ETS_AVAILABLE = False

def _make_forecast_df(df: pd.DataFrame, horizon: int, damped: bool = True) -> pd.DataFrame:
    if df.empty or df["value"].dropna().shape[0] < 10:
        return df.assign(kind="history")

    ts = (df.set_index("date")["value"].asfreq("D").interpolate("time")).clip(lower=0)

    try:
        model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=7, damped_trend=damped
                                     ).fit(optimized=True, use_brute=True)
    except Exception:
        model = ExponentialSmoothing(ts, trend="add", seasonal=None, damped_trend=damped).fit(optimized=True)

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

def render_forecast_chart(country: str, metric: str, start_date: str, end_date: str, horizon: int = 14) -> go.Figure:
    display_name = to_canonical(country) or country
    if not is_forecastable(metric):
        return build_empty_fig("Forecast is unavailable for the selected metric.")
    if not _ETS_AVAILABLE:
        return build_empty_fig("Install statsmodels to enable forecasting (pip install statsmodels).")

    try:
        resp = requests.post(f"{API_BASE}/metrics",
                             json={"country": country, "metric": metric, "start": start_date, "end": end_date},
                             timeout=20)
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
