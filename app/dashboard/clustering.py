import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
from config import API_BASE, DATA_MIN
from utils import to_canonical
from charts import build_empty_fig


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def series_diagnostics(s: pd.Series) -> dict:
    if s is None or s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return {"ok": False, "reason": "empty-or-non-datetime", "len": 0, "nnz": 0, "first": None, "last": None}
    snnz = s.dropna()
    return {
        "ok": True, "reason": "",
        "len": int(len(s)), "nnz": int(len(snnz)),
        "first": s.index.min().date().isoformat() if len(s) else None,
        "last": s.index.max().date().isoformat() if len(s) else None,
    }

def has_usable_weekly_series(s: pd.Series, min_week_points: int = 4) -> bool:
    if s is None or s.empty or not isinstance(s.index, pd.DatetimeIndex):
        return False
    s = s.sort_index()
    try:
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        s = s.reindex(full_idx).interpolate("time")
        w = s.resample("W").mean()
    except Exception:
        return False
    return w.dropna().shape[0] >= min_week_points

@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_series(country: str, metric_value: str, start: str, end: str) -> pd.Series:
    def _empty():
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    payload = {"country": country, "metric": metric_value, "start": start, "end": end}
    try:
        r = requests.post(f"{API_BASE}/metrics", json=payload, timeout=20)
    except requests.exceptions.RequestException:
        return _empty()
    if r.status_code != 200:
        return _empty()

    d = r.json().get("data", [])
    if not d:
        return _empty()

    df = pd.DataFrame(d)
    if df.empty or "date" not in df or "value" not in df:
        return _empty()

    idx = pd.DatetimeIndex(pd.to_datetime(df["date"], errors="coerce")).tz_localize(None)
    s = pd.Series(df["value"].values, index=idx).sort_index()
    s = s[~s.index.isna()]
    if s.empty:
        return _empty()

    try:
        s = s.asfreq("D").interpolate("time")
    except Exception:
        pass
    return s

def build_features(series: pd.Series) -> dict:
    if series is None or len(series) == 0:
        return {}
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(series.index, errors="coerce")
            series = pd.Series(series.values, index=idx)
            series = series[~series.index.isna()]
        except Exception:
            return {}
    if series.empty:
        return {}

    series = series.sort_index()
    try:
        if series.index.freq is None:
            series = series.asfreq("D")
    except Exception:
        try:
            full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
            series = series.reindex(full_idx).interpolate("time")
        except Exception:
            return {}

    try:
        w = series.resample("W").mean()
    except Exception:
        return {}

    if w.dropna().shape[0] < 4:
        return {}

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

    def _clean(x): return float(x) if pd.notna(x) else 0.0
    return {
        "growth": _clean(growth),
        "volatility": _clean(vol),
        "peak_level": _clean(peak_level),
        "time_to_peak": int(time_to_peak),
        "recent_mean": _clean(recent_mean),
        "recent_trend": _clean(recent_trend),
    }

def run_clustering(country_selection, metric_label: str, metric_value: str,
                   window_days: int, end_dt, min_start: str = DATA_MIN,
                   standardize: bool = True):
    """
    Returns dict with:
      diag_df, result_df (features+cluster), silhouette, pca_fig (or None)
      and possibly warnings as list of strings.
    """
    warnings = []
    end_c_dt = end_dt
    start_c_dt = max(pd.to_datetime(min_start), end_c_dt - pd.Timedelta(days=int(window_days)))
    start_c = start_c_dt.strftime("%Y-%m-%d")
    end_c   = end_c_dt.strftime("%Y-%m-%d")

    diag_rows = []
    usable_rows, usable_names = [], []
    for ctry in country_selection[:80]:
        s = fetch_series(ctry, metric_value, start_c, end_c)
        d = series_diagnostics(s)
        d["country"] = to_canonical(ctry) or ctry
        if d["ok"] and has_usable_weekly_series(s, min_week_points=4):
            feats = build_features(s)
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

    diag_df = pd.DataFrame(diag_rows)[["country", "ok", "reason", "len", "nnz", "first", "last"]]
    if not usable_rows:
        return {
            "warnings": ["No usable data after filtering. Try a different metric/window or more countries."],
            "diag_df": diag_df.sort_values(["ok", "country"], ascending=[False, True]),
            "result_df": pd.DataFrame(),
            "silhouette": float("nan"),
            "pca_fig": build_empty_fig("Not enough samples for PCA."),
        }

    X = pd.DataFrame(usable_rows, index=usable_names).fillna(0.0)
    Z = StandardScaler().fit_transform(X.values) if (standardize) else X.values

    n_samples = Z.shape[0]
    if n_samples < 2:
        warnings.append("`Please, select more samples`")
        return {
            "warnings": warnings,
            "diag_df": diag_df.sort_values(["ok", "country"], ascending=[False, True]),
            "result_df": X.round(3),
            "silhouette": float("nan"),
            "pca_fig": build_empty_fig("Not enough samples to draw PCA."),
        }

    k = max(2, min(4, n_samples))
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(Z)
    try:
        sil = silhouette_score(Z, labels) if (len(set(labels)) > 1 and n_samples > k) else float("nan")
    except Exception:
        sil = float("nan")

    result = X.copy()
    result["cluster"] = labels

    pca_fig = None
    if min(2, n_samples, Z.shape[1]) >= 2:
        p = PCA(n_components=2, random_state=42).fit_transform(Z)
        figc = go.Figure()
        for k_lab in sorted(set(labels)):
            m = labels == k_lab
            figc.add_trace(go.Scatter(
                x=p[m, 0], y=p[m, 1], mode="markers+text",
                text=[n for n, mm in zip(usable_names, m) if mm],
                textposition="top center", name=f"Cluster {k_lab}", opacity=0.9
            ))
        figc.update_layout(
            title=f"Clusters (PCA view) — {metric_label} — last {int(window_days)} days",
            xaxis_title="PC1", yaxis_title="PC2",
            margin=dict(l=40, r=30, t=60, b=40),
            legend=dict(orientation="h")
        )
        pca_fig = figc
    else:
        pca_fig = build_empty_fig("Not enough samples to draw a 2D PCA plot.")

    return {
        "warnings": warnings,
        "diag_df": diag_df.sort_values(["ok", "country"], ascending=[False, True]),
        "result_df": result.sort_values(["cluster"] + list(X.columns)).round(3),
        "silhouette": sil,
        "pca_fig": pca_fig,
    }
