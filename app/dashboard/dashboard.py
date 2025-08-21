from __future__ import annotations

import os
import sys
import pandas as pd
import streamlit as st

# project root is importable (kept for local runs)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.dashboard.config import set_page, DATA_MIN_D, DATA_MAX_D
from app.dashboard.styles import inject as inject_styles
from app.dashboard.metrics import (
    build_metric_choices,
    non_gdp_metric_labels,
    base_metric_options,
    is_forecastable,
)
from app.dashboard.data_access import (
    mart_global_date_span,
    load_countries_from_mart,
    mr_rising_streaks,
    mr_spike_days,
    mr_vax_surge,
)
from app.dashboard.charts import render_chart, render_match_recognition_chart
from app.dashboard.utils import to_canonical, gdp_allowed, clamp_date
from app.dashboard.forecasting import render_forecast_chart
from app.dashboard.clustering import run_clustering
from app.dashboard.comments import fetch_comments, add_comment

# Unique key namespace (avoids Streamlit duplicate-key collisions)
KEY_PREFIX = __name__.replace(".", "_")

# Page + styles
set_page()
inject_styles()

# Session defaults
st.session_state.setdefault("chart_fig", None)
st.session_state.setdefault("chart_params", {})
st.session_state.setdefault("countries_default", ["Lithuania"])

# Filters header
st.write("### Filters")

g1, g2, g3, g4 = st.columns([1.8, 1.5, 2, 0.8])

# Countries (g1)
with g1:
    all_countries = load_countries_from_mart()

    prev_defaults = st.session_state.countries_default or []
    safe_defaults = [c for c in prev_defaults if c in all_countries] or (
        [all_countries[0]] if all_countries else []
    )

    country_selection = st.multiselect(
        "Countries",
        options=all_countries,
        default=safe_defaults,
        key=f"{KEY_PREFIX}_countries_ms",
        help="Select one or more countries.",
    )

    # persist last good selection
    st.session_state.countries_default = (
        [c for c in country_selection if c in all_countries] or safe_defaults
    )

# Primary country = first selection (or first available)
primary_country = (
    country_selection[0]
    if country_selection
    else (all_countries[0] if all_countries else "")
)

allowed_gdp = gdp_allowed(primary_country) if primary_country else False
st.markdown(
    '<span class="badge badge-success">GDP metrics available (for primary country)</span>'
    if allowed_gdp
    else '<span class="badge badge-secondary">GDP metrics unavailable for this country</span>',
    unsafe_allow_html=True,
)

# Metric choices
metric_choices = build_metric_choices(allowed_gdp)
metric_labels = [lbl for lbl, _ in metric_choices]
metric_map = {lbl: val for lbl, val in metric_choices}
default_label = "New Cases per 100k"
default_metric_value = metric_map.get(
    default_label, metric_choices[0][1] if metric_choices else "NEW_CASES_PER_100K"
)

st.session_state.setdefault("metric_value", default_metric_value)
metric = st.session_state["metric_value"]

# DB date span (clamp UI range to actual data)
db_min_dt, db_max_dt = mart_global_date_span()
default_start = max(DATA_MIN_D, db_min_dt.date())
default_end = min(DATA_MAX_D, db_max_dt.date())

# Section switch (acts like tabs)
section_labels = ["Overview", "Match recognition"]
if is_forecastable(metric) and len(country_selection) == 1:
    section_labels.append("Forecast")
section_labels += ["Clustering", "Comments"]

section = st.radio(
    "Section",
    options=section_labels,
    horizontal=True,
    key=f"{KEY_PREFIX}_section",
)

# Date range (always visible)
rng_key = f"{KEY_PREFIX}_date_range"
if not st.session_state.get(rng_key):
    st.session_state[rng_key] = (default_start, default_end)

with g3:
    c_start, c_end = st.columns(2)
    with c_start:
        start_d = st.date_input(
            "Start date",
            key=f"{KEY_PREFIX}_start_date",
            value=st.session_state[rng_key][0],
            min_value=DATA_MIN_D,
            max_value=DATA_MAX_D,
        )
    with c_end:
        end_d = st.date_input(
            "End date",
            key=f"{KEY_PREFIX}_end_date",
            value=st.session_state[rng_key][1],
            min_value=DATA_MIN_D,
            max_value=DATA_MAX_D,
        )

    # keep dates sane
    if start_d > end_d:
        start_d, end_d = end_d, start_d
    st.session_state[rng_key] = (start_d, end_d)

# Metric control (g2): enabled on Overview & Forecast
with g2:
    try:
        current_label = next(lbl for lbl, val in metric_choices if val == st.session_state["metric_value"])
    except StopIteration:
        current_label = default_label if default_label in metric_labels else (metric_labels[0] if metric_labels else "")
        if current_label:
            st.session_state["metric_value"] = metric_map[current_label]

    control_enabled = section in ("Overview", "Forecast")

    sel_label = st.selectbox(
        "Metric",
        options=metric_labels,
        index=metric_labels.index(current_label) if current_label in metric_labels else 0,
        key=f"{KEY_PREFIX}_metric_label",
        help=("Choose what to visualize." if control_enabled
              else "Metric is shown for context. Switch to Overview or Forecast to change it."),
        disabled=not control_enabled,
    )
    if control_enabled and sel_label in metric_map:
        st.session_state["metric_value"] = metric_map[sel_label]
    metric = st.session_state["metric_value"]

# Update button (g4) only on Overview
with g4:
    st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
    update_clicked_overview = st.button("Update chart", key=f"{KEY_PREFIX}_update_chart_btn_overview")

# String dates used by renderers
start_str = clamp_date(start_d.strftime("%Y-%m-%d"))
end_str = clamp_date(end_d.strftime("%Y-%m-%d"))

# Sections

# Overview
if section == "Overview":
    if not country_selection:
        st.info("Select at least one country to render the chart.")
    else:
        if len(country_selection) > 1:
            st.caption(
                f"Showing Overview for primary country: **{to_canonical(primary_country) or primary_country}** "
                f"(you selected {len(country_selection)} countries)."
            )

        # render on first load / button click / param changes
        params_changed = any(
            st.session_state.chart_params.get(k) != v
            for k, v in {
                "metric": metric,
                "country": primary_country,
                "start": start_str,
                "end": end_str,
            }.items()
        )
        if update_clicked_overview or st.session_state.chart_fig is None or params_changed:
            with st.spinner("Rendering chart…"):
                fig = render_chart(primary_country, metric, start_str, end_str)
            st.session_state.chart_fig = fig
            st.session_state.chart_params = {
                "country": primary_country,
                "metric": metric,
                "start": start_str,
                "end": end_str,
            }

        st.plotly_chart(st.session_state.chart_fig, use_container_width=True, theme=None)

# Match recognition
elif section == "Match recognition":
    st.markdown("#### Match recognition")

    if not primary_country:
        st.info("Select at least one country to run patterns.")
    else:
        pattern = st.selectbox(
            "Pattern",
            ["Rising streaks (cases)", "Spike day (cases)", "Vaccination surge"],
            key=f"{KEY_PREFIX}_mr_pattern",
            help="Only the relevant controls for the chosen pattern will appear.",
        )

        # relevant controls only
        params_box = st.container()
        with params_box:
            if pattern == "Rising streaks (cases)":
                min_len = st.number_input(
                    "Min streak length",
                    min_value=2, max_value=30, value=3, step=1,
                    key=f"{KEY_PREFIX}_mr_min_len",
                    help="Number of consecutive up-days required.",
                )
            elif pattern == "Spike day (cases)":
                spike_mult = st.number_input(
                    "Spike × previous",
                    min_value=1.1, max_value=10.0, value=1.5, step=0.1,
                    key=f"{KEY_PREFIX}_mr_spike_mult",
                    help="A day is a spike if cases ≥ this multiplier × previous day.",
                )
            else:
                vax_min_len = st.number_input(
                    "Vax surge min len",
                    min_value=2, max_value=30, value=5, step=1,
                    key=f"{KEY_PREFIX}_mr_vax_min_len",
                    help="Consecutive up-days in DAILY_VACCINATIONS required.",
                )

        st.caption(
            f"Country: **{to_canonical(primary_country) or primary_country}**  |  "
            f"Window: **{start_str} → {end_str}**"
        )

        # run + show
        if st.button("Run MATCH_RECOGNIZE", key=f"{KEY_PREFIX}_mr_run_btn"):
            with st.spinner("Querying…"):
                if pattern == "Rising streaks (cases)":
                    raw_df = mr_rising_streaks(primary_country, start_str, end_str, int(min_len))
                    cols = ["MR_COUNTRY","MR_START_DAY","MR_END_DAY","MR_STREAK_LEN","MR_START_CASES","MR_END_CASES"]
                    rename = {
                        "MR_COUNTRY":"Country","MR_START_DAY":"Start Day","MR_END_DAY":"End Day",
                        "MR_STREAK_LEN":"Streak Length (days)","MR_START_CASES":"Cases (start)","MR_END_CASES":"Cases (end)"
                    }
                    title = "Rising streaks of new cases"
                elif pattern == "Spike day (cases)":
                    raw_df = mr_spike_days(primary_country, start_str, end_str, float(spike_mult))
                    cols = ["MR_COUNTRY","MR_SPIKE_DAY","MR_PREV_CASES","MR_SPIKE_CASES","MR_PCT_JUMP"]
                    rename = {
                        "MR_COUNTRY":"Country","MR_SPIKE_DAY":"Spike Day","MR_PREV_CASES":"Cases (prev day)",
                        "MR_SPIKE_CASES":"Cases (spike day)","MR_PCT_JUMP":"Jump (%)"
                    }
                    title = f"Spike days (≥ {spike_mult}× previous day)"
                else:
                    raw_df = mr_vax_surge(primary_country, start_str, end_str, int(vax_min_len))
                    cols = ["MR_COUNTRY","MR_START_DAY","MR_END_DAY","MR_SURGE_LEN","MR_TOTAL_NEW_SHOTS","MR_START_SHOTS","MR_END_SHOTS"]
                    rename = {
                        "MR_COUNTRY":"Country","MR_START_DAY":"Start Day","MR_END_DAY":"End Day",
                        "MR_SURGE_LEN":"Surge Length (days)","MR_TOTAL_NEW_SHOTS":"Total Doses (surge)",
                        "MR_START_SHOTS":"Doses (start)","MR_END_SHOTS":"Doses (end)"
                    }
                    title = f"Vaccination surges (≥ {vax_min_len} up-days)"

            # table
            if raw_df is None or raw_df.empty:
                st.warning("No matches found for the chosen configuration.")
            else:
                st.subheader(title)
                present = [c for c in cols if c in raw_df.columns]
                df = raw_df[present].rename(columns=rename)
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"match_recognize_{to_canonical(primary_country) or primary_country}_{pattern.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    key=f"{KEY_PREFIX}_mr_dl_btn",
                )

            # overlay chart (always render)
            mr_fig = render_match_recognition_chart(
                pattern=pattern,
                country=primary_country,
                start_date=start_str,
                end_date=end_str,
                min_len=int(min_len) if pattern == "Rising streaks (cases)" else None,
                spike_mult=float(spike_mult) if pattern == "Spike day (cases)" else None,
                vax_min_len=int(vax_min_len) if pattern == "Vaccination surge" else None,
            )
            st.plotly_chart(mr_fig, use_container_width=True, theme=None)

# Forecast
elif section == "Forecast" and is_forecastable(metric) and len(country_selection) == 1:
    st.markdown("#### Forecast")
    c1, c2 = st.columns([1, 1])
    with c1:
        horizon = st.number_input(
            "Horizon (days)",
            min_value=7, max_value=90, value=14, step=1,
            key=f"{KEY_PREFIX}_forecast_horizon",
        )
    with c2:
        st.caption("Model: ETS (weekly seasonality, damped trend)")

    if st.button("Run forecast", key=f"{KEY_PREFIX}_run_fc_btn"):
        with st.spinner("Fitting model…"):
            fc_fig = render_forecast_chart(primary_country, metric, start_str, end_str, horizon=int(horizon))
        st.plotly_chart(fc_fig, use_container_width=True, theme=None)

# Clustering
elif section == "Clustering":
    st.markdown("#### Clustering (regions)")

    cl_metric_label = st.selectbox(
        "Metric for clustering",
        non_gdp_metric_labels(),
        key=f"{KEY_PREFIX}_cl_metric_label",
    )
    cl_metric = dict(base_metric_options())[cl_metric_label]

    c1, c2 = st.columns([1, 1])
    with c1:
        window_days = st.number_input(
            "Look-back window (days)",
            min_value=30, max_value=400, value=180, step=10,
            key=f"{KEY_PREFIX}_window_days",
        )
    with c2:
        # push the checkbox a bit lower in the column
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        normalize_switch = st.checkbox(
            "Standardize features",
            value=True,
            key=f"{KEY_PREFIX}_normalize_switch",
            help="Scale each feature to zero mean and unit variance: z = (x−μ)/σ. "
                 "Helps K-Means/PCA when features have different scales."
        )

    if st.button("Run clustering", key=f"{KEY_PREFIX}_run_clusters_btn"):
        if not country_selection:
            st.warning("No countries selected. Pick at least one in the Filters section.")
        else:
            res = run_clustering(
                country_selection=country_selection,
                metric_label=cl_metric_label,
                metric_value=cl_metric,
                window_days=int(window_days),
                end_dt=pd.to_datetime(end_str),
                min_start=str(default_start),
                standardize=bool(normalize_switch),
            )
            for w in res["warnings"]:
                st.warning(w)
            with st.expander("Why some countries were skipped? (diagnostics)", expanded=False):
                st.dataframe(res["diag_df"], use_container_width=True)
            if res["result_df"].empty:
                st.warning("No usable clustering result.")
            else:
                sil = res["silhouette"]
                st.caption(f"Silhouette score: {sil:.3f}" if sil == sil else "Silhouette score: n/a")
                st.dataframe(res["result_df"], use_container_width=True)
                st.plotly_chart(res["pca_fig"], use_container_width=True, theme=None)

# Comments
elif section == "Comments":
    st.session_state.setdefault("comment_seed", 0)
    st.markdown("### Add annotation")

    # Who to apply the comment to?
    tgt_label_all = f"All selected ({len(country_selection)})"
    tgt_mode = st.radio(
        "Apply comment to",
        ["Primary only", "Specific country", tgt_label_all],
        horizontal=True,
        key=f"{KEY_PREFIX}_tgt_mode",
    )

    if tgt_mode == "Specific country":
        specific_options = country_selection or all_countries
        specific_country = st.selectbox(
            "Country", options=specific_options, key=f"{KEY_PREFIX}_tgt_specific_country"
        )
        targets = [specific_country] if specific_country else []
    elif tgt_mode == tgt_label_all:
        targets = list(country_selection)
    else:
        targets = [primary_country] if primary_country else []

    # Inputs where we use the chart/filter end date for the per-day comment
    seed = st.session_state.comment_seed
    user_key = f"{KEY_PREFIX}_comment_user_{seed}"
    text_key = f"{KEY_PREFIX}_comment_text_{seed}"
    btn_key = f"{KEY_PREFIX}_comment_submit_{seed}"

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 3, 0.6])
        with c1:
            st.text_input("Your name", key=user_key)
        with c2:
            st.text_input(
                "Comment",
                key=text_key,
                help="Date is auto-set to the current chart’s end date.",
            )
        with c3:
            submit = st.button("Submit", key=btn_key)

    if submit:
        name = (st.session_state.get(user_key) or "").strip()
        text = (st.session_state.get(text_key) or "").strip()

        if not (name and text):
            st.warning("Please provide your name and a comment.")
        elif not targets:
            st.warning("Select at least one country.")
        else:
            # choose the day to attach the comment to (end of the current window)
            end_for_comment = clamp_date((st.session_state.get("chart_params", {}).get("end")) or end_str)
            metric_for_comment = st.session_state.get("metric_value")

            successes, failures = 0, []
            for ctry in targets:
                canon = to_canonical(ctry or "")
                code, msg = add_comment(canon, end_for_comment, metric_for_comment, name, text)
                if code == 200:
                    successes += 1
                else:
                    failures.append(f"{ctry}: {code} {msg}")

            if successes:
                st.toast(
                    f"Comment added to {successes} countr{'y' if successes == 1 else 'ies'} ✅",
                    icon="✅",
                )
                st.session_state.comment_seed += 1
                st.rerun()
            if failures:
                st.error("Some saves failed:\n" + "\n".join(failures))

    # Show comments per selected country (for the current metric)
    metric_for_list = st.session_state.get("metric_value")
    if country_selection:
        for c in country_selection:
            canon = to_canonical(c or "")
            ok, items = fetch_comments(canon, metric_for_list)
            with st.expander(f"Comments — {canon or c}", expanded=False):
                if not items:
                    st.caption("No comments yet.")
                else:
                    for line in items:
                        st.write(line if ok else f"⚠️ {line}")
    else:
        st.info("Select at least one country to see comments.")
