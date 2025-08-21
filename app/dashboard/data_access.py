from __future__ import annotations

from typing import Dict, Set, Tuple

import pandas as pd
import streamlit as st

from app.core.snowflake_conn import have_sf_config, get_sf_conn, db_schema


# Basic lookups
@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def mart_global_date_span() -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return the overall [min_date, max_date] present in MART_COUNTRY_DAY.
    Falls back to a fixed span if Snowflake isn't configured.
    """
    if not have_sf_config():
        return pd.to_datetime("2020-01-01"), pd.to_datetime("2023-03-09")

    db, sch = db_schema()
    conn = get_sf_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT MIN(CAST(D AS DATE)) AS mn, MAX(CAST(D AS DATE)) AS mx
                FROM {db}.{sch}.MART_COUNTRY_DAY
                """
            )
            mn, mx = cur.fetchone()
            return pd.to_datetime(mn), pd.to_datetime(mx)
    finally:
        conn.close()


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def load_alias_map() -> Dict[str, str]:
    """
    Map UPPER(alias) -> canonical country name from COUNTRY_ALIAS.
    Empty when Snowflake isn't configured.
    """
    if not have_sf_config():
        return {}

    db, sch = db_schema()
    conn = get_sf_conn()
    try:
        with conn.cursor() as cur:
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
        conn.close()


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def load_allowed_gdp_canonical() -> Set[str]:
    """
    Set of canonical country names allowed for GDP metrics.
    Uses COUNTRY_ALIAS to normalize names where possible.
    """
    if not have_sf_config():
        return set()

    db, sch = db_schema()
    conn = get_sf_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT COALESCE(a.canonical, g.country) AS country_norm
                FROM {db}.{sch}.GDP_PPP_LONG g
                LEFT JOIN {db}.{sch}.COUNTRY_ALIAS a
                  ON UPPER(a.alias) = UPPER(g.country)
                WHERE g.country IS NOT NULL
                """
            )
            return {r[0] for r in cur.fetchall() if r[0]}
    finally:
        conn.close()


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def load_countries_from_mart() -> list[str]:
    """
    Distinct canonical country names present in MART_COUNTRY_DAY.
    Returns a small fallback list if Snowflake isn't configured.
    """
    fallback = ["Lithuania", "Latvia", "Estonia", "Poland", "Germany"]
    if not have_sf_config():
        return fallback

    db, sch = db_schema()
    conn = get_sf_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT COALESCE(a.canonical, m.country) AS country_norm
                FROM {db}.{sch}.MART_COUNTRY_DAY m
                LEFT JOIN {db}.{sch}.COUNTRY_ALIAS a
                  ON UPPER(a.alias) = UPPER(m.country)
                WHERE m.country IS NOT NULL
                ORDER BY country_norm
                """
            )
            rows = [r[0] for r in cur.fetchall() if r[0]]
            return rows or fallback
    finally:
        conn.close()


# MATCH_RECOGNIZE helpers
def _df_from_cursor(cur) -> pd.DataFrame:
    """Turn the current cursor result into a DataFrame."""
    cols = [c[0] for c in (cur.description or [])]
    rows = cur.fetchall() if cur.description else []
    return pd.DataFrame(rows, columns=cols)


@st.cache_data(ttl=30 * 60, show_spinner=False)
def mr_rising_streaks(country: str, start: str, end: str, min_len: int) -> pd.DataFrame:
    """
    Find N+ day rising streaks of NEW_CASES for a single country.
    Returns a DataFrame with MR_* columns.
    """
    if not have_sf_config():
        return pd.DataFrame()

    db, sch = db_schema()
    sql = f"""
    WITH prep AS (
      SELECT
        COUNTRY, D, NEW_CASES,
        LAG(NEW_CASES) OVER (PARTITION BY COUNTRY ORDER BY D) AS prev_cases
      FROM {db}.{sch}.MART_COUNTRY_DAY
      WHERE COUNTRY = %s
        AND D BETWEEN %s AND %s
    ),
    flags AS (
      SELECT
        COUNTRY, D, NEW_CASES,
        CASE WHEN prev_cases IS NOT NULL AND NEW_CASES > prev_cases THEN 1 ELSE 0 END AS is_up_cases
      FROM prep
    )
    SELECT *
    FROM flags
    MATCH_RECOGNIZE (
      PARTITION BY COUNTRY
      ORDER BY D
      MEASURES
        FIRST(COUNTRY)   AS MR_COUNTRY,
        FIRST(D)         AS MR_START_DAY,
        LAST(D)          AS MR_END_DAY,
        COUNT(*)         AS MR_STREAK_LEN,
        FIRST(NEW_CASES) AS MR_START_CASES,
        LAST(NEW_CASES)  AS MR_END_CASES
      ONE ROW PER MATCH
      PATTERN (U{{{min_len},}})
      DEFINE
        U AS is_up_cases = 1
    )
    ORDER BY MR_START_DAY;
    """
    conn = get_sf_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (country, start, end))
            return _df_from_cursor(cur)
    finally:
        conn.close()


@st.cache_data(ttl=30 * 60, show_spinner=False)
def mr_spike_days(country: str, start: str, end: str, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Find spike days where NEW_CASES >= multiplier × previous day.
    Returns a DataFrame with MR_* columns.
    """
    if not have_sf_config():
        return pd.DataFrame()

    db, sch = db_schema()
    sql = f"""
    WITH prep AS (
      SELECT
        COUNTRY, D, NEW_CASES,
        LAG(NEW_CASES) OVER (PARTITION BY COUNTRY ORDER BY D) AS prev_cases
      FROM {db}.{sch}.MART_COUNTRY_DAY
      WHERE COUNTRY = %s
        AND D BETWEEN %s AND %s
    ),
    flags AS (
      SELECT
        COUNTRY, D, NEW_CASES, prev_cases,
        CASE WHEN prev_cases IS NOT NULL AND prev_cases > 0
             AND NEW_CASES >= %s * prev_cases
             THEN 1 ELSE 0 END AS is_spike
      FROM prep
    )
    SELECT *
    FROM flags
    MATCH_RECOGNIZE (
      PARTITION BY COUNTRY
      ORDER BY D
      MEASURES
        LAST(COUNTRY)    AS MR_COUNTRY,
        LAST(D)          AS MR_SPIKE_DAY,
        LAST(prev_cases) AS MR_PREV_CASES,
        LAST(NEW_CASES)  AS MR_SPIKE_CASES,
        ROUND( (LAST(NEW_CASES) / NULLIF(LAST(prev_cases),0) - 1) * 100, 1) AS MR_PCT_JUMP
      ONE ROW PER MATCH
      PATTERN (S)
      DEFINE
        S AS is_spike = 1
    )
    ORDER BY MR_SPIKE_DAY;
    """
    conn = get_sf_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (country, start, end, float(multiplier)))
            return _df_from_cursor(cur)
    finally:
        conn.close()


@st.cache_data(ttl=30 * 60, show_spinner=False)
def mr_vax_surge(country: str, start: str, end: str, min_len: int = 5) -> pd.DataFrame:
    """
    Find vaccination surges: ≥ min_len consecutive up-days in DAILY_VACCINATIONS.
    Returns a DataFrame with MR_* columns.
    """
    if not have_sf_config():
        return pd.DataFrame()

    db, sch = db_schema()
    sql = f"""
    WITH prep AS (
      SELECT
        COUNTRY, D, DAILY_VACCINATIONS,
        LAG(DAILY_VACCINATIONS) OVER (PARTITION BY COUNTRY ORDER BY D) AS prev_vax
      FROM {db}.{sch}.MART_COUNTRY_DAY
      WHERE COUNTRY = %s
        AND D BETWEEN %s AND %s
    ),
    flags AS (
      SELECT
        COUNTRY, D, DAILY_VACCINATIONS,
        CASE WHEN prev_vax IS NOT NULL AND DAILY_VACCINATIONS > prev_vax THEN 1 ELSE 0 END AS is_up_vax
      FROM prep
    )
    SELECT *
    FROM flags
    MATCH_RECOGNIZE (
      PARTITION BY COUNTRY
      ORDER BY D
      MEASURES
        FIRST(COUNTRY)            AS MR_COUNTRY,
        FIRST(D)                  AS MR_START_DAY,
        LAST(D)                   AS MR_END_DAY,
        COUNT(*)                  AS MR_SURGE_LEN,
        SUM(DAILY_VACCINATIONS)   AS MR_TOTAL_NEW_SHOTS,
        FIRST(DAILY_VACCINATIONS) AS MR_START_SHOTS,
        LAST(DAILY_VACCINATIONS)  AS MR_END_SHOTS
      ONE ROW PER MATCH
      PATTERN (V{{{min_len},}})
      DEFINE
        V AS is_up_vax = 1
    )
    ORDER BY MR_START_DAY;
    """
    conn = get_sf_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (country, start, end))
            return _df_from_cursor(cur)
    finally:
        conn.close()
