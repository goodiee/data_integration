from typing import Any

from app.core.constants import METRIC_COLUMNS, GDP_METRICS
from app.core.snowflake_conn import get_sf_conn, db_schema


def _fetchall_dict(cur) -> list[dict[str, Any]]:
    """Return all rows from current cursor as list[dict]."""
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _run_query(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Open a connection, run a parameterized query, return rows as dicts."""
    con = get_sf_conn()
    try:
        with con.cursor() as cur:
            cur.execute(sql, params)
            return _fetchall_dict(cur)
    finally:
        con.close()


def _fetch_gdp_ppp_per_capita(country: str, start: str, end: str) -> list[dict[str, Any]]:
    """For each day, pick the latest available GDP PPP per capita up to that year."""
    db, sch = db_schema()
    sql = f"""
        WITH m AS (
            SELECT COUNTRY, CAST(D AS DATE) AS DATE, YEAR(D) AS Y
            FROM {db}.{sch}.MART_COUNTRY_DAY
            WHERE UPPER(COUNTRY) = UPPER(%s)
              AND CAST(D AS DATE) BETWEEN %s AND %s
        )
        SELECT m.COUNTRY,
               m.DATE,
               'GDP_PPP_PER_CAPITA' AS METRIC,
               g.GDP_PPP_PER_CAPITA AS VALUE
        FROM m
        LEFT JOIN {db}.{sch}.GDP_PPP_LONG g
               ON UPPER(g.COUNTRY) = UPPER(m.COUNTRY)
              AND g.YEAR <= m.Y
        QUALIFY ROW_NUMBER()
               OVER (PARTITION BY m.COUNTRY, m.DATE ORDER BY g.YEAR DESC) = 1
        ORDER BY m.DATE;
    """
    return _run_query(sql, (country, start, end))


def _fetch_gdp_vs_cases_per100k_year(country: str, start: str, end: str) -> list[dict[str, Any]]:
    """
    Return two series (stacked together):
      - GDP PPP per capita (latest known for each day’s year)
      - New cases per 100k (daily)
    """
    db, sch = db_schema()
    sql = f"""
        WITH m AS (
            SELECT COUNTRY, CAST(D AS DATE) AS DATE, YEAR(D) AS Y, NEW_CASES_PER_100K
            FROM {db}.{sch}.MART_COUNTRY_DAY
            WHERE UPPER(COUNTRY) = UPPER(%s)
              AND CAST(D AS DATE) BETWEEN %s AND %s
        ),
        g AS (
            SELECT COUNTRY, YEAR, GDP_PPP_PER_CAPITA
            FROM {db}.{sch}.GDP_PPP_LONG
        ),
        joined AS (
            SELECT m.COUNTRY, m.DATE,
                   m.NEW_CASES_PER_100K AS CASES_PER_100K,
                   g.GDP_PPP_PER_CAPITA AS GDP
            FROM m
            LEFT JOIN g
                   ON UPPER(g.COUNTRY) = UPPER(m.COUNTRY)
                  AND g.YEAR <= m.Y
            QUALIFY ROW_NUMBER()
                   OVER (PARTITION BY m.COUNTRY, m.DATE ORDER BY g.YEAR DESC) = 1
        )
        SELECT COUNTRY, DATE, 'GDP_PPP_PER_CAPITA' AS METRIC, GDP AS VALUE
          FROM joined
         WHERE GDP IS NOT NULL
        UNION ALL
        SELECT COUNTRY, DATE, 'NEW_CASES_PER_100K' AS METRIC, CASES_PER_100K AS VALUE
          FROM joined
         WHERE GDP IS NOT NULL
        ORDER BY DATE, METRIC;
    """
    return _run_query(sql, (country, start, end))


def fetch_metric_series(country: str, metric: str, start: str, end: str) -> list[dict[str, Any]]:
    """
    Generic metric fetcher for MART_COUNTRY_DAY. Special-cases some composed metrics.
    - All inputs are strings; dates must be 'YYYY-MM-DD'.
    """
    metric_u = metric.upper()

    # Special cases that require joins/QUALIFY
    if metric_u == "GDP_PPP_PER_CAPITA":
        return _fetch_gdp_ppp_per_capita(country, start, end)
    if metric_u == "GDP_VS_CASES_PER100K_YEAR":
        return _fetch_gdp_vs_cases_per100k_year(country, start, end)

    # Simple column lookups from MART_COUNTRY_DAY
    col = METRIC_COLUMNS.get(metric_u)
    if not col:
        allowed = sorted(set(METRIC_COLUMNS.keys()) | set(GDP_METRICS))
        raise ValueError(f"Unknown metric '{metric}'. Allowed: {allowed}")

    db, sch = db_schema()
    sql = f"""
        SELECT COUNTRY,
               CAST(D AS DATE) AS DATE,
               '{metric_u}' AS METRIC,
               {col} AS VALUE
          FROM {db}.{sch}.MART_COUNTRY_DAY
         WHERE UPPER(COUNTRY) = UPPER(%s)
           AND CAST(D AS DATE) BETWEEN %s AND %s
         ORDER BY DATE;
    """
    return _run_query(sql, (country, start, end))
