from typing import List, Dict
import snowflake.connector
from .config import SNOWFLAKE

# Direct columns on MART_COUNTRY_DAY
METRIC_COLUMNS = {
    "NEW_CASES": "NEW_CASES",
    "NEW_DEATHS": "NEW_DEATHS",
    "CASES_7DMA": "CASES_7DMA",
    "DEATHS_7DMA": "DEATHS_7DMA",
    "NEW_CASES_PER_100K": "NEW_CASES_PER_100K",
    "NEW_DEATHS_PER_100K": "NEW_DEATHS_PER_100K",
    "TOTAL_VACCINATIONS": "TOTAL_VACCINATIONS",
    "PEOPLE_VACCINATED": "PEOPLE_VACCINATED",
    "PEOPLE_FULLY_VACCINATED": "PEOPLE_FULLY_VACCINATED",
    "DAILY_VACCINATIONS": "DAILY_VACCINATIONS",
    "TOTAL_VACCINATIONS_PER_HUNDRED": "TOTAL_VACCINATIONS_PER_HUNDRED",
    "PEOPLE_VACCINATED_PER_HUNDRED": "PEOPLE_VACCINATED_PER_HUNDRED",
    "PEOPLE_FULLY_VACCINATED_PER_HUNDRED": "PEOPLE_FULLY_VACCINATED_PER_HUNDRED",
}

# Metrics that require custom joins/queries
GDP_METRICS = {
    "GDP_PPP_PER_CAPITA",
    "GDP_VS_CASES_PER100K_YEAR",
}

def _connect():
    return snowflake.connector.connect(**SNOWFLAKE)

# -------------------------------------------------------------------
# GDP PPP per Capita
# -------------------------------------------------------------------
def _fetch_gdp_ppp_per_capita(country: str, start: str, end: str) -> List[Dict]:
    db = SNOWFLAKE["database"]
    sch = SNOWFLAKE["schema"]
    sql = f"""
        WITH m AS (
            SELECT COUNTRY, CAST(D AS DATE) AS DATE, YEAR(D) AS Y
            FROM {db}.{sch}.MART_COUNTRY_DAY
            WHERE UPPER(COUNTRY) = UPPER(%s)
              AND DATE BETWEEN %s AND %s
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
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(sql, (country, start, end))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()

# -------------------------------------------------------------------
# GDP vs Cases per 100k (Year)
# -------------------------------------------------------------------
def _fetch_gdp_vs_cases_per100k_year(country: str, start: str, end: str) -> List[Dict]:
    db = SNOWFLAKE["database"]
    sch = SNOWFLAKE["schema"]
    sql = f"""
                    WITH m AS (
                SELECT COUNTRY, CAST(D AS DATE) AS DATE, YEAR(D) AS Y,
                    NEW_CASES_PER_100K
                FROM {db}.{sch}.MART_COUNTRY_DAY
                WHERE UPPER(COUNTRY) = UPPER(%s)
                AND DATE BETWEEN %s AND %s
            ),
            g AS (
                SELECT COUNTRY, YEAR, GDP_PPP_PER_CAPITA
                FROM {db}.{sch}.GDP_PPP_LONG
            ),
            joined AS (
                SELECT m.COUNTRY,
                    m.DATE,
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
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(sql, (country, start, end))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()

# -------------------------------------------------------------------
# Main Fetch Function
# -------------------------------------------------------------------
def fetch_metric_series(country: str, metric: str, start: str, end: str) -> List[Dict]:
    metric_u = metric.upper()

    # GDP branches
    if metric_u == "GDP_PPP_PER_CAPITA":
        return _fetch_gdp_ppp_per_capita(country, start, end)
    if metric_u == "GDP_VS_CASES_PER100K_YEAR":
        return _fetch_gdp_vs_cases_per100k_year(country, start, end)
    
    # Standard MART columns
    col = METRIC_COLUMNS.get(metric_u)
    if not col:
        allowed = list(METRIC_COLUMNS.keys()) + list(GDP_METRICS)
        raise ValueError(f"Unknown metric '{metric}'. Allowed: {allowed}")

    db = SNOWFLAKE["database"]
    sch = SNOWFLAKE["schema"]
    sql = f"""
        SELECT COUNTRY, CAST(D AS DATE) AS DATE, '{metric_u}' AS METRIC, {col} AS VALUE
        FROM {db}.{sch}.MART_COUNTRY_DAY
        WHERE UPPER(COUNTRY) = UPPER(%s)
          AND DATE BETWEEN %s AND %s
        ORDER BY DATE;
    """
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(sql, (country, start, end))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()


