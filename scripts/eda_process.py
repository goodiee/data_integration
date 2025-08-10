import os
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from ydata_profiling import ProfileReport


load_dotenv()

COUNTRIES = ["Lithuania", "Germany", "Poland", "Latvia", "Ukraine", "Estonia"]
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
INSIGHTS_DIR = REPORTS_DIR / "gdp_insights"
GDP_CSV = DATA_DIR / "gdp_ppp_long.csv"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def sf_connect():
    """Open a Snowflake connection from env vars."""
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        role=os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database="COVID_DB",
        schema="PUBLIC",
    )


def q(conn, sql: str) -> pd.DataFrame:
    """Run a query and return a DataFrame with column names."""
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [c[0] for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def load_and_upload_gdp(conn, csv_path: Path) -> int:
    """Load GDP CSV, normalize columns, and upload to Snowflake."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find GDP CSV at {csv_path.resolve()}")

    gdp = pd.read_csv(csv_path)
    gdp.columns = gdp.columns.str.strip()

    for s in ("Sr.No", "Sr. No", "S.No", "S No"):
        if s in gdp.columns:
            gdp = gdp.drop(columns=[s])
            break

    gdp = gdp.rename(columns={
        "Country": "COUNTRY",
        "Year": "YEAR",
        "GDP_PPP_Per_Capita": "GDP_PPP_PER_CAPITA",
    })

    gdp["YEAR"] = pd.to_numeric(gdp["YEAR"], errors="coerce").astype("Int64")
    gdp["GDP_PPP_PER_CAPITA"] = pd.to_numeric(gdp["GDP_PPP_PER_CAPITA"], errors="coerce")
    gdp = gdp.dropna(subset=["YEAR", "GDP_PPP_PER_CAPITA"]).reset_index(drop=True)

    with conn.cursor() as cur:
        cur.execute("""
            CREATE OR REPLACE TABLE GDP_PPP_LONG (
                COUNTRY STRING,
                YEAR INT,
                GDP_PPP_PER_CAPITA FLOAT
            )
        """)

    write_pandas(conn, gdp, "GDP_PPP_LONG")
    return len(gdp)


def load_covid_gdp_join(conn) -> pd.DataFrame:
    """
    Join MART_COUNTRY_DAY with GDP using COUNTRY_ALIAS to normalize names on both sides.
    Carries the last-known GDP <= date for each country-day.
    """
    df = q(conn, """
        WITH m_norm AS (
            SELECT
                COALESCE(a.canonical, m.country) AS country_norm,
                m.*
            FROM COVID_DB.PUBLIC.MART_COUNTRY_DAY m
            LEFT JOIN COVID_DB.PUBLIC.COUNTRY_ALIAS a
              ON UPPER(a.alias) = UPPER(m.country)
        ),
        g_norm AS (
            SELECT
                COALESCE(a.canonical, g.country) AS country_norm,
                g.*
            FROM GDP_PPP_LONG g
            LEFT JOIN COVID_DB.PUBLIC.COUNTRY_ALIAS a
              ON UPPER(a.alias) = UPPER(g.country)
        ),
        joined AS (
            SELECT
              m.country_norm            AS country,
              m.d,
              m.new_cases, m.new_deaths, m.cases_7dma, m.deaths_7dma,
              m.population, m.new_cases_per_100k, m.new_deaths_per_100k,
              m.total_vaccinations, m.people_vaccinated, m.people_fully_vaccinated,
              m.daily_vaccinations, m.total_vaccinations_per_hundred,
              m.people_vaccinated_per_hundred, m.people_fully_vaccinated_per_hundred,
              g.gdp_ppp_per_capita,
              ROW_NUMBER() OVER (
                  PARTITION BY m.country_norm, m.d
                  ORDER BY g.year DESC
              ) AS rn
            FROM m_norm m
            LEFT JOIN g_norm g
              ON UPPER(g.country_norm) = UPPER(m.country_norm)
             AND g.year <= YEAR(m.d)
        )
        SELECT
          country, d, new_cases, new_deaths, cases_7dma, deaths_7dma,
          population, new_cases_per_100k, new_deaths_per_100k,
          total_vaccinations, people_vaccinated, people_fully_vaccinated,
          daily_vaccinations, total_vaccinations_per_hundred,
          people_vaccinated_per_hundred, people_fully_vaccinated_per_hundred,
          gdp_ppp_per_capita
        FROM joined
        QUALIFY rn = 1
    """)

    df.columns = [c.upper() for c in df.columns]

    if "D" in df.columns:
        df["D"] = pd.to_datetime(df["D"], errors="coerce")

    df = df.sort_values(["COUNTRY", "D"]).reset_index(drop=True)
    return df



def compute_gdp_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute GDP tiers (Low/Mid/High) from max GDP PPP per capita per country.
    Returns a DataFrame with columns: COUNTRY, MAX_GDP_PPP_PER_CAPITA, GDP_TIER
    """
    latest = (
        df.groupby("COUNTRY", as_index=False)["GDP_PPP_PER_CAPITA"]
          .max()
          .rename(columns={"GDP_PPP_PER_CAPITA": "MAX_GDP_PPP_PER_CAPITA"})
    )

    try:
        latest["GDP_TIER"] = pd.qcut(
            latest["MAX_GDP_PPP_PER_CAPITA"],
            3, labels=["Low GDP", "Mid GDP", "High GDP"]
        )
    except ValueError:
        latest["GDP_TIER"] = pd.cut(
            latest["MAX_GDP_PPP_PER_CAPITA"],
            3, labels=["Low GDP", "Mid GDP", "High GDP"]
        )
    return latest


def plot_series_by_country(df: pd.DataFrame, country: str, col: str, title_suffix: str, out_file: Path):
    sub = df[df["COUNTRY"] == country]
    if sub.empty or col not in sub.columns:
        return

    plt.figure()
    plt.plot(sub["D"], sub[col])
    plt.title(f"{country} - {title_suffix}")
    plt.xlabel("Date")
    plt.ylabel(title_suffix)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def scatter_gdp_vs(df: pd.DataFrame, y_col: str, y_label: str, title: str, out_file: Path):
    if y_col not in df.columns:
        return

    snap = (
        df.groupby("COUNTRY", as_index=False)["D"].max()
          .merge(df, on=["COUNTRY", "D"], how="left")
    )
    snap = snap.dropna(subset=["GDP_PPP_PER_CAPITA", y_col])

    plt.figure()
    plt.scatter(snap["GDP_PPP_PER_CAPITA"], snap[y_col], alpha=0.6)
    plt.xlabel("GDP PPP per Capita (USD)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def scatter_gdp_vs_by_year(df: pd.DataFrame, y_col: str, y_label: str, title: str, out_file: Path):
    """Static scatter plot, colored by calendar year (from D)."""
    if "D" not in df.columns or y_col not in df.columns:
        return
    tmp = df.dropna(subset=["GDP_PPP_PER_CAPITA", y_col, "D"]).copy()
    tmp["YEAR"] = tmp["D"].dt.year

    snap = (
        tmp.sort_values(["COUNTRY", "YEAR", "D"])
           .groupby(["COUNTRY", "YEAR"], as_index=False)
           .tail(1)
    )

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        snap["GDP_PPP_PER_CAPITA"],
        snap[y_col],
        c=snap["YEAR"],
        alpha=0.7
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Year")
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    plt.xlabel("GDP PPP per Capita (USD)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def time_series_by_gdp_tier(df: pd.DataFrame, y_col: str, y_label: str, title: str, out_file: Path):
    """Average time series grouped into GDP tiers (quantiles); falls back if too few unique values."""
    if y_col not in df.columns or "GDP_PPP_PER_CAPITA" not in df.columns:
        return

    latest_gdp = compute_gdp_tiers(df)  
    df_tiered = df.merge(latest_gdp[["COUNTRY", "GDP_TIER"]], on="COUNTRY", how="left")
    avg_ts = df_tiered.dropna(subset=[y_col]).groupby(["D", "GDP_TIER"], as_index=False)[y_col].mean()

    plt.figure(figsize=(10, 6))
    for tier, sub in avg_ts.groupby("GDP_TIER"):
        plt.plot(sub["D"], sub[y_col], label=str(tier))
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def main():
    conn = sf_connect()
    try:
        uploaded = load_and_upload_gdp(conn, GDP_CSV)
        print(f"Uploaded GDP rows: {uploaded}")

        df = load_covid_gdp_join(conn)

        out_csv = REPORTS_DIR / "mart_country_day_with_gdp.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved joined CSV: {out_csv}")

        tiers = compute_gdp_tiers(df)
        tiers_csv = INSIGHTS_DIR / "gdp_tiers.csv"
        tiers.sort_values(["GDP_TIER", "MAX_GDP_PPP_PER_CAPITA"]).to_csv(tiers_csv, index=False)
        print(f"Saved GDP tiers: {tiers_csv}")

        for t in ["Low GDP", "Mid GDP", "High GDP"]:
            members = tiers.loc[tiers["GDP_TIER"] == t, "COUNTRY"].sort_values().tolist()
            print(f"{t} ({len(members)}): {', '.join(members)}")

   
        df_clean = df.replace([np.inf, -np.inf], np.nan)

        try:
            sample_n = min(len(df_clean), 150_000)
            prof = ProfileReport(
                df_clean.sample(sample_n, random_state=42) if sample_n > 0 else df_clean,
                title="EDA: MART_COUNTRY_DAY + GDP",
                explorative=True,
                missing_diagrams={"heatmap": False, "dendrogram": False},
            )
            eda_path = REPORTS_DIR / "eda_mart_country_day_with_gdp.html"
            prof.to_file(str(eda_path))
            print(f"HTML report: {eda_path}")
        except Exception as e:
            print(f"[ydata-profiling] Skipped HTML report due to: {e}")

       
        for c in COUNTRIES:
            plot_series_by_country(
                df, c, "NEW_CASES_PER_100K", "New cases per 100k",
                REPORTS_DIR / f"{c.lower()}_cases_per_100k.png"
            )
            plot_series_by_country(
                df, c, "NEW_DEATHS_PER_100K", "New deaths per 100k",
                REPORTS_DIR / f"{c.lower()}_deaths_per_100k.png"
            )

        scatter_gdp_vs(
            df, "NEW_CASES_PER_100K",
            "New Cases per 100k (latest date)",
            "GDP vs COVID New Cases per 100k",
            INSIGHTS_DIR / "gdp_vs_cases_per100k.png"
        )

        scatter_gdp_vs(
            df, "NEW_DEATHS_PER_100K",
            "New Deaths per 100k (latest date)",
            "GDP vs COVID New Deaths per 100k",
            INSIGHTS_DIR / "gdp_vs_deaths_per100k.png"
        )

        scatter_gdp_vs(
            df, "PEOPLE_FULLY_VACCINATED_PER_HUNDRED",
            "Fully Vaccinated per 100 (latest date)",
            "GDP vs Vaccination Coverage (latest)",
            INSIGHTS_DIR / "gdp_vs_vax_coverage.png"
        )

        scatter_gdp_vs_by_year(
            df, "NEW_CASES_PER_100K",
            "New Cases per 100k",
            "GDP vs COVID New Cases per 100k (Year-colored)",
            INSIGHTS_DIR / "gdp_vs_cases_per100k_year_color.png"
        )

        scatter_gdp_vs_by_year(
            df, "PEOPLE_FULLY_VACCINATED_PER_HUNDRED",
            "Fully Vaccinated per 100",
            "GDP vs Vaccination Coverage (Year-colored)",
            INSIGHTS_DIR / "gdp_vs_vax_year_color.png"
        )

        time_series_by_gdp_tier(
            df, "NEW_CASES_PER_100K",
            "New Cases per 100k",
            "COVID New Cases per 100k by GDP Tier",
            INSIGHTS_DIR / "cases_by_gdp_tier.png"
        )

        time_series_by_gdp_tier(
            df, "PEOPLE_FULLY_VACCINATED_PER_HUNDRED",
            "Fully Vaccinated per 100",
            "Vaccination Coverage by GDP Tier",
            INSIGHTS_DIR / "vax_by_gdp_tier.png"
        )

        print("Done. Charts saved in:")
        print(f" - {REPORTS_DIR}")
        print(f" - {INSIGHTS_DIR}")

    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
