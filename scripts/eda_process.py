# scripts/eda_task2.py
import os
import pandas as pd
from dotenv import load_dotenv
import snowflake.connector
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

load_dotenv()

conn = snowflake.connector.connect(
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    role=os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    database="COVID_DB", schema="PUBLIC"
)

def q(sql):
    cur = conn.cursor()
    cur.execute(sql)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=cols)

df = q("""
SELECT country, d, new_cases, new_deaths, cases_7dma, deaths_7dma,
       population, new_cases_per_100k, new_deaths_per_100k,
       total_vaccinations, people_vaccinated, people_fully_vaccinated,
       daily_vaccinations, total_vaccinations_per_hundred,
       people_vaccinated_per_hundred, people_fully_vaccinated_per_hundred
FROM COVID_DB.PUBLIC.MART_COUNTRY_DAY
""")

for col in df.columns:
    if "date" in col.lower() or col.lower() in ["d"]:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass

df = df.sort_values(["COUNTRY", "D"])

os.makedirs("reports", exist_ok=True)
df.to_csv("reports/mart_country_day_full.csv", index=False)

prof = ProfileReport(
    df.sample(min(len(df), 150000), random_state=42),
    title="EDA: MART_COUNTRY_DAY",
    explorative=True
)
prof.to_file("reports/eda_mart_country_day.html")

latest = df.groupby("COUNTRY")["D"].max().reset_index()
latest_df = latest.merge(df, on=["COUNTRY", "D"], how="left")
top_cases = latest_df.sort_values("CASES_7DMA", ascending=False).head(15)
top_cases[["COUNTRY", "D", "CASES_7DMA", "NEW_CASES_PER_100K", "PEOPLE_FULLY_VACCINATED_PER_HUNDRED"]]\
    .to_csv("reports/top15_latest_cases7dma.csv", index=False)


for ctry in ["Lithuania", "Germany", "Poland", "Latvia", "Ukraine", "Estonia"]:
    sub = df[df["COUNTRY"] == ctry]
    if sub.empty:
        continue

  
    plt.figure()
    plt.plot(sub["D"], sub["NEW_CASES_PER_100K"])
    plt.title(f"{ctry} - New cases per 100k")
    plt.xlabel("Date")
    plt.ylabel("New cases per 100k")

    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"reports/{ctry.lower()}_cases_per_100k.png")
    plt.close()

    plt.figure()
    plt.plot(sub["D"], sub["NEW_CASES_PER_100K"])
    plt.title(f"{ctry} - New deaths per 100k")
    plt.xlabel("Date")
    plt.ylabel("New deaths per 100k")

    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"reports/{ctry.lower()}_deaths_per_100k.png")
    plt.close()

print("OK: wrote reports/*.html, *.csv, and PNG charts")
