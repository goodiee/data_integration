import pandas as pd
import os
from dotenv import load_dotenv

from eda_process import sf_connect

load_dotenv()

gdp_file = "M:/accenture/final_project/covid19-platform/data/gdp_long.csv"
gdp = pd.read_csv(gdp_file)
gdp.columns = gdp.columns.str.strip()

if "Country" not in gdp.columns:
    raise ValueError("The column 'Country' was not found in the GDP dataset.")

gdp_countries = set(gdp["Country"].dropna().str.strip().unique())

conn = sf_connect()

with conn.cursor() as cur:
    cur.execute("SELECT DISTINCT country FROM MART_COUNTRY_DAY")
    mart_countries = set(r[0].strip() for r in cur.fetchall() if r[0])

conn.close()

missing_in_gdp = sorted(mart_countries - gdp_countries)
missing_in_mart = sorted(gdp_countries - mart_countries)

print(f"Countries in MART_COUNTRY_DAY but missing in GDP: {len(missing_in_gdp)}")
print(missing_in_gdp)

print(f"\nCountries in GDP but missing in MART_COUNTRY_DAY: {len(missing_in_mart)}")
print(missing_in_mart)
