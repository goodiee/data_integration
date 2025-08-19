# COVID-19 Dashboard (Streamlit + FastAPI + Snowflake + MongoDB)

Interactive Streamlit dashboard backed by a FastAPI service.
Time-series metrics are read from Snowflake; user comments/annotations are stored in MongoDB.

## Project layout
<pre>
.
|-- app/
|   |-- api/
|   |   |-- db_mongo.py          # Mongo client + indexes + CRUD
|   |   |-- db_snowflake.py      # Metric queries to Snowflake
|   |   |-- main.py              # FastAPI app + routes
|   |   \-- schemas.py           # Pydantic models
|   |-- core/
|   |   |-- config.py            # env vars
|   |   |-- constants.py
|   |   \-- snowflake_conn.py    # Snowflake helpers
|   \-- dashboard/
|       |-- charts.py            # charts + API calls
|       |-- clustering.py        # features + KMeans + PCA
|       |-- comments.py          # list/add comments via API
|       |-- config.py            # API_BASE, page config
|       |-- data_access.py       # small Snowflake reads (UI/MR)
|       |-- dashboard.py         # Streamlit entry
|       |-- forecasting.py       # ETS forecast (optional)
|       |-- metrics.py           # metric lists
|       |-- styles.py            # CSS
|       \-- utils.py             # helpers
|-- data/
|   |-- gdp_long.csv
|   \-- gdp_ppp_long.csv
|-- notebooks/
|   \-- conversion.ipynb         # GDP → normalized long format
|-- reports/                      # (optional) generated plots/reports
|-- scripts/
|   |-- eda_process.py           # Snowflake table flow + per-country reports
|   \-- test.py                  # detect missing/unmapped countries
|-- .env.example
|-- .gitignore
\-- requirements.txt

</pre>

## Tech stack

- Backend: FastAPI, Uvicorn, Pydantic, cachetools
- DBs: Snowflake (metrics), MongoDB (comments)
- Frontend: Streamlit, Plotly
- ML / Stats: scikit-learn (KMeans, PCA), statsmodels (ETS), SciPy
- Utilities: pandas, numpy, requests, python-dotenv
- Python: 3.11.9

## Setup 

```
git clone https://github.com/goodiee/data_integration.git
cd <YOUR_REPO>
```
### create & activate venv
```
python -m venv .venv
.venv\Scripts\activate.bat
```

### install dependencies
```
pip install -r requirements.txt
```

### create .env from example and edit it
```
copy .env.example .env
notepad .env
```


## What you can do

- 📈 Explore time-series metrics (cases, deaths, vaccinations; GDP views when available)
- 🗒️ Add comments to a specific (country, date, metric) and list them
- 🔎 Pattern recognition: rising streaks, spike days, vaccination surges
- 🧩 Clustering: K-means with PCA 2D visualization
- 🔮 Optional ETS forecast with 95% interval (for supported metrics)

## Data preparation & utilities

```
python scripts\eda_process.py
```
- Shows how Snowflake tables are used with the COVID-19 dataset (load/transform/inspect).
- Can generate per-country reports (details & plots) and save them under reports/

```
python scripts\test.py
```
- Helps detect missed or unmapped countries (e.g., alias mismatches or data gaps).
- Useful to understand which regions can’t be plotted/predicted due to missing series.

```
notebooks/conversion.ipynb
```
- Used to normalize the GDP PPP dataset into a clean long format (data/gdp_long.csv, data/gdp_ppp_long.csv).
- Makes the GDP data readable and join-friendly with the COVID time series for comparisons/insights.

 ## GDP dataset note

This project also uses an additional GDP PPP dataset to compare macroeconomic context with COVID metrics.
Some countries may be missing or have inconsistent naming; the alias map and scripts/test.py help identify and resolve those gaps.


# Run the app (two terminals)
```
uvicorn app.api.main:app --reload --port 8000
```
```
streamlit run app\dashboard\dashboard.py
```

### Open:

- Dashboard → http://localhost:8501

  <img width="1732" height="839" alt="image" src="https://github.com/user-attachments/assets/cd16f8b8-44ae-43c3-8c65-36b6494da90d" />




