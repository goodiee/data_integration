import time
import requests
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from copy import deepcopy
from app.core.constants import GDP_METRICS
from app.core.snowflake_conn import have_sf_config, get_sf_conn

API_BASE = "http://localhost:8000"

DATA_MIN = "2020-01-01"
DATA_MAX = "2023-12-31"
DEFAULT_START = DATA_MIN
DEFAULT_END = DATA_MAX

def clamp_date(date_str: str) -> str:
    if not date_str:
        return DATA_MIN
    return max(DATA_MIN, min(date_str, DATA_MAX))

# caches (auto-refreshed)
_ALIAS_MAP = {}                 # {"ALIAS_UPPER": "Canonical"}
_ALLOWED_GDP_CANONICAL = set()  # {"Canonical Name", ...}
_last_refresh = 0

def _load_alias_map(conn) -> dict:
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT UPPER(alias) AS alias_u, canonical
            FROM COVID_DB.PUBLIC.COUNTRY_ALIAS
            WHERE alias IS NOT NULL AND canonical IS NOT NULL
        """)
        return {alias_u: canonical for alias_u, canonical in cur.fetchall()}
    finally:
        cur.close()

def _load_allowed_gdp_canonical(conn) -> set:
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT DISTINCT
                   COALESCE(a.canonical, g.country) AS country_norm
            FROM COVID_DB.PUBLIC.GDP_PPP_LONG g
            LEFT JOIN COVID_DB.PUBLIC.COUNTRY_ALIAS a
              ON UPPER(a.alias) = UPPER(g.country)
            WHERE g.country IS NOT NULL
        """)
        return {r[0] for r in cur.fetchall() if r[0]}
    finally:
        cur.close()

def _bootstrap_country_gates(force=False):
    global _ALIAS_MAP, _ALLOWED_GDP_CANONICAL, _last_refresh
    if not force and (time.time() - _last_refresh) < 12 * 3600:
        return
    if not have_sf_config():
        _ALIAS_MAP = {}
        _ALLOWED_GDP_CANONICAL = set()
        _last_refresh = time.time()
        return
    conn = get_sf_conn()
    try:
        _ALIAS_MAP = _load_alias_map(conn)
        _ALLOWED_GDP_CANONICAL = _load_allowed_gdp_canonical(conn)
        _last_refresh = time.time()
    finally:
        conn.close()

def to_canonical(country: str) -> str:
    """Normalize a free-typed country to canonical using COUNTRY_ALIAS; fallback to input."""
    if not country:
        return ""
    c = country.strip()
    canon = _ALIAS_MAP.get(c.upper())
    return canon if canon else c

def gdp_allowed(country: str) -> bool:
    """Return True if country (after alias normalization) is present in GDP table."""
    _bootstrap_country_gates()
    return to_canonical(country) in _ALLOWED_GDP_CANONICAL


metric_options = [
    {"label": "🦠 Infection Rates", "value": "HEADER_INFECTION", "disabled": True},
    {"label": "New Cases", "value": "NEW_CASES"},
    {"label": "New Cases per 100k", "value": "NEW_CASES_PER_100K"},

    {"label": "💀 Mortality Rates", "value": "HEADER_MORTALITY", "disabled": True},
    {"label": "New Deaths", "value": "NEW_DEATHS"},
    {"label": "New Deaths per 100k", "value": "NEW_DEATHS_PER_100K"},

    {"label": "💉 Vaccination Rates", "value": "HEADER_VAX", "disabled": True},
    {"label": "Total Vaccinations", "value": "TOTAL_VACCINATIONS"},
    {"label": "Daily Vaccinations", "value": "DAILY_VACCINATIONS"},
    {"label": "People Vaccinated", "value": "PEOPLE_VACCINATED"},
    {"label": "People Fully Vaccinated", "value": "PEOPLE_FULLY_VACCINATED"},
    {"label": "Total Vaccinations per 100", "value": "TOTAL_VACCINATIONS_PER_HUNDRED"},
    {"label": "People Vaccinated per 100", "value": "PEOPLE_VACCINATED_PER_HUNDRED"},
    {"label": "People Fully Vaccinated per 100", "value": "PEOPLE_FULLY_VACCINATED_PER_HUNDRED"},

    {"label": "💵 GDP Metrics", "value": "HEADER_GDP", "disabled": True},
    {"label": "GDP PPP per Capita", "value": "GDP_PPP_PER_CAPITA"},
    {"label": "GDP vs Cases per 100k (Year)", "value": "GDP_VS_CASES_PER100K_YEAR"},
]

external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "COVID-19 Dashboard"

navbar = dbc.Navbar(
    dbc.Container([
        html.Div(
            [
                html.Span("🧭", className="me-2 text-white"),  
                dbc.NavbarBrand(
                    "COVID-19 Interactive Dashboard",
                    className="fw-semibold text-white"
                ),
            ],
            className="d-flex align-items-center"
        ),
        dbc.NavbarToggler(id="navbar-toggler", className="text-white"),
        dbc.Collapse(
            dbc.Nav(
                [
                    dbc.Label(
                        "Chart theme",
                        className="me-2 mb-0 text-white", 
                        html_for="theme"
                    ),
                    dcc.Dropdown(
                        id="theme",
                        options=[
                            {"label": "Light", "value": "light"},
                            {"label": "Dark", "value": "dark"}
                        ],
                        value="light",
                        clearable=False,
                        style={
                            "width": 120,
                            "backgroundColor": "white",
                            "color": "black"
                        },
                    ),
                ],
                className="ms-auto",
                navbar=True
            ),
            id="navbar-collapse",
            navbar=True,
            is_open=True,
        ),
    ]),
    color=None,  
    dark=True,
    style={"backgroundColor": "#3f87c0"}, 
    className="mb-4 shadow-sm"
)

filters_card = dbc.Card(
    dbc.CardBody(
        [
            
            dbc.Row([dbc.Col(html.Div(id="gdp-status"), width="auto", className="mb-2")]),

          
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Country"),
                            dcc.Input(
                                id="country",
                                type="text",
                                value="Lithuania",
                                placeholder="Type a country...",
                                className="form-control"
                            ),
                        ],
                        md=3
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Metric"),
                            dcc.Dropdown(
                                id="metric",
                                options=metric_options,
                                value="NEW_CASES_PER_100K",
                                clearable=False,
                            ),
                        ],
                        md=4
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Date range"),
                            dcc.DatePickerRange(
                                id="date-range",
                                start_date=DEFAULT_START,
                                end_date=DEFAULT_END,
                                min_date_allowed=DATA_MIN,
                                max_date_allowed=DATA_MAX,
                                display_format="YYYY-MM-DD",
                                className="w-100"
                            ),
                        ],
                        md=4
                    ),
                    dbc.Col(
                        [
                            dbc.Label(" "), 
                            dbc.Button(
                                "Update chart",
                                id="update-btn",
                                style={
                                    "backgroundColor": "#3f87c0",
                                    "borderColor": "#3f87c0",
                                    "color": "white",
                                    "whiteSpace": "nowrap",
                                    "height": "38px",
                                    "padding": "0 16px",
                                    "borderRadius": "8px"
                                },
                                className="w-100"
                            ),
                        ],
                        width="auto",  
                        className="align-self-end"  
                    ),
                ],
                className="g-3 align-items-end"
            ),
        ]
    ),
    className="mb-3 shadow-sm border-0"
)


chart_card = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Row(
                [
                    dbc.Col(html.H5("Metric Visualization", className="mb-0"), xs=8),
                    dbc.Col(dbc.Badge("Interactive", color="info", className="float-end"), xs=4),
                ],
                align="center",
                className="g-0"
            )
        ),
        dbc.CardBody(
            dcc.Loading(
                id="loading-graph",
                type="circle",
                children=dcc.Graph(id="metric-chart", config={"displayModeBar": True}),
            )
        ),
    ],
    className="mb-4 shadow-sm border-0"
)

annot_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Add annotation", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(dcc.Input(id="user", type="text", placeholder="Your name", className="form-control"), md=3),
                    dbc.Col(dcc.Input(id="comment", type="text", placeholder="Comment", className="form-control"), md=7),
                    dbc.Col(dbc.Button("Submit", id="submit-comment", color="success", className="w-100"), md=2),
                ],
                className="g-2"
            ),
            html.Div(id="status-msg", className="mt-3"),
        ]
    ),
    className="mb-3 shadow-sm border-0"
)

comments_card = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Button(
                "▼ Hide Comments",
                id="toggle-comments",
                color="light",
                className="w-100 text-start"
            )
        ),
        dbc.Collapse(
            dbc.CardBody(
                [
                    html.H6("Previous comments", className="mb-2"),
                    html.Div(
                        id="comments-list",
                        style={"border": "1px solid #e9ecef", "padding": "8px", "maxHeight": "220px", "overflowY": "auto"},
                        className="rounded"
                    ),
                ]
            ),
            id="comments-collapse",
            is_open=True
        ),
    ],
    className="mb-5 shadow-sm border-0"
)

clear_status = dcc.Interval(id="clear-status", interval=3000, n_intervals=0, disabled=True)


app.layout = dbc.Container(
    [
        navbar,
        filters_card,
        clear_status,
        chart_card,
        annot_card,
        comments_card,
        dbc.Tooltip("Type a country name (e.g., Lithuania, Germany).", target="country", placement="bottom"),
        dbc.Tooltip("Choose what to visualize. Headers are separators.", target="metric", placement="bottom"),
        dbc.Tooltip("Pick your date window.", target="date-range", placement="bottom"),
        dbc.Tooltip("Render chart with current filters.", target="update-btn", placement="bottom"),
    ],
    fluid=True
)


def fetch_comments(country, metric):
    try:
        r = requests.get(f"{API_BASE}/comments/{country}/{metric}", timeout=10)
        if r.status_code != 200:
            return [dbc.Alert(f"Error loading comments: {r.text}", color="danger", className="mb-2")]
        comments = r.json().get("comments", [])
        if not comments:
            return [html.Div("No comments yet.")]
        return [html.Div(f"{c['timestamp'][:10]} — {c['user']}: {c['text']}") for c in comments]
    except Exception as e:
        return [dbc.Alert(f"Error fetching comments: {e}", color="warning", className="mb-2")]



@app.callback(
    Output("metric", "options"),
    Output("metric", "value"),
    Output("gdp-status", "children"),
    Input("country", "value"),
    State("metric", "value")
)
def gate_gdp_options(country, current_metric):
    allowed = gdp_allowed(country)
    opts = deepcopy(metric_options)

    for opt in opts:
        val = opt.get("value")
        if val in GDP_METRICS:
            opt["disabled"] = opt.get("disabled", False) or (not allowed)

    new_value = current_metric
    if (current_metric in GDP_METRICS) and (not allowed):
        new_value = "NEW_CASES_PER_100K"  # safe default

    status = (
        dbc.Badge("GDP metrics available", color="success", className="me-2")
        if allowed else
        dbc.Badge("GDP metrics unavailable for this country", color="secondary", className="me-2")
    )
    return opts, new_value, status


@app.callback(
    Output("metric-chart", "figure"),
    Input("update-btn", "n_clicks"),
    State("country", "value"),
    State("metric", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("theme", "value"),
    prevent_initial_call=True
)
def update_chart(_, country, metric, start_date, end_date, theme):
    start_date = clamp_date(start_date)
    end_date = clamp_date(end_date)

    canon = to_canonical(country) or country

    if metric in GDP_METRICS and not gdp_allowed(country):
        template = "plotly_dark" if (theme or "light") == "dark" else "plotly_white"
        paper_bg = "#151a1e" if template == "plotly_dark" else "white"
        plot_bg = "#151a1e" if template == "plotly_dark" else "white"
        fig = go.Figure().add_annotation(
            text=f"GDP metrics are not available for '{canon}'. Choose a different metric or country."
        )
        fig.update_layout(template=template, paper_bgcolor=paper_bg, plot_bgcolor=plot_bg)
        return fig

    payload = {"country": canon, "metric": metric, "start": start_date, "end": end_date}

    template = "plotly_dark" if (theme or "light") == "dark" else "plotly_white"
    paper_bg = "#151a1e" if template == "plotly_dark" else "white"
    plot_bg = "#151a1e" if template == "plotly_dark" else "white"

    try:
        resp = requests.post(f"{API_BASE}/metrics", json=payload, timeout=20)
    except requests.exceptions.RequestException as e:
        fig = go.Figure().add_annotation(text=f"Error connecting to API: {e}")
        fig.update_layout(template=template, paper_bgcolor=paper_bg, plot_bgcolor=plot_bg)
        return fig

    if resp.status_code != 200:
        fig = go.Figure().add_annotation(text=f"API returned {resp.status_code}: {resp.text}")
        fig.update_layout(template=template, paper_bgcolor=paper_bg, plot_bgcolor=plot_bg)
        return fig

    data = resp.json().get("data", [])
    if not data:
        fig = go.Figure().add_annotation(text="No data available")
        fig.update_layout(template=template, paper_bgcolor=paper_bg, plot_bgcolor=plot_bg)
        return fig

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    if metric.upper() == "GDP_VS_CASES_PER100K_YEAR":
        df_pivot = df.pivot(index="date", columns="metric", values="value").reset_index()
        needed = {"GDP_PPP_PER_CAPITA", "NEW_CASES_PER_100K"}
        if not needed.issubset(df_pivot.columns):
            fig = go.Figure().add_annotation(text="Required columns not returned by API.")
            fig.update_layout(template=template, paper_bgcolor=paper_bg, plot_bgcolor=plot_bg)
            return fig

        df_pivot["year"] = df_pivot["date"].dt.year
        df_pivot = df_pivot.dropna(subset=["GDP_PPP_PER_CAPITA", "NEW_CASES_PER_100K"])
        df_pivot = df_pivot[df_pivot["GDP_PPP_PER_CAPITA"] > 0]
        df_pivot = df_pivot[df_pivot["NEW_CASES_PER_100K"] >= 0]

        if df_pivot.empty:
            fig = go.Figure().add_annotation(text="No valid GDP/cases rows to plot.")
            fig.update_layout(template=template, paper_bgcolor=paper_bg, plot_bgcolor=plot_bg)
            return fig

        max_cases = df_pivot["NEW_CASES_PER_100K"].max()
        df_pivot["bubble_size"] = (df_pivot["NEW_CASES_PER_100K"] / max_cases) * 40 + 6

        fig = go.Figure(go.Scatter(
            x=df_pivot["GDP_PPP_PER_CAPITA"],
            y=df_pivot["NEW_CASES_PER_100K"],
            mode="markers",
            marker=dict(
                size=df_pivot["bubble_size"],
                color=df_pivot["year"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=dict(text="Year", side="top"),
                              tickvals=[2020, 2021, 2022, 2023]),
                line=dict(width=0.5, color="DarkSlateGrey"),
                opacity=0.8
            ),
            hovertemplate=(
                "GDP PPP per Capita: %{x:,.0f}<br>"
                "New Cases per 100k: %{y:,.1f}<br>"
                "Year: %{marker.color}<br>"
                "Date: %{text}<extra></extra>"
            ),
            text=df_pivot["date"].dt.strftime("%Y-%m-%d"),
            name="GDP vs Cases per 100k"
        ))

        fig.update_layout(
            title=f"GDP vs Cases per 100k in {canon} ({start_date} → {end_date})",
            xaxis_title="GDP PPP per Capita (USD)",
            yaxis_title="New Cases per 100k",
            xaxis=dict(tickformat=","),
            legend=dict(orientation="h", y=1.12),
            template=template,
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            margin=dict(l=40, r=30, t=60, b=40)
        )
        return fig

    if metric.upper() == "GDP_PPP_PER_CAPITA":
        df_year = (
            df.set_index("date")
              .resample("YE")["value"]
              .last()
              .reset_index()
        )
        if df_year.empty:
            fig = go.Figure().add_annotation(text="No GDP data to plot.")
            fig.update_layout(template=template, paper_bgcolor=paper_bg, plot_bgcolor=plot_bg)
            return fig

        df_year["year"] = df_year["date"].dt.year
        fig = go.Figure(go.Bar(
            x=df_year["year"],
            y=df_year["value"],
            name="GDP PPP per Capita",
            hovertemplate="Year: %{x}<br>GDP PPP per Capita: %{y:,.0f}<extra></extra>"
        ))

        fig.update_layout(
            title=f"GDP PPP per Capita in {canon} ({start_date} → {end_date})",
            xaxis_title="Year",
            yaxis_title="USD",
            xaxis=dict(type="category"),
            bargap=0.2,
            template=template,
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            margin=dict(l=40, r=30, t=60, b=40)
        )
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["value"], mode="lines+markers", name="Value",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"{metric} in {canon} ({start_date} → {end_date})",
        xaxis_title="Date",
        yaxis_title="Value",
        template=template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        margin=dict(l=40, r=30, t=60, b=40)
    )
    return fig


@app.callback(
    Output("comments-list", "children"),
    Input("country", "value"),
    Input("metric", "value")
)
def load_comments(country, metric):
    return fetch_comments(to_canonical(country) or country, metric)

@app.callback(
    Output("comments-list", "children", allow_duplicate=True),
    Output("status-msg", "children"),
    Output("clear-status", "disabled"),
    Input("submit-comment", "n_clicks"),
    State("country", "value"),
    State("metric", "value"),
    State("user", "value"),
    State("comment", "value"),
    State("date-range", "end_date"),
    prevent_initial_call=True
)
def add_comment(_, country, metric, user, comment, end_date):
    if not (user and comment):
        alert = dbc.Alert("Please provide your name and a comment.", color="warning", className="py-2 px-3 mb-0")
        return fetch_comments(country, metric), alert, False

    canon = to_canonical(country) or country
    payload = {
        "country": canon,
        "date": clamp_date(end_date),
        "metric": metric,
        "user": user,
        "comment": comment,
        "value": None
    }
    try:
        r = requests.post(f"{API_BASE}/comments/add", json=payload, timeout=15)
        if r.status_code != 200:
            alert = dbc.Alert(f"Failed to add comment: {r.status_code} {r.text}", color="danger", className="py-2 px-3 mb-0")
            return fetch_comments(canon, metric), alert, False
    except requests.exceptions.RequestException as e:
        alert = dbc.Alert(f"Error: {e}", color="danger", className="py-2 px-3 mb-0")
        return fetch_comments(canon, metric), alert, False

    alert = dbc.Alert("✅ Comment added!", color="success", className="py-2 px-3 mb-0")
    return fetch_comments(canon, metric), alert, False


@app.callback(
    Output("comments-collapse", "is_open"),
    Output("toggle-comments", "children"),
    Input("toggle-comments", "n_clicks"),
    State("comments-collapse", "is_open"),
    prevent_initial_call=False
)
def toggle_comments(n_clicks, is_open):
    if not n_clicks:
        return True, "▼ Hide Comments"
    new_state = not is_open
    label = "▼ Hide Comments" if new_state else "▲ Show Comments"
    return new_state, label


@app.callback(
    Output("status-msg", "children", allow_duplicate=True),
    Output("clear-status", "disabled", allow_duplicate=True),
    Input("clear-status", "n_intervals"),
    prevent_initial_call=True
)
def clear_status(_):
    return "", True


if __name__ == "__main__":
    app.run(debug=True, port=8050)
