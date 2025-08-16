from app.core.constants import GDP_METRICS

_BASE_METRIC_OPTIONS = [
    ("New Cases", "NEW_CASES"),
    ("New Cases per 100k", "NEW_CASES_PER_100K"),
    ("New Deaths", "NEW_DEATHS"),
    ("New Deaths per 100k", "NEW_DEATHS_PER_100K"),
    ("Total Vaccinations", "TOTAL_VACCINATIONS"),
    ("Daily Vaccinations", "DAILY_VACCINATIONS"),
    ("People Vaccinated", "PEOPLE_VACCINATED"),
    ("People Fully Vaccinated", "PEOPLE_FULLY_VACCINATED"),
    ("Total Vaccinations per 100", "TOTAL_VACCINATIONS_PER_HUNDRED"),
    ("People Vaccinated per 100", "PEOPLE_VACCINATED_PER_HUNDRED"),
    ("People Fully Vaccinated per 100", "PEOPLE_FULLY_VACCINATED_PER_HUNDRED"),
    ("GDP PPP per Capita", "GDP_PPP_PER_CAPITA"),
    ("GDP vs Cases per 100k (Year)", "GDP_VS_CASES_PER100K_YEAR"),
]

def base_metric_options():
    return _BASE_METRIC_OPTIONS

def non_gdp_metric_labels():
    return [l for (l, v) in _BASE_METRIC_OPTIONS if v not in GDP_METRICS]

def build_metric_choices(allow_gdp: bool):
    choices = []
    for label, value in _BASE_METRIC_OPTIONS:
        if value in GDP_METRICS and not allow_gdp:
            continue
        choices.append((label, value))
    return choices

def is_forecastable(metric: str) -> bool:
    return metric not in GDP_METRICS
