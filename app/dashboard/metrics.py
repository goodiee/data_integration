from __future__ import annotations

from typing import Iterable

from app.core.constants import GDP_METRICS

# (label, value) pairs shown to the user (order matters)
_BASE_METRIC_OPTIONS: tuple[tuple[str, str], ...] = (
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
)

def base_metric_options() -> Iterable[tuple[str, str]]:
    """All metrics in display order (label, value)."""
    return _BASE_METRIC_OPTIONS

def non_gdp_metric_labels() -> list[str]:
    """Labels for non-GDP metrics (useful for clustering UI)."""
    return [label for (label, value) in _BASE_METRIC_OPTIONS if value not in GDP_METRICS]

def build_metric_choices(allow_gdp: bool) -> list[tuple[str, str]]:
    """Filter metrics based on GDP allowance for the current primary country."""
    return [
        (label, value)
        for (label, value) in _BASE_METRIC_OPTIONS
        if allow_gdp or value not in GDP_METRICS
    ]

def is_forecastable(metric: str) -> bool:
    """Forecasting is enabled only for non-GDP metrics."""
    return metric not in GDP_METRICS
