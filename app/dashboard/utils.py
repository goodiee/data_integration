from __future__ import annotations

from app.dashboard.data_access import load_alias_map, load_allowed_gdp_canonical
from app.dashboard.config import DATA_MIN, DATA_MAX


def clamp_date(date_str: str | None) -> str:
    """
    Clamp an ISO date string to the allowed range [DATA_MIN, DATA_MAX].
    If date_str is empty/None, return DATA_MIN.
    """
    if not date_str:
        return DATA_MIN
    # Strings compare lexicographically fine for YYYY-MM-DD format.
    return max(DATA_MIN, min(str(date_str), DATA_MAX))


def to_canonical(country: str) -> str:
    """
    Map a user-entered country name to its canonical form using the alias map.
    Returns the trimmed original if no alias match is found.
    """
    if not country:
        return ""
    alias_map = load_alias_map()  # cached by Streamlit in data_access
    return alias_map.get(country.strip().upper(), country.strip())


def gdp_allowed(country: str) -> bool:
    """
    True if the (canonical) country is permitted to use GDP metrics.
    """
    return to_canonical(country) in load_allowed_gdp_canonical()
