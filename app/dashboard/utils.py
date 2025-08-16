from typing import Optional
from data_access import load_alias_map, load_allowed_gdp_canonical
from config import DATA_MIN, DATA_MAX

def clamp_date(date_str: Optional[str]) -> str:
    if not date_str:
        return DATA_MIN
    return max(DATA_MIN, min(date_str, DATA_MAX))

def to_canonical(country: str) -> str:
    if not country:
        return ""
    alias_map = load_alias_map()
    canon = alias_map.get(country.strip().upper())
    return canon if canon else country.strip()

def gdp_allowed(country: str) -> bool:
    return to_canonical(country) in load_allowed_gdp_canonical()
