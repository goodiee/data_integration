from __future__ import annotations

import requests
from urllib.parse import quote
from app.dashboard.config import API_BASE

DEFAULT_GET_TIMEOUT = 10
DEFAULT_POST_TIMEOUT = 15


def _comments_url(country: str, metric: str) -> str:
    """Build the comments endpoint URL with safe encoding."""
    return f"{API_BASE}/comments/{quote(country)}/{quote(metric)}"


def fetch_comments(country: str, metric: str):
    """
    Load comments for (country, metric).
    Returns (ok: bool, lines: list[str]).
    """
    try:
        r = requests.get(_comments_url(country, metric), timeout=DEFAULT_GET_TIMEOUT)
        r.raise_for_status()
        comments = (r.json() or {}).get("comments", [])
        if not comments:
            return True, ["No comments yet."]

        def _line(c: dict) -> str:
            ts = (c.get("timestamp") or "")[:10]  
            user = c.get("user") or "—"
            txt = (c.get("text") or c.get("comment") or "").strip()  
            return f"{ts} — {user}: {txt}"

        return True, [_line(c) for c in comments]
    except Exception as e:
        return False, [f"Error fetching comments: {e}"]


def add_comment(country: str, date: str, metric: str, user: str, text: str):
    """
    Add a text-only comment to a specific day (no date munging).
    Backend expects: {country, date, metric, user, comment}.
    """
    payload = {
        "country": country,
        "date": date,       
        "metric": metric,
        "user": user,
        "comment": text,    
        "text": text,       
    }
    r = requests.post(f"{API_BASE}/comments/add", json=payload, timeout=DEFAULT_POST_TIMEOUT)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text
