import requests
from app.dashboard.config import API_BASE  # keep imports consistent

def fetch_comments(country: str, metric: str):
    try:
        r = requests.get(f"{API_BASE}/comments/{country}/{metric}", timeout=10)
        r.raise_for_status()
        comments = r.json().get("comments", [])
        if not comments:
            return True, ["No comments yet."]

        def _line(c):
            ts = (c.get("timestamp") or "")[:10]
            user = c.get("user") or "—"
            # be tolerant to either 'text' or 'comment'
            txt = c.get("text") or c.get("comment") or ""
            return f"{ts} — {user}: {txt}"

        return True, [_line(c) for c in comments]
    except Exception as e:
        return False, [f"Error fetching comments: {e}"]

def add_comment(country: str, date: str, metric: str, user: str, text: str):
    # send BOTH keys for compatibility with different backends
    payload = {
        "country": country,
        "date": date,
        "metric": metric,
        "user": user,
        "text": text,
        "comment": text,
        "value": None,
    }
    r = requests.post(f"{API_BASE}/comments/add", json=payload, timeout=15)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text
