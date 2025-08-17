from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from cachetools import TTLCache, cached
import pandas as pd

from app.core.config import CORS_ORIGINS
from app.api.schemas import MetricsQuery, CommentIn, MetricsRow
from app.api.db_snowflake import fetch_metric_series
from app.api.db_mongo import ensure_indexes, get_annotations, upsert_comment

# NEW: handle Snowflake-specific errors gracefully
from snowflake.connector.errors import ProgrammingError, DatabaseError


app = FastAPI(title="COVID API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_indexes()

_cache = TTLCache(maxsize=256, ttl=300)

@cached(_cache)
def _get_sf_series(country, metric, start, end):
    return fetch_metric_series(country, metric, start, end)

# --- helper to convert Snowflake errors to HTTPException
def _snowflake_http_exc(e: Exception) -> HTTPException:
    msg = str(e)
    low = msg.lower()
    if ("resource monitor" in low) or ("cannot be resumed" in low):
        return HTTPException(
            status_code=503,
            detail=("Snowflake warehouse budget exceeded (resource monitor). "
                    "Increase or reset the monitor, attach a different monitor, "
                    "or switch to a warehouse with credits."),
        )
    if ("is suspended" in low) or ("paused" in low):
        return HTTPException(
            status_code=503,
            detail=("Snowflake warehouse is suspended and cannot run queries. "
                    "Resume the warehouse or enable AUTO_RESUME."),
        )
    # default: bubble the exact Snowflake message for debugging
    return HTTPException(status_code=500, detail=f"Snowflake error: {msg}")

@app.post("/metrics", response_model=dict)
def metrics(q: MetricsQuery):
    try:
        rows = _get_sf_series(q.country, q.metric, q.start, q.end)
    except ValueError as e:
        # your own validation errors
        raise HTTPException(400, str(e))
    except (ProgrammingError, DatabaseError) as e:
        # graceful error instead of 500
        raise _snowflake_http_exc(e)

    if not rows:
        return {"data": [], "cached": True}

    df = pd.DataFrame(rows)
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")

    # annotations from Mongo
    anns = get_annotations(q.country, q.metric, q.start, q.end)
    ann_map = {
        (
            a["country"],
            a["date"].date().isoformat() if hasattr(a["date"], "date") else a["date"],
            a["metric"],
        ): a
        for a in anns
    }

    out = []
    for _, r in df.iterrows():
        k = (r["COUNTRY"], str(r["DATE"]), r["METRIC"])
        out.append(
            MetricsRow(
                country=r["COUNTRY"],
                date=str(r["DATE"]),
                metric=r["METRIC"],
                value=0.0 if pd.isna(r["VALUE"]) else float(r["VALUE"]),
                annotation=ann_map.get(k),
            ).dict()
        )

    return {"data": out, "cached": True}

@app.post("/comments/add", response_model=dict)
def comments_add(c: CommentIn):
    res = upsert_comment(c.country, c.date, c.metric, c.user, c.comment, c.value)
    _cache.clear()
    return {
        "ok": True,
        "matched": res.matched_count,
        "modified": res.modified_count,
        "upserted_id": str(res.upserted_id) if res.upserted_id else None,
    }

@app.get("/comments/{country}/{metric}", response_model=dict)
def comments_list(
    country: str,
    metric: str,
    start: str | None = Query(None),
    end: str | None = Query(None),
):
    start = start or "2020-01-01"
    end = end or "2100-01-01"

    docs = get_annotations(country, metric, start, end)

    items = []
    for d in docs:
        for c in d.get("comments", []):
            ts = c.get("timestamp")
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            items.append(
                {
                    "timestamp": ts_str,
                    "user": c.get("user"),
                    "text": c.get("text"),
                    "value": c.get("value"),
                }
            )

    items.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"comments": items[:200]}

@app.get("/healthz")
def healthz():
    return {"ok": True}
