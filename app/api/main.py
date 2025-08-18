from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List

import pandas as pd
from cachetools import TTLCache, cached
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import CORS_ORIGINS
from app.api.schemas import MetricsQuery, CommentIn, MetricsRow
from app.api.db_snowflake import fetch_metric_series
from app.api.db_mongo import ensure_indexes, get_annotations, upsert_comment

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="COVID API", version="1.0")

# simple, permissive CORS (tighten in prod if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# make sure Mongo indexes exist (idempotent)
ensure_indexes()

# tiny in-process cache for Snowflake reads
_cache = TTLCache(maxsize=256, ttl=300)


@cached(_cache)
def _get_sf_series(country: str, metric: str, start: str, end: str):
    """Light wrapper so Snowflake reads are cached for a bit."""
    return fetch_metric_series(country, metric, start, end)


# API metrics
@app.post("/metrics", response_model=dict)
def metrics(q: MetricsQuery):
    try:
        rows = _get_sf_series(q.country, q.metric, q.start, q.end)
    except Exception as e:
        logger.exception("Error in metrics()")
        raise HTTPException(500, str(e))

    if not rows:
        return {"data": [], "cached": True}

    # coerce VALUE to numeric; NaNs → 0.0 later
    df = pd.DataFrame(rows)
    df["VALUE"] = pd.to_numeric(df.get("VALUE"), errors="coerce")

    # preload any annotations for this window; index by (country, date, metric)
    anns = get_annotations(q.country, q.metric, q.start, q.end)
    ann_map = {
        (
            a["country"],
            (a["date"].date().isoformat() if hasattr(a["date"], "date") else a["date"]),
            a["metric"],
        ): a
        for a in anns
    }

    out: List[Dict[str, Any]] = []
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


# API comments
@app.post("/comments/add", response_model=dict)
def comments_add(c: CommentIn):
    try:
        res = upsert_comment(c.country, c.date, c.metric, c.user, c.comment)
        _cache.clear()
        return {
            "ok": True,
            "matched": res.matched_count,
            "modified": res.modified_count,
            "upserted_id": str(res.upserted_id) if res.upserted_id else None,
        }
    except Exception as e:
        logger.exception("comments_add failed")
        raise HTTPException(500, str(e))

@app.get("/comments/{country}/{metric}", response_model=dict)
def comments_list(
    country: str,
    metric: str,
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    try:
        start = start or "2020-01-01"
        end = end or "2100-01-01"
        docs = get_annotations(country, metric, start, end)

        items = []
        for d in docs:
            for c in d.get("comments", []):
                ts = c.get("timestamp")
                ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                items.append(
                    {"timestamp": ts_str, "user": c.get("user"), "text": c.get("text")}
                )
        items.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"comments": items[:200]}
    except Exception as e:
        logger.exception("comments_list failed")
        raise HTTPException(500, str(e))
