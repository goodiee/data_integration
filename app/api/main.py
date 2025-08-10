from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from cachetools import TTLCache, cached
import pandas as pd

from app.core.config import CORS_ORIGINS
from app.api.schemas import MetricsQuery, CommentIn, MetricsRow
from app.api.db_snowflake import fetch_metric_series
from app.api.db_mongo import ensure_indexes, get_annotations, upsert_comment


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

@app.post("/metrics", response_model=dict)
def metrics(q: MetricsQuery):
    try:
        rows = _get_sf_series(q.country, q.metric, q.start, q.end)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not rows:
        return {"data": [], "cached": True}

    df = pd.DataFrame(rows)
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")

    anns = get_annotations(q.country, q.metric, q.start, q.end)
    ann_map = {(a["country"], a["date"].date().isoformat() if hasattr(a["date"], "date") else a["date"], a["metric"]): a
               for a in anns}

    out = []
    for _, r in df.iterrows():
        k = (r["COUNTRY"], str(r["DATE"]), r["METRIC"])
        out.append(MetricsRow(
            country=r["COUNTRY"],
            date=str(r["DATE"]),
            metric=r["METRIC"],
            value=0.0 if pd.isna(r["VALUE"]) else float(r["VALUE"]),
            annotation=ann_map.get(k)
        ).dict())

    return {"data": out, "cached": True}

@app.post("/comments/add", response_model=dict)
def comments_add(c: CommentIn):
    res = upsert_comment(c.country, c.date, c.metric, c.user, c.comment, c.value)
    _cache.clear()
    return {"ok": True, "matched": res.matched_count, "modified": res.modified_count, "upserted_id": str(res.upserted_id) if res.upserted_id else None}


@app.get("/healthz")
def healthz():
    return {"ok": True}
