# app/api/main.py
from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from cachetools import TTLCache, cached
import pandas as pd

from app.core.config import CORS_ORIGINS, SNOWFLAKE
from app.api.schemas import MetricsQuery, CommentIn, MetricsRow
from app.api.db_snowflake import fetch_metric_series
from app.api.db_mongo import ensure_indexes, get_annotations, upsert_comment

# Import all common Snowflake connector errors
from snowflake.connector.errors import (
    ProgrammingError,
    DatabaseError,
    InterfaceError,
    OperationalError,
)

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="COVID API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Make sure Mongo indexes exist
ensure_indexes()

_cache = TTLCache(maxsize=256, ttl=300)


@cached(_cache)
def _get_sf_series(country: str, metric: str, start: str, end: str):
    return fetch_metric_series(country, metric, start, end)


def _snowflake_http_exc(e: Exception) -> HTTPException:
    msg = str(e)
    low = msg.lower()
    if ("resource monitor" in low) or ("cannot be resumed" in low) or ("credit quota" in low):
        return HTTPException(503, "Snowflake warehouse blocked by resource monitor/credit quota.")
    if ("is suspended" in low) or ("warehouse is suspended" in low) or ("paused" in low):
        return HTTPException(503, "Snowflake warehouse is suspended. Resume or enable AUTO_RESUME.")
    if ("not authorized" in low) or ("insufficient privileges" in low):
        return HTTPException(403, "Insufficient privileges for warehouse/database/schema/tables.")
    return HTTPException(500, f"Snowflake error: {msg}")


# ---------- health & debug ----------
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/debug/env")
def debug_env():
    # never return secrets
    return {
        "snowflake": {
            "account": SNOWFLAKE.get("account"),
            "user": SNOWFLAKE.get("user"),
            "role": SNOWFLAKE.get("role"),
            "warehouse": SNOWFLAKE.get("warehouse"),
            "database": SNOWFLAKE.get("database"),
            "schema": SNOWFLAKE.get("schema"),
        }
    }

@app.get("/debug/sf")
def debug_sf():
    from app.core.snowflake_conn import get_sf_conn
    try:
        conn = get_sf_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_ROLE()")
            wh, db, sc, role = cur.fetchone()
        conn.close()
        return {"ok": True, "warehouse": wh, "database": db, "schema": sc, "role": role}
    except (ProgrammingError, DatabaseError, InterfaceError, OperationalError) as e:
        raise _snowflake_http_exc(e)
    except Exception as e:
        logger.exception("debug_sf failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}")

@app.get("/debug/routes")
def debug_routes():
    return {"routes": [r.path for r in app.routes]}

@app.on_event("startup")
def _log_loaded_main():
    logger.info("Loaded app from: %s", __file__)


# ---------- API: metrics ----------
@app.post("/metrics", response_model=dict)
def metrics(q: MetricsQuery):
    try:
        rows = _get_sf_series(q.country, q.metric, q.start, q.end)
    except (ProgrammingError, DatabaseError, InterfaceError, OperationalError) as e:
        raise _snowflake_http_exc(e)
    except RuntimeError as e:
        logger.exception("Runtime error in metrics()")
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    except Exception as e:
        logger.exception("Unhandled error in metrics()")
        raise HTTPException(500, f"{type(e).__name__}: {e}")

    if not rows:
        return {"data": [], "cached": True}

    df = pd.DataFrame(rows)
    df["VALUE"] = pd.to_numeric(df.get("VALUE"), errors="coerce")

    anns = get_annotations(q.country, q.metric, q.start, q.end)
    ann_map = {
        (a["country"],
         (a["date"].date().isoformat() if hasattr(a["date"], "date") else a["date"]),
         a["metric"]): a
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


# ---------- API: comments ----------
@app.post("/comments/add", response_model=dict)
def comments_add(c: CommentIn):
    try:
        res = upsert_comment(c.country, c.date, c.metric, c.user, c.comment, c.value)
        _cache.clear()
        return {
            "ok": True,
            "matched": res.matched_count,
            "modified": res.modified_count,
            "upserted_id": str(res.upserted_id) if res.upserted_id else None,
        }
    except Exception as e:
        logger.exception("comments_add failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}")

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
                    {"timestamp": ts_str, "user": c.get("user"), "text": c.get("text"), "value": c.get("value")}
                )
        items.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"comments": items[:200]}
    except Exception as e:
        logger.exception("comments_list failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}")
