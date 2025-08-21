from datetime import datetime
from typing import Any
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.results import UpdateResult
from app.core.config import MONGO_URI, MONGO_DB, MONGO_COLLECTION

_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
_db = _client[MONGO_DB]
annotations: Collection = _db[MONGO_COLLECTION]

def ensure_indexes() -> None:
    annotations.create_index(
        [("country", ASCENDING), ("date", ASCENDING), ("metric", ASCENDING)],
        unique=True,
        name="uniq_country_date_metric",
    )
    annotations.create_index([("date", ASCENDING)], name="by_date")

def get_annotations(country: str, metric: str, start: str, end: str) -> list[dict[str, Any]]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    cur = annotations.find(
        {"country": country, "metric": metric, "date": {"$gte": start_dt, "$lte": end_dt}},
        {"_id": 0},
    )
    return list(cur)

def upsert_comment(
    country: str,
    date_str: str,
    metric: str,
    user: str,
    comment: str,
) -> UpdateResult:
    """Create the daily doc if missing; otherwise append a text comment."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    comment_doc = {
        "user": user,
        "text": comment,
        "timestamp": datetime.utcnow(),  # store UTC
    }
    return annotations.update_one(
        {"country": country, "date": dt, "metric": metric},
        {
            "$setOnInsert": {"created_at": datetime.utcnow()},
            "$push": {"comments": comment_doc},
        },
        upsert=True,
    )
