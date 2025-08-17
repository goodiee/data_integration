from datetime import datetime
from typing import List, Dict, Any

from pymongo import MongoClient, ASCENDING
from app.core.config import MONGO_URI, MONGO_DB, MONGO_COLLECTION


_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
_db = _client[MONGO_DB]
annotations = _db[MONGO_COLLECTION]


def ensure_indexes() -> None:
    """Ensure indexes used by the app exist."""
    annotations.create_index(
        [("country", ASCENDING), ("date", ASCENDING), ("metric", ASCENDING)],
        unique=True,
        name="uniq_country_date_metric",
    )
    annotations.create_index([("date", ASCENDING)], name="by_date")


def get_annotations(country: str, metric: str, start: str, end: str) -> List[Dict[str, Any]]:
    """Return docs for a given country/metric in the inclusive date range [start, end]."""
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
    value: float | None = None,
):
    """
    Create the datapoint doc if missing; otherwise append a comment.
    Stores user + text + value + timestamp inside comments[].
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")

    comment_doc = {
        "user": user,
        "text": comment,
        "value": value,
        "timestamp": datetime.utcnow(),
    }

    return annotations.update_one(
        {"country": country, "date": dt, "metric": metric},
        {
            "$setOnInsert": {"value": value, "created_at": datetime.utcnow()},
            "$push": {"comments": comment_doc},
        },
        upsert=True,
    )