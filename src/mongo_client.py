from __future__ import annotations
import os
from pathlib import Path
from urllib.parse import quote_plus
from datetime import datetime

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
from pymongo.errors import OperationFailure, ConfigurationError, ServerSelectionTimeoutError

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MONGO_URI = os.getenv("MONGO_URI")  
MONGO_HOST = os.getenv("MONGO_HOST")  
MONGO_USER = os.getenv("MONGO_USER")  
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD") 

DB_NAME = os.getenv("MONGO_DB", "covid_annotations")
COLL_NAME = os.getenv("MONGO_COLLECTION", "annotations")

def build_uri() -> str:
    """Return a valid MongoDB URI from either MONGO_URI or components."""
    if MONGO_URI and MONGO_URI.strip():
        return MONGO_URI.strip()
    missing = [k for k, v in {
        "MONGO_HOST": MONGO_HOST, "MONGO_USER": MONGO_USER, "MONGO_PASSWORD": MONGO_PASSWORD
    }.items() if not v]
    if missing:
        raise RuntimeError(
            "Missing Mongo credentials. Provide either MONGO_URI, "
            "or MONGO_HOST + MONGO_USER + MONGO_PASSWORD in .env. "
            f"Missing: {', '.join(missing)}"
        )
    return (
        f"mongodb+srv://{quote_plus(MONGO_USER)}:{quote_plus(MONGO_PASSWORD)}@{MONGO_HOST}"
        "/?retryWrites=true&w=majority"
    )

URI = build_uri()

def get_client():
    return MongoClient(URI, serverSelectionTimeoutMS=8000)

def get_collection():
    client = get_client()
    return client, client[DB_NAME][COLL_NAME]

def ensure_unique_index():
    client, col = get_collection()
    try:
        col.create_index(
            [("country", ASCENDING), ("date", ASCENDING), ("metric", ASCENDING)],
            unique=True,
            name="uniq_country_date_metric"
        )
    finally:
        client.close()

def insert_annotation(country: str, date: str, metric: str, user: str, comment: str, value: float):
    """Simple insert (will fail if unique index exists and doc already present)."""
    client, col = get_collection()
    try:
        doc = {
            "country": country,
            "date": datetime.strptime(date, "%Y-%m-%d"),
            "metric": metric,
            "user": user,
            "comment": comment,
            "value": float(value),
            "created_at": datetime.utcnow(),
        }
        res = col.insert_one(doc)
        return str(res.inserted_id)
    finally:
        client.close()

def upsert_comment(country: str, date: str, metric: str, user: str, text: str, value: float | None = None):
    """Create doc if missing; otherwise append a comment to 'comments' array."""
    client, col = get_collection()
    try:
        key = {"country": country, "date": datetime.strptime(date, "%Y-%m-%d"), "metric": metric}
        update = {
            "$setOnInsert": {"value": value, "created_at": datetime.utcnow()},
            "$push": {"comments": {"user": user, "timestamp": datetime.utcnow(), "text": text}}
        }
        res = col.update_one(key, update, upsert=True)
        return {"matched": res.matched_count, "modified": res.modified_count,
                "upserted_id": str(res.upserted_id) if res.upserted_id else None}
    finally:
        client.close()

def find_annotations(country: str | None = None):
    client, col = get_collection()
    try:
        q = {"country": country} if country else {}
        return list(col.find(q))
    finally:
        client.close()

if __name__ == "__main__":
    try:
        with get_client() as c:
            c.admin.command("ping")
        print("Ping OK")

        ensure_unique_index()
        print("Unique index ensured.")

        _id = insert_annotation("Lithuania", "2021-05-15", "cases_per_100k", "Maksym",
                                "Peak during second wave", 512.4)
        print("Inserted ID:", _id)

        print("All annotations:")
        for d in find_annotations():
            print(d)

        print("Upsert extra comment:")
        print(upsert_comment("Lithuania", "2021-05-15", "cases_per_100k", "Analyst",
                             "Second comment on same datapoint", None))

    except (OperationFailure, ConfigurationError, ServerSelectionTimeoutError, RuntimeError) as e:
        print("Mongo setup/connection error:", e)
