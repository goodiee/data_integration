from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def _split_csv(value: str) -> list[str]:
    """Turn 'a,b,c' into ['a','b','c']; keep '*' as ['*']."""
    value = (value or "").strip()
    if value == "*" or not value:
        return ["*"]
    return [x.strip() for x in value.split(",") if x.strip()]


# CORS 
CORS_ORIGINS = _split_csv(os.getenv("CORS_ORIGINS", "*"))


# Snowflake 
SNOWFLAKE = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "role": os.getenv("SNOWFLAKE_ROLE"),         
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
}


# Mongo 
MONGO_URI = os.getenv("MONGO_URI")  
MONGO_DB = os.getenv("MONGO_DB", "covid_annotations")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "annotations")

