# app/core/snowflake_conn.py
import snowflake.connector
from snowflake.connector.errors import ProgrammingError, DatabaseError, InterfaceError
from typing import Tuple
from .config import SNOWFLAKE

_REQUIRED = ("account", "user", "warehouse")

def have_sf_config() -> bool:
    has_required = all(SNOWFLAKE.get(k) for k in _REQUIRED)
    has_auth = bool(SNOWFLAKE.get("password")) or bool(SNOWFLAKE.get("authenticator"))
    return has_required and has_auth

def get_sf_conn():
    if not have_sf_config():
        raise RuntimeError("Snowflake config missing or incomplete")

    user      = SNOWFLAKE["user"]
    account   = SNOWFLAKE["account"]
    warehouse = SNOWFLAKE["warehouse"]
    role      = SNOWFLAKE.get("role")
    database  = SNOWFLAKE.get("database", "COVID_DB")
    schema    = SNOWFLAKE.get("schema", "PUBLIC")

    kwargs = dict(user=user, account=account, warehouse=warehouse, database=database, schema=schema)
    if role:
        kwargs["role"] = role
    if SNOWFLAKE.get("authenticator"):
        kwargs["authenticator"] = SNOWFLAKE["authenticator"]
    else:
        kwargs["password"] = SNOWFLAKE["password"]

    conn = snowflake.connector.connect(**kwargs)

    # Ensure session context; try resume if suspended.
    with conn.cursor() as cur:
        if role:
            cur.execute(f'USE ROLE "{role}"')
        cur.execute(f'USE DATABASE "{database}"')
        cur.execute(f'USE SCHEMA "{schema}"')
        cur.execute(f'USE WAREHOUSE "{warehouse}"')
        # Attempt to resume; if a resource monitor blocks it, this will raise DatabaseError.
        try:
            cur.execute(f'ALTER WAREHOUSE "{warehouse}" RESUME IF SUSPENDED')
        except (ProgrammingError, DatabaseError, InterfaceError):
            # Ignore here; the real reason (monitor/privilege) will appear when we run a query.
            pass
        cur.execute("SELECT CURRENT_WAREHOUSE()")
        active = (cur.fetchone() or [None])[0]
        if not active or active.upper() != str(warehouse).upper():
            raise RuntimeError(f"Expected warehouse '{warehouse}' but session is on '{active}'")

    return conn

def db_schema() -> Tuple[str, str]:
    return SNOWFLAKE.get("database", "COVID_DB"), SNOWFLAKE.get("schema", "PUBLIC")
