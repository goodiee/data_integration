from __future__ import annotations

import snowflake.connector
from .config import SNOWFLAKE

# minimal set required to even try connecting
_REQUIRED = ("account", "user", "warehouse")


def have_sf_config() -> bool:
    """True if required fields + either password or authenticator are present."""
    has_required = all(SNOWFLAKE.get(k) for k in _REQUIRED)
    has_auth = bool(SNOWFLAKE.get("password") or SNOWFLAKE.get("authenticator"))
    return bool(has_required and has_auth)


def get_sf_conn():
    """
    Open a Snowflake connection and set session context (role/db/schema/warehouse).
    Tries to resume the warehouse; ignore if that fails here — queries will surface real issues.
    """
    if not have_sf_config():
        raise RuntimeError("Snowflake config missing or incomplete.")

    user = SNOWFLAKE["user"]
    account = SNOWFLAKE["account"]
    warehouse = SNOWFLAKE["warehouse"]
    role = SNOWFLAKE.get("role")
    database = SNOWFLAKE.get("database", "COVID_DB")
    schema = SNOWFLAKE.get("schema", "PUBLIC")

    kwargs = {
        "user": user,
        "account": account,
        "warehouse": warehouse,
        "database": database,
        "schema": schema,
    }
    if role:
        kwargs["role"] = role

    # password or external auth
    if SNOWFLAKE.get("authenticator"):
        kwargs["authenticator"] = SNOWFLAKE["authenticator"]
    else:
        kwargs["password"] = SNOWFLAKE["password"]

    conn = snowflake.connector.connect(**kwargs)

    # set session context; attempt resume without making noise
    with conn.cursor() as cur:
        if role:
            cur.execute(f'USE ROLE "{role}"')
        cur.execute(f'USE DATABASE "{database}"')
        cur.execute(f'USE SCHEMA "{schema}"')
        cur.execute(f'USE WAREHOUSE "{warehouse}"')
        try:
            cur.execute(f'ALTER WAREHOUSE "{warehouse}" RESUME IF SUSPENDED')
        except Exception:
            pass  

        # quick sanity check
        cur.execute("SELECT CURRENT_WAREHOUSE()")
        active = (cur.fetchone() or [None])[0]
        if not active or active.upper() != str(warehouse).upper():
            raise RuntimeError(f"Expected warehouse '{warehouse}' but session is on '{active}'")

    return conn


def db_schema() -> tuple[str, str]:
    """Return (database, schema) to build fully qualified names in SQL."""
    return SNOWFLAKE.get("database"), SNOWFLAKE.get("schema")
