import snowflake.connector
from .config import SNOWFLAKE

_REQUIRED = ("account", "user", "warehouse")

def have_sf_config() -> bool:
    has_required = all(SNOWFLAKE.get(k) for k in _REQUIRED)
    has_auth = bool(SNOWFLAKE.get("password")) or bool(SNOWFLAKE.get("authenticator"))
    return has_required and has_auth

def get_sf_conn():
    if not have_sf_config():
        raise RuntimeError("Snowflake config missing or incomplete")
    kwargs = dict(
        user=SNOWFLAKE["user"],
        account=SNOWFLAKE["account"],
        warehouse=SNOWFLAKE["warehouse"],
        role=SNOWFLAKE.get("role"),
        database=SNOWFLAKE.get("database", "COVID_DB"),
        schema=SNOWFLAKE.get("schema", "PUBLIC"),
    )
    if SNOWFLAKE.get("authenticator"):
        kwargs["authenticator"] = SNOWFLAKE["authenticator"]
    else:
        kwargs["password"] = SNOWFLAKE["password"]
    return snowflake.connector.connect(**kwargs)

def db_schema():
    return SNOWFLAKE.get("database", "COVID_DB"), SNOWFLAKE.get("schema", "PUBLIC")
