import json

from snowflake.snowpark import Session
from snowflake.snowpark.version import VERSION


def _quote_id(name: str) -> str:
    """Double-quote a Snowflake identifier to prevent SQL injection."""
    return '"' + name.replace('"', '""') + '"'


def create_session(conf: dict) -> tuple:
    """Create a Snowpark session from a connection JSON file and set context.

    Reads credentials from the file at ``conf["snowflake"]["connection_file"]``,
    creates a session, and sets the active database, schema, role, and warehouse.

    Returns:
        Tuple of (session, database, schema, warehouse).
    """
    sf = conf["snowflake"]
    with open(sf["connection_file"]) as f:
        connection_parameters = json.load(f)

    session = Session.builder.configs(connection_parameters).create()
    session.sql_simplifier_enabled = True

    role = connection_parameters.get("role", sf["role"])
    session.sql(f"USE DATABASE {_quote_id(sf['database'])}").collect()
    session.sql(f"USE SCHEMA {_quote_id(sf['schema'])}").collect()
    session.sql(f"USE ROLE {_quote_id(role)}").collect()
    session.sql(f"USE WAREHOUSE {_quote_id(sf['warehouse'])}").collect()
    session.sql(
        f"ALTER WAREHOUSE {_quote_id(sf['warehouse'])} SET WAREHOUSE_SIZE = {_quote_id(sf['warehouse_size'])}"
    ).collect()

    env = session.sql("SELECT current_user(), current_version()").collect()
    print("\nConnection Established:")
    print(f"  User      : {env[0][0]}")
    print(f"  Role      : {session.get_current_role()}")
    print(f"  Database  : {session.get_current_database()}")
    print(f"  Schema    : {session.get_current_schema()}")
    print(f"  Warehouse : {session.get_current_warehouse()}")
    print(f"  Snowpark  : {VERSION[0]}.{VERSION[1]}.{VERSION[2]}\n")

    return session, sf["database"], sf["schema"], sf["warehouse"]
