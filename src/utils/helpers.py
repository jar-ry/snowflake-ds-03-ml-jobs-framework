def table_exists(session, fully_qualified_name: str) -> bool:
    """Check if a table exists in Snowflake."""
    try:
        _ = session.table(fully_qualified_name).schema
        return True
    except Exception:
        return False
