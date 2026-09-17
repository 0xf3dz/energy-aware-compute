"""PostgreSQL storage. The only module that knows SQL."""
from db.repository import PostgresQueue

__all__ = ["PostgresQueue"]
