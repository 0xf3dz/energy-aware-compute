"""Apply packaged migrations with ``python -m db.migrate``."""

import argparse
import asyncio
import hashlib
import os
from importlib.resources import files

from psycopg import AsyncConnection

# Distinct locks for schema changes, runtime ownership, and job claims.
MIGRATION_LOCK = 0x45414301
OWNERSHIP_LOCK = 0x45414302
CLAIM_LOCK = 0x45414303


async def migrate(connection: AsyncConnection) -> None:
    """Apply missing SQL versions atomically; reject edits to applied migrations."""
    async with connection.transaction():
        await connection.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK,))
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version integer PRIMARY KEY, name text NOT NULL, checksum text NOT NULL, "
            "applied_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        migrations = sorted(
            (
                item
                for item in files("db").joinpath("migrations").iterdir()
                if item.name.endswith(".sql")
            ),
            key=lambda item: item.name,
        )
        for migration in migrations:
            version = int(migration.name.split("_", 1)[0])
            sql = migration.read_text(encoding="utf-8")
            checksum = hashlib.sha256(sql.encode()).hexdigest()
            cursor = await connection.execute(
                "SELECT checksum FROM schema_migrations WHERE version = %s", (version,)
            )
            row = await cursor.fetchone()
            if row is not None:
                stored = row["checksum"] if isinstance(row, dict) else row[0]
                if stored != checksum:
                    raise RuntimeError(f"Applied migration {migration.name} has changed")
                continue
            await connection.execute(sql, prepare=False)
            await connection.execute(
                "INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s)",
                (version, migration.name, checksum),
            )


async def migrate_url(dsn: str) -> None:
    async with await AsyncConnection.connect(dsn) as connection:
        await migrate(connection)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply energy-aware compute database migrations")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    arguments = parser.parse_args()
    if not arguments.database_url:
        parser.error("provide --database-url or DATABASE_URL")
    asyncio.run(migrate_url(arguments.database_url))


if __name__ == "__main__":
    main()
