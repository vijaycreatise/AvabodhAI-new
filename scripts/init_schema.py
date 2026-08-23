"""
scripts/init_schema.py
-----------------------
Standalone database provisioning script. Run this once against ANY
Postgres instance (local install, a managed cloud instance, whatever) to
create everything this app needs to run: all tables (documents,
chat_threads, chat_messages — chunk storage lives in Qdrant now, see
scripts/init_qdrant.py), the restricted app role, and the Row-Level
Security policies.

This does exactly what previously only happened implicitly on API
startup (main.py calls the same init_db()) — pulled out here as its own
entry point because Postgres is no longer something docker-compose spins
up for you. Point DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD (see
config/settings.py) at your own Postgres instance via env vars or a
local .env file, then run:

    python scripts/init_schema.py
    python scripts/init_schema.py --create-database   # if DB_NAME doesn't exist yet

Safe to re-run — every step inside init_db() is idempotent, so running
this against an already-provisioned database just verifies everything
is still in place.

Requirements on the target Postgres instance:
  - DB_USER (see config/settings.py) must be a superuser, or at minimum
    able to CREATE ROLE and ALTER TABLE ... FORCE ROW LEVEL SECURITY —
    this is the one-time bootstrap identity, not the role the app
    connects as afterward. --create-database additionally needs CREATEDB.
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
# Run as `python scripts/init_schema.py` — sys.path[0] defaults to
# scripts/ itself, not the repo root, so config/db/utils wouldn't be
# importable without this (same pattern cli.py uses at the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text

from config.settings import get_settings
from db.database import init_db, check_db_connection
from utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_database_exists(settings) -> None:
    """
    --create-database: connects to Postgres's own always-present
    maintenance database ("postgres" — every Postgres server has one,
    it's what psql/pgAdmin log into before a target database exists) using
    DB_USER/DB_PASSWORD, and issues CREATE DATABASE for settings.DB_NAME
    if it doesn't already exist. CREATE DATABASE cannot run inside a
    transaction, hence the AUTOCOMMIT isolation level below — not
    optional, Postgres rejects it otherwise.

    Opt-in (only called when --create-database is passed) rather than
    automatic on every run — creating a database is a bigger, rarer
    action than the idempotent table/role/RLS provisioning init_db()
    already does unconditionally, so it gets its own explicit flag,
    matching this repo's pattern for other one-off operator actions.
    """
    maintenance_url = (
        f"postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/postgres"
    )
    engine = create_engine(maintenance_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": settings.DB_NAME},
            ).first()
            if exists:
                print(f"Database '{settings.DB_NAME}' already exists — skipping creation.")
                return
            # Database names can't be bind-parameterized in DDL — settings.DB_NAME
            # is operator-controlled config (an env var), not end-user input, so
            # this is the same accepted exception db/database.py's role-bootstrap
            # DDL already documents.
            safe_name = settings.DB_NAME.replace('"', '""')
            conn.execute(text(f'CREATE DATABASE "{safe_name}"'))
            print(f"✓ Created database '{settings.DB_NAME}'.")
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision the Postgres schema this app needs.")
    parser.add_argument(
        "--create-database", action="store_true",
        help="Create DB_NAME first (via Postgres's own 'postgres' maintenance database) if it doesn't already exist.",
    )
    args = parser.parse_args()

    settings = get_settings()

    if args.create_database:
        try:
            _ensure_database_exists(settings)
        except Exception as e:
            print(f"✗ Database creation failed: {e}")
            print("  Common causes: DB_USER lacks CREATEDB privilege, or the server is unreachable.")
            sys.exit(1)

    print(f"Provisioning schema on {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME} ...")

    try:
        init_db()
    except Exception as e:
        print(f"✗ Schema provisioning failed: {e}")
        print(f"  Common causes: database '{settings.DB_NAME}' doesn't exist yet (rerun with --create-database),")
        print("  DB_USER lacking superuser/CREATEROLE privileges, or the server unreachable.")
        sys.exit(1)

    if not check_db_connection():
        print("✗ Schema created, but the app role connection check failed.")
        print(f"  Verify APP_DB_USER='{settings.APP_DB_USER}' can reach the database.")
        sys.exit(1)

    print(f"✓ Schema provisioned: tables, app role '{settings.APP_DB_USER}', "
          "and Row-Level Security are all in place. Now run scripts/init_qdrant.py "
          "against your Qdrant instance to provision chunk storage.")


if __name__ == "__main__":
    main()
