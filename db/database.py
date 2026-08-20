"""
db/database.py
--------------
SQLAlchemy engines + session factories.

TWO engines, deliberately, not one:

  _admin_engine — connects using DB_USER/DB_PASSWORD (the Postgres
  superuser in this docker-compose setup). Used ONLY inside init_db(),
  ONLY for one-time schema bootstrap: creating tables, the full-text-
  search trigger, the restricted app role, and the Row-Level Security
  policies. Never used to serve a real request.

  engine — connects using APP_DB_USER/APP_DB_PASSWORD (a restricted,
  non-superuser role created by init_db()). This is what every real
  request in this app actually queries through.

Why the split exists: Postgres Row-Level Security CANNOT restrict a
superuser's queries — that's a fixed Postgres rule, no policy setting
changes it. If the app queried using the superuser for everything, the
RLS policies created below would provide zero real protection for the
app's own traffic. This split is what makes RLS meaningful at all.
"""

from contextlib import contextmanager
from typing import Generator, Optional

from fastapi import Request
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import OperationalError

from config.settings import get_settings
from db.models import Base
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Admin/bootstrap engine — schema DDL only, never used for requests ──────
_admin_engine = create_engine(settings.db_url, pool_pre_ping=True, echo=False)

# ── App engine — everything else uses this. RLS actually applies here. ─────
engine = create_engine(
    settings.app_db_url,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_pre_ping=True,
    echo=False,
)

SessionFactory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# The 4 tables that carry tenant_id/org_unit_id and need RLS.
_ISOLATED_TABLES = ["document_summaries", "document_chunks", "chat_threads", "chat_messages"]


def _bootstrap_app_role(conn) -> None:
    """
    Create (or update the password of) the restricted application role
    that `engine` above connects as. Idempotent — safe to run on every
    startup. Not a superuser, not BYPASSRLS, so RLS genuinely applies to
    it — this is the role teammates should also use in pgAdmin if they
    want tenant/org-unit isolation to actually bound their own queries.

    APP_DB_PASSWORD is operator-controlled config (an env var), not
    end-user input, so direct (escaped) interpolation into DDL here is
    an accepted, deliberate exception to "always use bind parameters" —
    CREATE ROLE / ALTER ROLE cannot take bind parameters at all.
    """
    escaped_password = settings.APP_DB_PASSWORD.replace("'", "''")
    conn.execute(text(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '{settings.APP_DB_USER}') THEN
                CREATE ROLE {settings.APP_DB_USER} LOGIN PASSWORD '{escaped_password}';
            ELSE
                ALTER ROLE {settings.APP_DB_USER} LOGIN PASSWORD '{escaped_password}';
            END IF;
        END
        $$;
    """))
    conn.execute(text(f'GRANT CONNECT ON DATABASE "{settings.DB_NAME}" TO {settings.APP_DB_USER}'))
    conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {settings.APP_DB_USER}"))
    # Runs AFTER create_all(), so this covers every table that exists at
    # this point — including ones added by future schema changes, since
    # create_all() always runs first in init_db() below.
    conn.execute(text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {settings.APP_DB_USER}"))
    logger.info("App role '%s' verified/updated.", settings.APP_DB_USER)


def _bootstrap_rls(conn) -> None:
    """
    Enable + FORCE Row-Level Security on the 4 isolated tables, with a
    policy that only allows a row to be SEEN if it matches the
    tenant_id/org_unit_id set on the current session (via set_config()
    in get_db_session_fastapi/get_db_session_context below).

    FORCE (not just ENABLE) is required — without it, RLS is silently
    skipped for the table's OWNER, and in a simple setup like this the
    app role often ends up owning the tables it creates. FORCE closes
    that gap. It does NOT and cannot restrict an actual Postgres
    superuser — nothing can, by Postgres design — which is the whole
    reason the app connects as the restricted role above, not as
    DB_USER.

    WITH CHECK (true) deliberately does NOT restrict INSERT/UPDATE the
    same way USING restricts SELECT. Every write path in this codebase
    already sets tenant_id/org_unit_id explicitly and correctly at the
    Python level (see pipeline/storage.py, embedder.py, chat_storage.py)
    — RLS's job here is to be the backstop for ad-hoc READS (a teammate
    in pgAdmin, a future report, a bug in some other service), not to
    re-validate writes that are already independently guaranteed correct.
    Making writes fail-closed too would mean any request whose session
    variables aren't set for any reason (a bug, a forgotten header on a
    brand-new endpoint) fails LOUD on the write path, which is arguably
    safer — but it also means this table becomes unwritable the moment
    that wiring has any bug at all, anywhere. Read-side enforcement gets
    most of the real-world protection (this is what stops the "someone
    ran an unscoped SELECT in pgAdmin" scenario) at a much smaller blast
    radius. Revisit this trade-off if write-path correctness is ever in
    doubt.
    """
    for table in _ISOLATED_TABLES:
        conn.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
        conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
        conn.execute(text(f"DROP POLICY IF EXISTS {table}_isolation ON {table}"))
        conn.execute(text(f"""
            CREATE POLICY {table}_isolation ON {table}
            USING (
                tenant_id = current_setting('app.tenant_id', true)
                AND org_unit_id = current_setting('app.org_unit_id', true)
            )
            WITH CHECK (true)
        """))
    logger.info("Row-Level Security enabled + forced on: %s", ", ".join(_ISOLATED_TABLES))


def _bootstrap_fts_trigger(conn) -> None:
    """
    Create the auto-update trigger that populates document_chunks.chunk_tsvector
    from chunk_text on every INSERT/UPDATE. tsvector_update_trigger is a
    Postgres BUILT-IN function — no custom PL/pgSQL function needed here,
    just the trigger that calls it.

    FTS_LANGUAGE is a fixed config (default 'english') applied to every
    row regardless of the document's own detected language. This is a
    known, deliberate limitation — good stemming/stopword handling for
    non-English content would need a language-aware trigger variant
    (tsvector_update_trigger_column, driven by a real regconfig column,
    not the free-text `language` column this table already has). Worth
    revisiting if non-English search quality turns out to matter; not
    implemented here to avoid guessing at a design nobody's confirmed is
    needed yet.
    """
    conn.execute(text("DROP TRIGGER IF EXISTS tsvectorupdate ON document_chunks"))
    conn.execute(text(f"""
        CREATE TRIGGER tsvectorupdate
        BEFORE INSERT OR UPDATE ON document_chunks
        FOR EACH ROW EXECUTE FUNCTION
        tsvector_update_trigger(chunk_tsvector, 'pg_catalog.{settings.FTS_LANGUAGE}', chunk_text)
    """))
    logger.info("Full-text search trigger created on document_chunks (language=%s).", settings.FTS_LANGUAGE)


def init_db() -> None:
    """
    Schema bootstrap — runs on every app startup, using the ADMIN engine
    only. Idempotent: safe to run repeatedly against an already-set-up
    database, which matters since this runs on every container start,
    not just the first one.
    """
    try:
        with _admin_engine.connect() as conn:
            # Enable pgvector extension — must run before creating vector columns
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logger.info("pgvector extension enabled.")

        Base.metadata.create_all(bind=_admin_engine)
        logger.info("Database tables verified / created successfully.")

        with _admin_engine.begin() as conn:
            _bootstrap_fts_trigger(conn)
            _bootstrap_app_role(conn)
            _bootstrap_rls(conn)

    except OperationalError as e:
        logger.critical("Cannot connect to PostgreSQL: %s", e)
        raise


def backfill_tsvector(batch_size: int = 1000) -> int:
    """
    Populate chunk_tsvector for rows that predate the trigger. New rows
    are already covered automatically going forward — this is a one-time
    catch-up for existing data, meant to be run explicitly by an
    operator (see cli.py --backfill-tsvector), NOT automatically on
    every startup. init_db() intentionally does not call this.

    Batched deliberately, not one giant UPDATE — a single massive UPDATE
    on a large existing table takes a long lock and a huge transaction;
    Postgres's CREATE INDEX CONCURRENTLY-style caution applies here too.
    This commits between batches so it stays safe to run against a live
    table with real traffic, rather than stalling everything for however
    long a full-table UPDATE would take.

    Uses the ADMIN engine, not the restricted app role — this needs to
    see and fix every row across every tenant/department by design; it's
    a deliberate maintenance operation invoked by an operator, not
    something that should be scoped by RLS the way a real request is.

    Returns total rows updated.
    """
    total_updated = 0
    with _admin_engine.connect() as conn:
        while True:
            result = conn.execute(text(f"""
                UPDATE document_chunks
                SET chunk_tsvector = to_tsvector('pg_catalog.{settings.FTS_LANGUAGE}', chunk_text)
                WHERE id IN (
                    SELECT id FROM document_chunks
                    WHERE chunk_tsvector IS NULL
                    LIMIT :batch_size
                )
            """), {"batch_size": batch_size})
            conn.commit()
            updated = result.rowcount
            total_updated += updated
            if updated > 0:
                logger.info("Backfilled %d rows (%d total so far)...", updated, total_updated)
            if updated == 0:
                break
    logger.info("Backfill complete — %d rows updated.", total_updated)
    return total_updated


def check_db_connection() -> bool:
    """Checks connectivity using the APP engine — this is what actually matters for serving traffic."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("DB health check failed: %s", e)
        return False


def _apply_rls_context(session: Session, tenant_id: Optional[str], org_unit_id: Optional[str]) -> None:
    """
    Sets the session-local variables the RLS policies in _bootstrap_rls()
    check against. Uses set_config(..., true) — the `true` means
    "local to this transaction", equivalent to SET LOCAL, so it can
    never leak into a later request that reuses this pooled connection.

    Uses set_config() as a normal parameterized function call (bind
    params work here, unlike the raw SET command), NOT string
    interpolation — tenant_id/org_unit_id originate from HTTP headers,
    which are untrusted input, so this is the one place in this file
    where safe parameter binding is non-negotiable, not a nicety.
    """
    if tenant_id is not None:
        session.execute(text("SELECT set_config('app.tenant_id', :v, true)"), {"v": tenant_id})
    if org_unit_id is not None:
        session.execute(text("SELECT set_config('app.org_unit_id', :v, true)"), {"v": org_unit_id})


@contextmanager
def get_db_session(tenant_id: Optional[str] = None, org_unit_id: Optional[str] = None) -> Generator[Session, None, None]:
    session: Session = SessionFactory()
    try:
        _apply_rls_context(session, tenant_id, org_unit_id)
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_db_session_context(tenant_id: Optional[str] = None, org_unit_id: Optional[str] = None) -> Generator[Session, None, None]:
    """
    Context manager for use outside FastAPI routes (pipeline/storage.py,
    embedder.py, chat_storage.py). Every call site in this codebase
    already has tenant_id/org_unit_id in scope as its own function
    parameters — pass them through here so RLS-protected SELECTs inside
    these functions (dedup checks especially) don't silently start
    returning zero rows once RLS is enabled.
    """
    session: Session = SessionFactory()
    try:
        _apply_rls_context(session, tenant_id, org_unit_id)
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session_fastapi(request: Request) -> Generator[Session, None, None]:
    """
    FastAPI dependency injection version. Reads tenant_id/org_unit_id
    from request.state (set by TenantGuardMiddleware from the
    X-Tenant-ID/X-Org-Unit-ID headers) and applies them automatically —
    every route using `db: Session = Depends(get_db_session_fastapi)`
    gets RLS context for free, with no per-route change needed.

    Routes that don't receive those headers (health checks, and the
    signed preview-file endpoint, which is authorized via a token
    instead) simply get no RLS context set here — request.state.tenant_id
    is None for those, and _apply_rls_context() no-ops on None. The
    preview-file endpoint sets its own RLS context manually from the
    verified token payload — see api/routes/documents.py.
    """
    session: Session = SessionFactory()
    try:
        tenant_id = getattr(request.state, "tenant_id", None)
        org_unit_id = getattr(request.state, "org_unit_id", None)
        _apply_rls_context(session, tenant_id, org_unit_id)
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from typing import AsyncGenerator

async_engine = create_async_engine(
    settings.async_db_url,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionFactory = async_sessionmaker(
    bind=async_engine,
    autoflush=False,
    autocommit=False,
    class_=AsyncSession,
)


async def get_async_db_session_fastapi() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency injection version for async routes."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
