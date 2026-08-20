import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import documents, health
from api.middleware.logging_middleware import LoggingMiddleware
from api.middleware.tenant_guard_middleware import TenantGuardMiddleware
from db.database import init_db
from config.settings import get_settings
from utils.logger import get_logger
from api.routes import search
from api.routes import chat
from api.routes.web import router as web_router
from api.routes import kb
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

_DEFAULT_SECRET_KEY = "dev-only-CHANGE-ME-in-production"
_DEFAULT_APP_DB_PASSWORD = "CHANGE-ME-app-role-password"


def _apply_langsmith_env() -> None:
    """
    Push LangSmith + OpenAI settings into os.environ so that LangChain
    picks them up at import time regardless of load order.
    """
    s = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", s.OPENAI_API_KEY)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", s.LANGCHAIN_TRACING_V2)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", s.LANGCHAIN_ENDPOINT)
    os.environ.setdefault("LANGCHAIN_PROJECT", s.LANGCHAIN_PROJECT)
    if s.LANGCHAIN_API_KEY:
        os.environ.setdefault("LANGCHAIN_API_KEY", s.LANGCHAIN_API_KEY)


_apply_langsmith_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Avabodh API...")
    init_db()
    logger.info("Database tables verified.")

    # If this fires in a real deployment, the signed preview file links
    # (GET /documents/{id}/file) are forgeable by anyone who's read this
    # codebase — the default is deliberately public knowledge, not a
    # real secret. This doesn't block startup (a broken preview link
    # feature is less bad than the app refusing to boot), but it's loud
    # on purpose. Set SECRET_KEY in your environment before going live.
    if settings.SECRET_KEY == _DEFAULT_SECRET_KEY:
        logger.warning(
            "SECURITY WARNING: SECRET_KEY is still the default dev value. "
            "Signed document preview links can be forged by anyone. "
            "Set a real SECRET_KEY in your environment before production use."
        )

    # If this fires, the app is connecting to Postgres as a role whose
    # password is a publicly-documented placeholder — anyone who's read
    # this codebase (or this repo, if it's ever public) could log in as
    # avabodh_app directly. Row-Level Security still wouldn't let that
    # login see other tenants' data (RLS applies to this role by design),
    # but it would still let someone impersonate the application itself
    # against whichever tenant/org-unit they choose to set. Set a real
    # APP_DB_PASSWORD in your environment before production use.
    if settings.APP_DB_PASSWORD == _DEFAULT_APP_DB_PASSWORD:
        logger.warning(
            "SECURITY WARNING: APP_DB_PASSWORD is still the default placeholder value. "
            "Set a real APP_DB_PASSWORD in your environment before production use."
        )

    yield
    logger.info("Shutting down Avabodh API...")


app = FastAPI(
    title="Avabodh API",
    description="Document Intelligence API — Upload documents, get AI summaries.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Order matters here: LoggingMiddleware must be added LAST so it ends up
# outermost — that way every request, including ones TenantGuardMiddleware
# rejects with a 400, still shows up in the standard access log with its
# real status code, not just the guard's own warning line.
app.add_middleware(TenantGuardMiddleware)
app.add_middleware(LoggingMiddleware)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(web_router, prefix="/web", tags=["Web Scraping"])
app.include_router(kb.router, prefix="/kb", tags=["Knowledge Base"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": "Avabodh API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }