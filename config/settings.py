"""
config/settings.py
------------------
Central configuration loaded from environment variables.
Never hardcode secrets — use .env file locally, secrets manager in production.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────
    # Ollama runs locally — no API key needed
    HUGGINGFACEHUB_API_TOKEN: str = ""   # keep empty, not used anymore

    MAP_MODEL: str = "llama3.2"
    REDUCE_MODEL: str = "llama3.2"
    EMBEDDING_MODEL: str = "nomic-embed-text"   # local embedding model for Ollama
    # ── pgvector
    EMBEDDING_DIMENSIONS: int = 1536    # text-embedding-3-small = 1536
                                     # text-embedding-3-large = 3072
    EMBEDDING_BATCH_SIZE: int = 100     # chunks per batch to OpenAI

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_DOCKER_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_CHAT_MODEL: str = "llama3.2"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_EMBEDDING_DIMENSIONS: int = 768

    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    LLM_PROVIDER: str = "auto"
    PYTHON_KB_INTERNAL_URL: str = "http://localhost:8000"

    MAP_MAX_TOKENS: int = 512
    REDUCE_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.0
    LLM_REQUEST_TIMEOUT: int = 120
    LLM_MAX_RETRIES: int = 3

    # ── Chunking ───────────────────────────────────────────────────────────
    CHUNK_BREAKPOINT_THRESHOLD: float = 0.85
    MAX_CHUNK_SIZE: int = 3000
    MIN_CHUNK_SIZE: int = 100

    # ── Summary ────────────────────────────────────────────────────────────
    SUMMARY_MAX_LENGTH: int = 2000
    SUMMARY_LANGUAGE: str = "English"

    # ── PostgreSQL ─────────────────────────────────────────────────────────
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "Avabodh"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres123"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30

    APP_DB_USER: str = "avabodh_app"
    APP_DB_PASSWORD: str = "CHANGE-ME-app-role-password"

    FTS_LANGUAGE: str = "english"
    HYBRID_RRF_K: int = 60

    DOCUMENTS_DIR: str = "./documents"
    SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".txt", ".docx", ".csv"]

    VISION_MODEL: str = "gpt-4o"
    IMAGE_MIN_SIZE_BYTES: int = 5000
    IMAGE_MAX_DIMENSION: int = 1024
    IMAGE_MAX_WORKERS: int = 4

    SECRET_KEY: str = "dev-only-CHANGE-ME-in-production"
    FILE_LINK_TTL_SECONDS: Optional[int] = None
    PUBLIC_BASE_URL: str = ""

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "avabodh.log"

    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "Avabodh Project"

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    def __init__(self, **data: Any):
        merged = self._load_env_file_data()
        merged.update(data)
        super().__init__(**merged)

    @classmethod
    def _load_env_file_data(cls) -> dict[str, Any]:
        paths = [
            Path(__file__).resolve().parents[1] / "local_demo" / ".env",
            Path(__file__).resolve().parents[1] / ".env",
        ]
        for env_path in paths:
            if env_path.exists():
                return cls._parse_env_file(env_path)
        return {}

    @staticmethod
    def _parse_env_file(env_path: Path) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip().strip('"').strip("'")
        return data

    @field_validator(
        "DB_PORT",
        "DB_POOL_SIZE",
        "DB_MAX_OVERFLOW",
        "DB_POOL_TIMEOUT",
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_BATCH_SIZE",
        "MAP_MAX_TOKENS",
        "REDUCE_MAX_TOKENS",
        "LLM_REQUEST_TIMEOUT",
        "LLM_MAX_RETRIES",
        "MAX_CHUNK_SIZE",
        "MIN_CHUNK_SIZE",
        "IMAGE_MIN_SIZE_BYTES",
        "IMAGE_MAX_DIMENSION",
        "IMAGE_MAX_WORKERS",
        "HYBRID_RRF_K",
        mode="before",
    )
    @classmethod
    def _blank_int_to_default(cls, value: Any) -> Any:
        if value in ("", None):
            return None
        return value

    def _ollama_health(self, base_url: str) -> bool:
        try:
            with urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=2) as response:
                return response.status == 200
        except (HTTPError, URLError, TimeoutError, OSError):
            return False

    @property
    def use_ollama(self) -> bool:
        if self.OPENAI_API_KEY:
            return False
        return self._ollama_health(self.ollama_url)

    @property
    def ollama_url(self) -> str:
        if os.path.exists("/.dockerenv"):
            return self.OLLAMA_DOCKER_BASE_URL
        return self.OLLAMA_BASE_URL

    @property
    def db_url(self) -> str:
        """Admin/superuser connection — used ONLY for schema bootstrap in init_db(). Not for regular queries."""
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def app_db_url(self) -> str:
        """Restricted app-role connection — used for all normal request-serving queries. RLS actually applies to this one."""
        return (
            f"postgresql+psycopg2://{self.APP_DB_USER}:{self.APP_DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def async_db_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — settings loaded once at startup."""
    return Settings()
