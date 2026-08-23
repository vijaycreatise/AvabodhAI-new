"""
scripts/init_qdrant.py
------------------------
Standalone Qdrant provisioning script — the vector-store counterpart to
scripts/init_schema.py. Run this once against ANY Qdrant instance (local
install, self-hosted service, whatever your team runs) to create the
avabodh_chunks and avabodh_chat_messages collections plus their payload
indexes.

Not Docker-managed by this project — same "no permanent data store owned
by docker-compose" rule as Postgres. Point QDRANT_URL/QDRANT_API_KEY (see
config/settings.py) at your own, separately-hosted Qdrant instance (never
shared with clariona-core's), then run:

    python scripts/init_qdrant.py

Safe to re-run — ensure_collections() only creates what doesn't already
exist.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
# Run as `python scripts/init_qdrant.py` — sys.path[0] defaults to
# scripts/ itself, not the repo root, so config/pipeline/utils wouldn't be
# importable without this (same pattern cli.py uses at the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from pipeline.vector_store import ensure_collections
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    print(f"Provisioning Qdrant collections on {settings.QDRANT_URL} ...")

    try:
        ensure_collections()
    except Exception as e:
        print(f"✗ Qdrant provisioning failed: {e}")
        print("  Common causes: Qdrant instance unreachable at QDRANT_URL, or QDRANT_API_KEY")
        print("  required but not set/incorrect.")
        sys.exit(1)

    print(f"✓ Qdrant provisioned: collections '{settings.QDRANT_COLLECTION}' and "
          f"'{settings.QDRANT_CHAT_COLLECTION}' with their payload indexes are all in place.")


if __name__ == "__main__":
    main()
