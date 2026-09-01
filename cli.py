"""
cli.py
------
Entry point — run from terminal or call from your API/backend.

Rewritten 2026-08-21 (Qdrant migration) — rewired to pipeline/ingest.py's
process_document() (runs synchronously here, no BackgroundTasks needed for
a CLI invocation) instead of the deleted pipeline/orchestrator.py.
--backfill-tsvector replaced by --check-qdrant (the FTS trigger it backfilled
no longer exists — chunk storage moved to Qdrant entirely).

Usage:
    python cli.py --file path/to/doc.pdf --tenant-id t1 --org-unit-id d1
    python cli.py --dir path/to/folder --tenant-id t1 --org-unit-id d1
    python cli.py --check-db
    python cli.py --check-qdrant
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'
import argparse
import hashlib
import json
from pathlib import Path

from db.database import init_db, check_db_connection
from pipeline.vector_store import ensure_collections
from pipeline.extractor import SUPPORTED_EXTENSIONS
from pipeline.storage import register_document
from pipeline.ingest import process_document
from utils.logger import get_logger

logger = get_logger(__name__)


def _file_hash(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _process_one(filepath: str, tenant_id: str, org_unit_id: str, ground_truth: bool, metadata: dict) -> bool:
    filepath = str(Path(filepath).resolve())
    doc_name = os.path.basename(filepath)
    ext = Path(filepath).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"✗ Skipped '{doc_name}': unsupported extension '{ext}'")
        return False

    file_hash = _file_hash(filepath)
    record, created = register_document(
        doc_name=doc_name, file_hash=file_hash, tenant_id=tenant_id, org_unit_id=org_unit_id,
        file_type=ext.lstrip("."), file_size=os.path.getsize(filepath), source="upload",
        source_path=filepath, stored_path=filepath, is_ground_truth=ground_truth, metadata=metadata,
    )
    if not created and record.status == "READY":
        print(f"↩ Skipped '{doc_name}' — already ingested (ID: {record.id})")
        return True

    process_document(
        str(record.id), tenant_id, org_unit_id, filepath, doc_name, file_hash, ground_truth, metadata,
    )
    # process_document sets status itself — re-read isn't needed for a CLI
    # print, its own logging already reports success/failure per-document.
    print(f"✓ Processed '{doc_name}' — ID: {record.id}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Avabodh — Document ingestion pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Process a single document")
    group.add_argument("--dir", type=str, help="Process all supported documents in a folder")
    group.add_argument("--check-db", action="store_true", help="Test Postgres connection")
    group.add_argument("--check-qdrant", action="store_true", help="Test Qdrant connection and ensure collections exist")
    parser.add_argument("--tenant-id", type=str, default=None, help="Tenant to attribute processed documents to. Required unless --check-db/--check-qdrant.")
    parser.add_argument("--org-unit-id", type=str, default=None, help="Department/org unit within that tenant. Required unless --check-db/--check-qdrant.")
    parser.add_argument("--ground-truth", action="store_true", help="Mark processed documents as ground truth (optional retrieval filter). Default False, matching the API upload default.")
    parser.add_argument("--metadata", type=str, default=None, help="JSON object of client metadata, e.g. '{\"country\":\"IN\"}'")

    args = parser.parse_args()

    if args.check_db:
        ok = check_db_connection()
        print("DB connection: OK" if ok else "DB connection: FAILED")
        sys.exit(0 if ok else 1)

    if args.check_qdrant:
        try:
            ensure_collections()
            print("Qdrant connection: OK — collections verified.")
            sys.exit(0)
        except Exception as e:
            print(f"Qdrant connection: FAILED — {e}")
            sys.exit(1)

    missing = []
    if not args.tenant_id:
        missing.append("--tenant-id")
    if not args.org_unit_id:
        missing.append("--org-unit-id")
    if missing:
        print(f"Error: {' and '.join(missing)} required (unless using --check-db/--check-qdrant)")
        sys.exit(1)

    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except Exception as e:
            print(f"Error: --metadata is not valid JSON: {e}")
            sys.exit(1)

    init_db()
    ensure_collections()

    if args.file:
        ok = _process_one(args.file, args.tenant_id, args.org_unit_id, args.ground_truth, metadata)
        sys.exit(0 if ok else 1)
    else:
        directory = Path(args.dir)
        if not directory.exists():
            print(f"Error: directory not found: {directory}")
            sys.exit(1)
        files = [f for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
        succeeded = failed = 0
        for f in files:
            if _process_one(str(f), args.tenant_id, args.org_unit_id, args.ground_truth, metadata):
                succeeded += 1
            else:
                failed += 1
        print(f"\nDone — {succeeded} processed | {failed} failed")
        sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
