"""
pipeline/storage.py
-------------------
Rewritten 2026-08-21 (Qdrant migration) — slimmed to document-registry
operations only. Chunk storage/embedding is gone from here entirely (see
pipeline/vector_store.py, called from pipeline/ingest.py). This module now
only manages the Document row's lifecycle:

  register_document() — insert a new UPLOADED-status row (or return the
                         existing one if this exact file was already
                         registered for this tenant+org_unit — dedup)
  check_duplicate()    — the dedup lookup register_document() uses
  set_status()          — PROCESSING -> READY | FAILED, called by
                         pipeline/ingest.py at each stage
  apply_summary()       — write summary_text + LLM-extracted metadata
                         once summarisation finishes

Both dedup lookups are scoped by (tenant_id, org_unit_id) together — never
either one alone. org_unit_id is a hard boundary WITHIN a tenant (department-
level), so a standalone org_unit_id match would leak across companies if two
different tenants happen to use the same department code.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import make_transient

from db.database import get_db_session_context
from db.models import Document, DocumentMetadataOutput
from utils.logger import get_logger

logger = get_logger(__name__)


def check_duplicate(file_hash: str, tenant_id: str, org_unit_id: str) -> Optional[Document]:
    """Existing Document with this exact hash, scoped to tenant+org_unit — or None."""
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        existing = (
            session.query(Document)
            .filter(
                Document.doc_hash == file_hash,
                Document.tenant_id == tenant_id,
                Document.org_unit_id == org_unit_id,
            )
            .first()
        )
        if existing:
            session.expunge(existing)
            make_transient(existing)
            return existing
        return None


def register_document(
    doc_name: str,
    file_hash: str,
    tenant_id: str,
    org_unit_id: str,
    file_type: Optional[str] = None,
    file_size: Optional[int] = None,
    source: str = "upload",
    source_path: Optional[str] = None,
    stored_path: Optional[str] = None,
    category: Optional[str] = None,
    effective_from: Optional[datetime] = None,
    effective_to: Optional[datetime] = None,
    is_ground_truth: bool = False,
    metadata: Optional[dict] = None,
) -> tuple[Document, bool]:
    """
    Register a new document (status=UPLOADED) — or, if same tenant+org_unit
    + hash already exists, return that existing row unchanged (dedup; the
    caller doesn't re-run ingestion for it). Returns (document, created).

    Same-name-but-different-hash (file updated) is NOT an implicit update
    here — that was the old pgvector-era upsert behavior for
    DocumentSummary; under the new lifecycle model a changed file is a
    new Document row with its own status/ingestion history, and the
    caller (api/routes/documents.py) decides whether to also soft-retire
    the old one. Keeping this function single-purpose (register, not
    upsert) avoids silently mutating a document that might still be mid-
    read by another request.
    """
    existing = check_duplicate(file_hash, tenant_id, org_unit_id)
    if existing:
        logger.info(
            "Document already registered for tenant=%s org_unit=%s hash=%s (status=%s) — no new row.",
            tenant_id, org_unit_id, file_hash[:12], existing.status,
        )
        return existing, False

    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        record = Document(
            tenant_id=tenant_id,
            org_unit_id=org_unit_id,
            doc_name=doc_name,
            doc_hash=file_hash,
            file_type=file_type,
            file_size=file_size,
            source=source,
            source_path=source_path,
            stored_path=stored_path,
            status="UPLOADED",
            category=category,
            effective_from=effective_from,
            effective_to=effective_to,
            is_ground_truth=is_ground_truth,
            metadata_json=metadata or {},
            summary_text="",   # filled by apply_summary() once ingestion runs
        )
        session.add(record)
        session.flush()
        logger.info("Registered new document tenant=%s org_unit=%s '%s' (id=%s)",
                    tenant_id, org_unit_id, doc_name, record.id)
        session.expunge(record)
        make_transient(record)
        return record, True


def get_summary_by_name(doc_name: str, tenant_id: str, org_unit_id: str) -> Optional[str]:
    """
    Look up a document's stored summary_text by exact doc_name, scoped to
    tenant+org_unit — used by pipeline/chat.py::generate_search_queries()
    to ground multi-query condensation in what the target document is
    actually about, when the chat request scopes to one document
    (request.doc_filter, name-based — the only handle the current chat API
    accepts). Returns None if no match or summary is empty — callers must
    treat that as "no summary available", not an error.

    doc_name is NOT guaranteed unique (dedup is by file hash, not name —
    two different Document rows can share a name) — get_summary_by_id()
    below is the exact-match alternative for callers that actually have a
    document_id in hand.
    """
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        doc = (
            session.query(Document)
            .filter(
                Document.doc_name == doc_name,
                Document.tenant_id == tenant_id,
                Document.org_unit_id == org_unit_id,
            )
            .first()
        )
        return doc.summary_text if doc and doc.summary_text else None


def get_summary_by_id(document_id, tenant_id: str, org_unit_id: str) -> Optional[str]:
    """
    Same lookup as get_summary_by_name(), but by primary key — exact match,
    no name-collision risk. Not wired to any call site yet: the current
    chat request schema only carries doc_filter (name), not a document_id
    — this exists so a future caller that DOES have one (e.g. Qdrant chunk
    payloads already carry document_id on every point) doesn't need a new
    Postgres query pattern invented from scratch.
    """
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        doc = (
            session.query(Document)
            .filter(
                Document.id == document_id,
                Document.tenant_id == tenant_id,
                Document.org_unit_id == org_unit_id,
            )
            .first()
        )
        return doc.summary_text if doc and doc.summary_text else None


def set_status(document_id, tenant_id: str, org_unit_id: str, status: str, status_detail: Optional[str] = None) -> None:
    """PROCESSING -> READY | FAILED (or UPLOADED -> PROCESSING). Called by pipeline/ingest.py at each stage."""
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        doc = session.query(Document).filter(
            Document.id == document_id,
            Document.tenant_id == tenant_id,
            Document.org_unit_id == org_unit_id,
        ).first()
        if not doc:
            logger.warning("set_status: document %s not found for tenant=%s org_unit=%s", document_id, tenant_id, org_unit_id)
            return
        doc.status = status
        doc.status_detail = status_detail
        if status in ("READY", "FAILED"):
            doc.processed_at = datetime.now(timezone.utc)
        session.add(doc)


def set_stored_path(document_id, tenant_id: str, org_unit_id: str, stored_path: str) -> None:
    """
    Repoint a document at where its ORIGINAL file now lives.

    2026-08-23 — used by api/routes/documents.py after pushing an upload to
    S3: the row is created with the local scratch path (so the object key
    can be built from a real document id), then rewritten to the
    's3://bucket/key' URI once the object is durably stored. Anything not
    starting with 's3://' is still a local path, which is why documents
    ingested before S3 existed keep working untouched.

    NOT non-fatal, unlike its neighbours here: if this write is lost the row
    keeps pointing at a scratch file that ingestion is about to delete, and
    the document becomes unservable and unreprocessable while its bytes sit
    in the bucket unreferenced. The caller treats a failure as a failed
    upload.
    """
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        doc = session.query(Document).filter(
            Document.id == document_id,
            Document.tenant_id == tenant_id,
            Document.org_unit_id == org_unit_id,
        ).first()
        if not doc:
            raise RuntimeError(f"set_stored_path: document {document_id} not found for tenant={tenant_id} org_unit={org_unit_id}")
        doc.stored_path = stored_path
        session.add(doc)


def set_image_count(document_id, tenant_id: str, org_unit_id: str, image_count: int) -> None:
    """
    Called by pipeline/ingest.py after images are indexed into Qdrant —
    without this, Document.image_count stayed permanently 0 regardless of
    how many role='image' points actually got upserted (confirmed via
    real testing: 48 image chunks in Qdrant, image_count still reporting
    0 through the API). Non-fatal by design, same as the rest of the
    image pipeline — a failure here shouldn't fail ingestion.
    """
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        doc = session.query(Document).filter(
            Document.id == document_id,
            Document.tenant_id == tenant_id,
            Document.org_unit_id == org_unit_id,
        ).first()
        if not doc:
            logger.warning("set_image_count: document %s not found for tenant=%s org_unit=%s", document_id, tenant_id, org_unit_id)
            return
        doc.image_count = image_count
        session.add(doc)


def apply_summary(
    document_id,
    tenant_id: str,
    org_unit_id: str,
    summary_text: str,
    key_topics: list[str],
    chunk_count: int,
    page_count: int,
    model_used: str,
    document_metadata: Optional[DocumentMetadataOutput] = None,
    document_type_override: Optional[str] = None,
) -> None:
    """
    Write summarisation results onto an already-registered Document.
    Non-fatal by design (called from pipeline/ingest.py): a summarisation
    failure sets metadata_status='failed' here but does NOT fail the whole
    ingest job — chunk indexing (the retrieval-critical part) can succeed
    independently.

    document_type_override: used by the standalone-image ingestion branch
    (pipeline/ingest.py::_process_standalone_image) to force
    document_type='image' per IMPLEMENTATION_PLAN (2).md §4 — that path
    never runs the LLM metadata-extraction call (document_metadata is
    always None there), so document_type would otherwise never get set.
    """
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        doc = session.query(Document).filter(
            Document.id == document_id,
            Document.tenant_id == tenant_id,
            Document.org_unit_id == org_unit_id,
        ).first()
        if not doc:
            logger.warning("apply_summary: document %s not found for tenant=%s org_unit=%s", document_id, tenant_id, org_unit_id)
            return

        doc.summary_text = summary_text
        doc.key_topics = ", ".join(key_topics) if key_topics else None
        doc.chunk_count = chunk_count
        doc.page_count = page_count
        doc.model_used = model_used
        doc.embedding_model = None   # set separately once chunks are actually embedded/indexed
        if document_type_override:
            doc.document_type = document_type_override

        if document_metadata is None:
            doc.metadata_status = "completed" if document_type_override else "failed"
        else:
            doc.title = document_metadata.title or None
            doc.author = document_metadata.author
            doc.document_type = document_metadata.document_type
            doc.domain = document_metadata.domain
            doc.detected_language = document_metadata.detected_language
            doc.key_entities = document_metadata.key_entities
            doc.mentioned_dates = document_metadata.mentioned_dates
            doc.target_audience = document_metadata.target_audience
            doc.sentiment = document_metadata.sentiment
            doc.confidentiality_level = document_metadata.confidentiality_level
            doc.metadata_status = "completed"
            doc.metadata_extracted_at = datetime.now(timezone.utc)

        session.add(doc)
        logger.info("Applied summary to document %s (tenant=%s org_unit=%s, metadata_status=%s)",
                    document_id, tenant_id, org_unit_id, doc.metadata_status)
