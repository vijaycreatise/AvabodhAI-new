"""
api/routes/search.py
--------------------
Rewritten 2026-08-21 (Qdrant migration). Same endpoint paths/methods;
response shapes are additive only (new fields, nothing removed) — see
IMPLEMENTATION_PLAN (2).md §3.

search_mode still controls "hybrid" (default, dense+sparse fused by
Qdrant via RRF) / "semantic" (dense only) / "keyword" (sparse only, via
fastembed SPLADE — replaces the old Postgres chunk_tsvector path).

New optional fields: org_ids, document_ids, filters (allowlisted metadata,
422 on an unknown key), as_of (effective-date filtering), is_ground_truth
(now an optional filter, not a hard gate), rerank (default True).
"""

import uuid
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from api.dependencies import get_tenant_id, get_org_unit_id
from pipeline.retriever import build_filter, search as retriever_search
from pipeline import vector_store
from config.settings import get_settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

_ALLOWED_ROLES = {"text", "image"}
_ALLOWED_MODES = {"hybrid", "semantic", "keyword"}


class SearchRequest(BaseModel):
    query:    str = Field(min_length=1, description="Search query")
    top_k:    int = Field(default=5, ge=1, le=50, description="Number of results")
    doc_name: Optional[str] = Field(default=None)
    role:     Optional[str] = Field(default=None, description="'text' or 'image'. Omit to search both.")
    search_mode: str = Field(default="hybrid", description="'hybrid', 'semantic', or 'keyword'.")
    org_ids: Optional[list[str]] = Field(default=None, description="Widen search to multiple departments within your tenant. Defaults to the caller's own org unit.")
    document_ids: Optional[list[uuid.UUID]] = Field(default=None)
    filters: Optional[dict[str, list[str]]] = Field(default=None, description="Allowlisted metadata filters, e.g. {\"country\": [\"IN\"]}")
    as_of: Optional[date] = Field(default=None, description="Effective-date filter — only documents valid on this date")
    is_ground_truth: Optional[bool] = Field(default=None, description="Optional filter — omit to search all documents")
    rerank: bool = Field(default=True)

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().lower()
        if v not in _ALLOWED_ROLES:
            raise ValueError(f"role must be one of {sorted(_ALLOWED_ROLES)}")
        return v

    @field_validator("search_mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in _ALLOWED_MODES:
            raise ValueError(f"search_mode must be one of {sorted(_ALLOWED_MODES)}")
        return v


class ChunkResult(BaseModel):
    id:          str
    doc_name:    str
    chunk_index: int
    chunk_text:  str
    chunk_size:  int
    page_number: Optional[int] = None
    doc_hash:    Optional[str] = None
    similarity:  float
    search_type: Optional[str] = None
    matched_by:  Optional[list[str]] = None
    role:            str = "text"
    image_type:      Optional[str] = None
    image_caption:   Optional[str] = None
    image_url:       Optional[str] = None
    section_heading: Optional[str] = None
    topic:           Optional[str] = None
    chunk_type:      Optional[str] = None
    table_html:      Optional[str] = None   # original <table> markup, when this chunk contains a table
    # NEW 2026-08-21
    document_id:   Optional[str] = None
    chunk_id:      Optional[str] = None
    chunk_number:  Optional[int] = None
    score:         Optional[float] = None
    rerank_score:  Optional[float] = None
    org_unit_id:   Optional[str] = None
    effective_from: Optional[str] = None
    effective_to:   Optional[str] = None
    metadata:      Optional[dict] = None


class SearchResponse(BaseModel):
    query:       str
    search_mode: str
    results:     list[ChunkResult]
    total:       int


def _result_to_chunk(r: dict) -> ChunkResult:
    return ChunkResult(
        id=r["id"], doc_name=r.get("doc_name", ""), chunk_index=r.get("chunk_index", 0),
        chunk_text=r.get("chunk_text", ""), chunk_size=r.get("chunk_size", 0), page_number=r.get("page_number"),
        doc_hash=r.get("doc_hash"), similarity=r.get("similarity", 0.0),
        search_type=r.get("search_type"), matched_by=r.get("matched_by"),
        role=r.get("role") or "text", image_type=r.get("image_type"), image_caption=r.get("image_caption"),
        image_url=r.get("image_url"), section_heading=r.get("section_heading"), topic=r.get("topic"),
        chunk_type=r.get("chunk_type"), table_html=r.get("table_html"),
        document_id=r.get("document_id"), chunk_id=r.get("chunk_id", r.get("id")),
        chunk_number=r.get("chunk_index"), score=r.get("score"), rerank_score=r.get("rerank_score"),
        org_unit_id=r.get("org_unit_id"), effective_from=r.get("effective_from"), effective_to=r.get("effective_to"),
        metadata={k[5:]: v for k, v in r.items() if k.startswith("meta_")} or None,
    )


@router.post("/", response_model=SearchResponse, summary="Semantic, keyword, or hybrid search across documents")
async def semantic_search(
    payload: SearchRequest,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
):
    try:
        query_filter = build_filter(
            tenant_id=tenant_id, org_unit_id=org_unit_id, org_ids=payload.org_ids,
            document_ids=[str(d) for d in payload.document_ids] if payload.document_ids else None,
            doc_name=payload.doc_name, role=payload.role, is_ground_truth=payload.is_ground_truth,
            metadata=payload.filters, as_of=payload.as_of,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        # run_in_threadpool: same event-loop-freezing issue as
        # api/routes/chat.py (this repo runs uvicorn --workers 1) —
        # embedding + Qdrant + cross-encoder reranking are all blocking
        # calls; called directly here they'd stall the whole API, not
        # just this request, for the ~10-15s a rerank pass takes.
        results = await run_in_threadpool(
            retriever_search,
            query=payload.query, query_filter=query_filter, mode=payload.search_mode,
            top_k=payload.top_k, do_rerank=payload.rerank,
        )
    except Exception as e:
        # Phase H #6 — log full detail server-side, generic message to
        # the caller. Correlate via the X-Request-ID response header
        # (api/middleware/logging_middleware.py) and this log line.
        logger.exception("Search failed (query=%r, mode=%s): %s", payload.query, payload.search_mode, e)
        raise HTTPException(status_code=500, detail="Search failed. Contact support with the X-Request-ID response header if this persists.")

    return SearchResponse(
        query=payload.query, search_mode=payload.search_mode,
        results=[_result_to_chunk(r) for r in results], total=len(results),
    )


@router.get("/chunks/{doc_id}", summary="Get all chunks for a document")
async def get_document_chunks(
    doc_id: uuid.UUID,
    role: Optional[str] = None,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
):
    """
    2026-08-21: now served from Qdrant (vector_store.scroll_document_chunks)
    instead of the old document_chunks Postgres table (gone). Same response
    shape minus `has_tsvector` (no FTS column exists any more).
    """
    if role is not None:
        role = role.strip().lower()
        if role not in _ALLOWED_ROLES:
            raise HTTPException(status_code=422, detail=f"role must be one of {sorted(_ALLOWED_ROLES)}")

    all_chunks = vector_store.scroll_document_chunks(tenant_id=tenant_id, org_unit_id=org_unit_id, document_id=str(doc_id))
    if not all_chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for ID '{doc_id}'")

    text_count = sum(1 for c in all_chunks if c.get("role", "text") == "text")
    image_count = sum(1 for c in all_chunks if c.get("role") == "image")
    chunks = [c for c in all_chunks if role is None or c.get("role") == role]

    return {
        "doc_id": str(doc_id),
        "doc_name": all_chunks[0].get("doc_name"),
        "total_chunks": len(chunks),
        "text_chunks": text_count,
        "image_chunks": image_count,
        "chunks": [
            {
                "id": c["id"], "chunk_index": c.get("chunk_index"), "chunk_text": c.get("chunk_text"),
                "chunk_size": c.get("chunk_size"), "page_number": c.get("page_number"),
                "model": c.get("embedding_model"), "role": c.get("role") or "text",
                "image_type": c.get("image_type"), "image_caption": c.get("image_caption"),
                "contains_chart": c.get("contains_chart"), "contains_table": c.get("contains_table"),
                "vision_confidence": c.get("vision_confidence"), "is_ground_truth": c.get("is_ground_truth"),
                "section_heading": c.get("section_heading"), "topic": c.get("topic"), "chunk_type": c.get("chunk_type"),
                "table_html": c.get("table_html"),
                "created_at": c.get("created_at"),
            }
            for c in chunks
        ]
    }
