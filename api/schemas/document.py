"""
api/schemas/document.py
-----------------------
Pydantic schemas for request and response validation.

NEW: org_unit_id, category, effective_from/to, is_ground_truth, and
preview_link added to all document response schemas. title is
deliberately NOT a client-supplied field anywhere — it stays
LLM-generated only (see pipeline/storage.py).
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class DocumentUploadResponse(BaseModel):
    id:           UUID
    doc_name:     str
    summary_text: str
    key_topics:   Optional[str] = None
    page_count:   int
    chunk_count:  int
    source_path:  Optional[str] = None
    language:     str
    model_used:   Optional[str] = None
    doc_hash:     Optional[str] = None
    tenant_id:    Optional[str] = None   # which tenant this document belongs to
    org_unit_id:  Optional[str] = None   # NEW — which department, within that tenant
    category:        Optional[str] = None   # NEW — client-defined free text
    effective_from:  Optional[datetime] = None   # NEW — metadata only, no retrieval effect
    effective_to:    Optional[datetime] = None   # NEW — metadata only, no retrieval effect
    is_ground_truth: Optional[bool] = None       # NEW — gates retrieval
    title:                  Optional[str] = None
    author:                 Optional[str] = None
    document_type:          Optional[str] = None
    domain:                 Optional[str] = None
    key_entities:           Optional[list[str]] = None
    mentioned_dates:        Optional[list[str]] = None
    target_audience:        Optional[str] = None
    sentiment:               Optional[str] = None
    confidentiality_level:  Optional[str] = None
    metadata_status:         Optional[str] = None
    image_count:             Optional[int] = 0
    preview_link:            Optional[str] = None   # NEW — signed file link or source URL
    # NEW 2026-08-21 (Qdrant migration) — background ingestion lifecycle.
    status:        Optional[str] = None   # UPLOADED | PROCESSING | READY | FAILED
    status_detail: Optional[str] = None   # error detail when status=FAILED
    metadata:      Optional[dict] = None  # client-supplied metadata (full payload)
    created_at:   datetime
    updated_at:   Optional[datetime] = None
    elapsed_sec:  Optional[float] = None

    class Config:
        from_attributes = True


class DocumentListItem(BaseModel):
    id:          UUID
    doc_name:    str
    page_count:  int
    chunk_count: int
    model_used:  Optional[str] = None
    document_type: Optional[str] = None
    domain:        Optional[str] = None
    category:      Optional[str] = None    # NEW
    is_ground_truth: Optional[bool] = None  # NEW
    image_count:   Optional[int] = 0
    status:        Optional[str] = None    # NEW 2026-08-21
    created_at:  datetime
    summary_preview: Optional[str] = None

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    total:     int
    page:      int
    per_page:  int
    documents: list[DocumentListItem]


class DocumentDetailResponse(BaseModel):
    """
    Also serves as the preview payload — no separate preview endpoint.
    Everything a preview UI needs is already here: summary, title,
    tenant_id, org_unit_id, category, effective dates, ground-truth flag,
    and preview_link (signed file link for uploads, source URL for
    web-scraped documents).
    """
    id:           UUID
    doc_name:     str
    summary_text: str
    key_topics:   Optional[str] = None
    page_count:   int
    chunk_count:  int
    source_path:  Optional[str] = None
    language:     str
    model_used:   Optional[str] = None
    doc_hash:     Optional[str] = None
    tenant_id:    Optional[str] = None
    org_unit_id:  Optional[str] = None
    category:        Optional[str] = None
    effective_from:  Optional[datetime] = None
    effective_to:    Optional[datetime] = None
    is_ground_truth: Optional[bool] = None
    title:                  Optional[str] = None
    author:                 Optional[str] = None
    document_type:          Optional[str] = None
    domain:                 Optional[str] = None
    key_entities:           Optional[list[str]] = None
    mentioned_dates:        Optional[list[str]] = None
    target_audience:        Optional[str] = None
    sentiment:               Optional[str] = None
    confidentiality_level:  Optional[str] = None
    metadata_status:         Optional[str] = None
    image_count:             Optional[int] = 0
    preview_link:            Optional[str] = None
    status:        Optional[str] = None   # NEW 2026-08-21
    status_detail: Optional[str] = None   # NEW 2026-08-21
    metadata:      Optional[dict] = None  # NEW 2026-08-21
    created_at:   datetime
    updated_at:   Optional[datetime] = None

    class Config:
        from_attributes = True


class DocumentUpdateRequest(BaseModel):
    doc_name: str = Field(min_length=1, max_length=512)

    @field_validator("doc_name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        return v.strip()


class DocumentUpdateResponse(BaseModel):
    id:         UUID
    doc_name:   str
    updated_at: Optional[datetime] = None
    message:    str = "Document updated successfully"

    class Config:
        from_attributes = True


class ReprocessResponse(BaseModel):
    id:     UUID
    status: str = "PROCESSING"


class DeleteResponse(BaseModel):
    id:      UUID
    message: str = "Document deleted successfully"


class ChunkMetadataResponse(BaseModel):
    section_heading:     Optional[str] = None
    chunk_type:           Optional[str] = None
    topic:                 Optional[str] = None
    entities:               Optional[list[str]] = None
    contains_data:          Optional[bool] = None
    metadata_confidence:    Optional[float] = None

    class Config:
        from_attributes = True


class ErrorResponse(BaseModel):
    error:   str
    detail:  Optional[str] = None
    status:  int


class UploadErrorResponse(BaseModel):
    error:     str
    step:      str
    detail:    Optional[str] = None
    status:    int = 500


class ProcessingStatus(BaseModel):
    step:        str
    message:     str
    progress:    int
    completed:   bool = False