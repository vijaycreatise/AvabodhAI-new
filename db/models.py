"""
db/models.py
------------
Four ORM tables:
1. DocumentSummaryOutput  — Pydantic LLM output validator
2. DocumentSummary        — document summaries table (+ document-level metadata)
3. DocumentChunk          — pgvector chunk embeddings table (+ chunk-level metadata)
4. ChatThread             — chat conversation threads
5. ChatMessage            — individual chat messages with embeddings
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Column, DateTime, Integer, String, Text, Float,
    Index, ForeignKey, JSON, Boolean, Enum
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pydantic — LLM output validation
# ─────────────────────────────────────────────────────────────────────────────

class DocumentSummaryOutput(BaseModel):
    doc_name:         str   = Field(default="unknown")
    summary_text:     str   = Field(min_length=10)
    key_topics:       list[str] = Field(default_factory=list)
    page_count:       int   = Field(default=0, ge=0)
    chunk_count:      int   = Field(default=0, ge=0)
    source_path:      str   = Field(default="")
    language:         str   = Field(default="English")
    model_used:       str   = Field(default="")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("summary_text", mode="before")
    @classmethod
    def clean_summary(cls, v: str) -> str:
        return " ".join(v.split()) if isinstance(v, str) else v

    @field_validator("doc_name", mode="before")
    @classmethod
    def sanitise_doc_name(cls, v: str) -> str:
        import os
        return os.path.basename(str(v)) if v else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# NEW — Pydantic schemas for metadata extraction (LLM structured output)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentMetadataOutput(BaseModel):
    """
    Document-level metadata — extracted once per document via LLM,
    using structured output during/after the Reduce step.
    """
    title:                str  = Field(default="", description="Real document title, not filename")
    author:               Optional[str] = Field(default=None, description="Author if mentioned in document")
    document_type:        str  = Field(default="other",
                                       description="resume, research_paper, contract, report, invoice, manual, article, other")
    domain:                str  = Field(default="general",
                                       description="legal, technical, financial, academic, medical, general")
    detected_language:    str  = Field(default="English")
    key_entities:          list[str] = Field(default_factory=list, description="Organizations, people, locations mentioned")
    mentioned_dates:       list[str] = Field(default_factory=list, description="Any dates referenced in the document")
    target_audience:       str  = Field(default="general", description="technical, general, executive")
    sentiment:              str  = Field(default="neutral", description="positive, negative, neutral, critical")
    confidentiality_level: str  = Field(default="public", description="public, internal, confidential")

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, v) -> str:
        return v.strip()[:512] if isinstance(v, str) else ""

    @field_validator("document_type", "domain", "target_audience", "sentiment", "confidentiality_level", mode="before")
    @classmethod
    def lowercase_enum(cls, v) -> str:
        return str(v).strip().lower() if v else "other"


class ChunkMetadataOutput(BaseModel):
    """
    Chunk-level metadata — extracted during the Map step alongside
    the existing summary, using the SAME LLM call (no extra cost).
    """
    summary:          str  = Field(min_length=1, description="Concise summary of this section")
    section_heading:  Optional[str] = Field(default=None, description="Section/heading this chunk likely belongs to")
    chunk_type:       str  = Field(default="paragraph", description="paragraph, table, list, heading, code")
    topic:             str  = Field(default="general", description="2-3 word topic label for this chunk")
    entities:           list[str] = Field(default_factory=list, description="Named entities mentioned in this chunk")
    contains_data:      bool = Field(default=False, description="True if chunk has numbers, statistics, or tabular data")
    confidence_score:   float = Field(default=0.8, ge=0.0, le=1.0, description="LLM confidence in this extraction")

    @field_validator("chunk_type", mode="before")
    @classmethod
    def validate_chunk_type(cls, v) -> str:
        allowed = {"paragraph", "table", "list", "heading", "code"}
        v = str(v).strip().lower() if v else "paragraph"
        return v if v in allowed else "paragraph"

    @field_validator("summary", "topic", mode="before")
    @classmethod
    def clean_text_field(cls, v) -> str:
        return " ".join(str(v).split()) if v else ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. SQLAlchemy Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. DocumentSummary table — with document-level metadata columns
# ─────────────────────────────────────────────────────────────────────────────

class DocumentSummary(Base):
    __tablename__ = "document_summaries"

    id                  = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # ── Multi-tenancy — REQUIRED on every row, set from the X-Tenant-ID
    # header at write time. Every read in this app must filter on this. ──
    tenant_id           = Column(String(128), nullable=False, index=True)
    # ── Department-level isolation WITHIN a tenant — REQUIRED, set from
    # X-Org-Unit-ID header, same trust model as tenant_id. ALWAYS filtered
    # together with tenant_id, never alone (see composite indexes below —
    # a standalone org_unit_id match would leak across tenants if two
    # different companies happen to use the same department code). ──────
    org_unit_id          = Column(String(128), nullable=False, index=True)
    doc_name            = Column(String(512), nullable=False, index=True)
    summary_text        = Column(Text, nullable=False)
    key_topics          = Column(Text, nullable=True)
    page_count          = Column(Integer, default=0)
    chunk_count         = Column(Integer, default=0)
    source_path         = Column(String(1024), nullable=True)
    language            = Column(String(64), default="English")
    model_used          = Column(String(128), nullable=True)
    confidence          = Column(Float, nullable=True)
    doc_hash            = Column(String(64), nullable=True, index=True)
    avg_chunk_size      = Column(Integer, nullable=True)
    embedding_model     = Column(String(128), nullable=True)
    embedding_status    = Column(String(32), default="pending")
    embedding_stored_at = Column(DateTime(timezone=True), nullable=True)

    # ── NEW — Document-level extracted metadata ───────────────────────────────
    title                  = Column(String(512), nullable=True, index=True)
    author                 = Column(String(256), nullable=True)
    document_type          = Column(String(64), nullable=True, index=True)   # resume/contract/report/etc
    domain                 = Column(String(64), nullable=True, index=True)   # legal/technical/financial/etc
    detected_language      = Column(String(64), nullable=True)
    key_entities            = Column(ARRAY(String), nullable=True)            # ["Avik Bhattacharya", "IIT Bombay"]
    mentioned_dates         = Column(ARRAY(String), nullable=True)
    target_audience         = Column(String(64), nullable=True)
    sentiment                = Column(String(32), nullable=True)
    confidentiality_level   = Column(String(32), nullable=True, default="public")
    metadata_status          = Column(String(32), default="pending")          # pending/completed/failed
    metadata_extracted_at   = Column(DateTime(timezone=True), nullable=True)

    # ── Image tracking ────────────────────────────────────────────────────────
    image_count             = Column(Integer, default=0)   # how many images extracted and embedded

    # ── NEW — client-supplied document fields (Upload Knowledge Document form) ──
    category       = Column(String(128), nullable=True, index=True)   # client-defined, free text: "Policy", "Guide"
    effective_from = Column(DateTime(timezone=True), nullable=True)   # validity window start — metadata only,
    effective_to   = Column(DateTime(timezone=True), nullable=True)   # no retrieval effect (by design — confirmed)
    is_ground_truth = Column(Boolean, nullable=False, default=False)  # gates retrieval — see DocumentChunk.is_ground_truth

    created_at          = Column(DateTime(timezone=True),
                                 default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at          = Column(DateTime(timezone=True),
                                 default=lambda: datetime.now(timezone.utc),
                                 onupdate=lambda: datetime.now(timezone.utc))

    chunks = relationship("DocumentChunk", back_populates="summary",
                          cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_doc_summaries_source_path", "source_path"),
        Index("ix_doc_summaries_doc_type_domain", "document_type", "domain"),
        # Tenant-scoped dedup lookups (check_duplicate / same_name upsert
        # in pipeline/storage.py) hit these directly.
        Index("ix_doc_summaries_tenant_hash", "tenant_id", "doc_hash"),
        Index("ix_doc_summaries_tenant_name", "tenant_id", "doc_name"),
        # NEW — org_unit_id is a hard boundary WITHIN a tenant, so every
        # lookup that used to be (tenant_id, X) becomes (tenant_id,
        # org_unit_id, X) — never org_unit_id alone (see column comment).
        Index("ix_doc_summaries_tenant_org_hash", "tenant_id", "org_unit_id", "doc_hash"),
        Index("ix_doc_summaries_tenant_org_name", "tenant_id", "org_unit_id", "doc_name"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. DocumentChunk table — pgvector — with chunk-level metadata columns
# ─────────────────────────────────────────────────────────────────────────────

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # ── Multi-tenancy — denormalized from the parent DocumentSummary at
    # insert time. Filtering here directly (rather than joining through
    # summary_id) keeps the pgvector similarity query fast and simple:
    # WHERE tenant_id = :tenant_id ORDER BY embedding <=> :vector ──────────
    tenant_id       = Column(String(128), nullable=False, index=True)
    # ── Department-level isolation, denormalized same as tenant_id — see
    # DocumentSummary.org_unit_id comment. ALWAYS paired with tenant_id. ──
    org_unit_id     = Column(String(128), nullable=False, index=True)
    summary_id      = Column(UUID(as_uuid=True),
                             ForeignKey("document_summaries.id", ondelete="CASCADE"),
                             nullable=False, index=True)
    doc_hash        = Column(String(64), nullable=False, index=True)
    doc_name        = Column(String(512), nullable=False, index=True)
    chunk_index     = Column(Integer, nullable=False)
    total_chunks    = Column(Integer, nullable=False)
    chunk_text      = Column(Text, nullable=False)
    chunk_size      = Column(Integer, nullable=True)
    page_number     = Column(Integer, nullable=True)
    source_path     = Column(String(1024), nullable=True)
    language        = Column(String(64), default="English")
    embedding       = Column(Vector(1536), nullable=False)
    embedding_model = Column(String(128), nullable=True)

    # ── Chunk-level extracted metadata ───────────────────────────────────────
    section_heading   = Column(String(512), nullable=True, index=True)
    chunk_type         = Column(String(32), nullable=True, default="paragraph")
    topic               = Column(String(128), nullable=True, index=True)
    entities             = Column(ARRAY(String), nullable=True)
    contains_data        = Column(Boolean, default=False)
    metadata_confidence  = Column(Float, nullable=True)

    # ── Image columns (role='image' chunks only) ──────────────────────────────
    # role='text' for normal text chunks (default), role='image' for image chunks
    role               = Column(String(16), nullable=False, default="text")
    image_url          = Column(String(2048), nullable=True)   # original URL (web) or NULL (PDF)
    image_bytes_hash   = Column(String(64),  nullable=True, index=True)  # SHA-256 of raw bytes — dedup
    image_format       = Column(String(16),  nullable=True)   # jpeg/png/webp/gif
    image_width        = Column(Integer,     nullable=True)
    image_height       = Column(Integer,     nullable=True)
    image_size_bytes   = Column(Integer,     nullable=True)
    image_caption      = Column(Text,        nullable=True)   # full GPT-4o Vision caption
    image_type         = Column(String(32),  nullable=True)   # chart/table/diagram/photo/screenshot/infographic/other
    image_alt_text     = Column(Text,        nullable=True)   # HTML alt attr (web) or suggested (PDF)
    image_context      = Column(Text,        nullable=True)   # surrounding paragraph text
    contains_chart     = Column(Boolean,     nullable=True, default=False)
    contains_table     = Column(Boolean,     nullable=True, default=False)
    contains_text_img  = Column(Boolean,     nullable=True, default=False)  # infographic with embedded text
    key_elements       = Column(ARRAY(String), nullable=True)  # ["bar chart", "Q4 label"]
    vision_model_used  = Column(String(64),  nullable=True)   # gpt-4o
    vision_confidence  = Column(Float,       nullable=True)   # 0.0-1.0

    # ── NEW — gates retrieval. Denormalized from DocumentSummary at insert
    # time so vector_search() can filter WHERE is_ground_truth = true
    # directly, same pattern as tenant_id/role. No user-facing toggle at
    # query time — this is always enforced, not optional (confirmed: only
    # ground-truth documents are ever retrievable by chat/search). ────────
    is_ground_truth    = Column(Boolean,     nullable=False, default=False)

    # ── NEW — full-text search. Nullable and NOT set anywhere in Python —
    # a Postgres trigger (see db/database.py init_db()) populates this
    # automatically from chunk_text on every INSERT/UPDATE. This is a
    # second, independent way to search this table (exact keyword match)
    # alongside the existing `embedding` column (semantic similarity).
    # See pipeline/retriever.py for how the two get combined (hybrid
    # search via Reciprocal Rank Fusion).
    chunk_tsvector     = Column(TSVECTOR,    nullable=True)

    created_at      = Column(DateTime(timezone=True),
                             default=lambda: datetime.now(timezone.utc), nullable=False)

    summary = relationship("DocumentSummary", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_doc_hash_index", "doc_hash", "chunk_index"),
        Index("ix_chunks_topic", "topic"),
        Index("ix_chunks_role", "role"),
        Index("ix_chunks_image_bytes_hash", "image_bytes_hash"),
        # Every retrieval query filters WHERE tenant_id = ... AND
        # org_unit_id = ... AND is_ground_truth = true, all together —
        # this composite index covers that exact query shape directly.
        Index("ix_chunks_tenant_org_gt_role", "tenant_id", "org_unit_id", "is_ground_truth", "role"),
        # NEW — tenant+org-scoped image dedup (check_image_hash_exists)
        Index("ix_chunks_tenant_org_image_hash", "tenant_id", "org_unit_id", "image_bytes_hash"),
        # NEW — GIN index for full-text search. postgresql_using="gin" is
        # what makes this a proper FTS index rather than a useless btree
        # on a tsvector column (btree can't do @@ containment lookups).
        Index("document_chunks_tsvector_idx", "chunk_tsvector", postgresql_using="gin"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. ChatThread
# ─────────────────────────────────────────────────────────────────────────────

class ChatThread(Base):
    __tablename__ = "chat_threads"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # ── Multi-tenancy — same rule as documents: every thread belongs to
    # exactly one tenant, set from X-Tenant-ID when the thread is created. ──
    tenant_id  = Column(String(128), nullable=False, index=True)
    # ── Department-level isolation, same rule as documents. ─────────────
    org_unit_id = Column(String(128), nullable=False, index=True)
    title      = Column(String(512), nullable=True)
    user_id    = Column(String(256), nullable=True)
    doc_filter = Column(String(512), nullable=True)
    message_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    messages = relationship("ChatMessage", back_populates="thread",
                            cascade="all, delete-orphan",
                            order_by="ChatMessage.created_at")

    __table_args__ = (
        Index("ix_chat_threads_user_id", "user_id"),
        # Thread list/get/patch/delete all filter WHERE tenant_id = ...
        # AND org_unit_id = ... together.
        Index("ix_chat_threads_tenant_org_user", "tenant_id", "org_unit_id", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<ChatThread id={self.id} title={self.title}>"


# ─────────────────────────────────────────────────────────────────────────────
# 6. ChatMessage
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id        = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # ── Multi-tenancy — denormalized from the parent ChatThread. Matters
    # most for the /chat/search semantic-search-over-history endpoint,
    # which queries chat_messages directly and must not cross tenants. ────
    tenant_id = Column(String(128), nullable=False, index=True)
    # ── Department-level isolation, denormalized same as tenant_id. ─────
    org_unit_id = Column(String(128), nullable=False, index=True)
    thread_id = Column(UUID(as_uuid=True),
                       ForeignKey("chat_threads.id", ondelete="CASCADE"),
                       nullable=False, index=True)

    role      = Column(String(16), nullable=False)
    content   = Column(Text, nullable=False)

    embedding       = Column(Vector(1536), nullable=True)
    embedding_model = Column(String(128), nullable=True)

    sources   = Column(JSON, nullable=True)

    # ── NEW — multimodal chat (image attached to this turn) ────────────────
    has_image     = Column(Boolean, nullable=False, default=False)
    image_caption = Column(Text, nullable=True)   # GPT-4o Vision caption of the attached image

    prompt_tokens     = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc), nullable=False)

    thread = relationship("ChatThread", back_populates="messages")

    __table_args__ = (
        Index("ix_chat_messages_thread_role", "thread_id", "role"),
        # GET /chat/search filters WHERE tenant_id = ... AND org_unit_id = ...
        Index("ix_chat_messages_tenant_org", "tenant_id", "org_unit_id"),
    )

    def __repr__(self) -> str:
        return f"<ChatMessage role={self.role} thread={self.thread_id}>"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Knowledge Base (KB) Tables
# ─────────────────────────────────────────────────────────────────────────────

class KBDocument(Base):
    __tablename__ = "kb_documents"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id      = Column(String(256), nullable=False, index=True)
    file_name     = Column(String(512), nullable=False)
    status        = Column(
        Enum("processing", "ready", "failed", name="kb_document_status"),
        nullable=False,
        default="processing",
    )
    error_message = Column(Text, nullable=True)
    created_at    = Column(DateTime(timezone=True),
                           default=lambda: datetime.now(timezone.utc), nullable=False)


class KBChatHistory(Base):
    __tablename__ = "kb_chat_history"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(String(256), nullable=False, index=True)
    owner_id        = Column(String(256), nullable=False, index=True)
    role            = Column(
        Enum("human", "ai", name="kb_chat_role"),
        nullable=False,
    )
    content         = Column(Text, nullable=False)
    created_at      = Column(DateTime(timezone=True),
                           default=lambda: datetime.now(timezone.utc), nullable=False)
