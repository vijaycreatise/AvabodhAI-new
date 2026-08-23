"""
pipeline/chat_storage.py
------------------------
Saves chat messages and threads to PostgreSQL.
Both human and AI messages saved as separate rows.
Message embeddings generated and stored for semantic search.

Multi-tenancy + department isolation: every function here takes BOTH
tenant_id and org_unit_id, and either stamps them onto a new row or
filters an existing lookup by both together. Thread ownership is
validated once (create_thread / get_thread), and downstream functions
that operate purely on thread_id (save_human_message, save_ai_message,
update_thread_title, increment_message_count) trust that the caller
already confirmed the thread belongs to this tenant_id+org_unit_id — but
they still stamp both onto every row they write, so a bad thread_id can
never result in a cross-tenant or cross-department row being created
even if the upstream check were ever skipped.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy.orm import make_transient

from db.models import ChatThread, ChatMessage
from db.database import get_db_session_context
from pipeline import embedder, vector_store
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _index_message_in_qdrant(message_id, tenant_id: str, org_unit_id: str, thread_id: str, role: str, content: str) -> None:
    """
    2026-08-21: chat-message vectors moved to Qdrant's avabodh_chat_messages
    collection (ChatMessage.embedding/embedding_model columns removed —
    see db/models.py). Non-fatal — GET /chat/search just won't find this
    message if indexing fails, the message itself is still saved in Postgres.
    """
    try:
        dense_vector = embedder.embed_dense_query(content)
        vector_store.upsert_chat_message(
            message_id=str(message_id), tenant_id=tenant_id, org_unit_id=org_unit_id,
            thread_id=str(thread_id), role=role, content=content,
            dense_vector=dense_vector,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.warning("Chat message Qdrant indexing failed (non-fatal): %s", e)


def create_thread(
    tenant_id: str,
    org_unit_id: str,
    title: Optional[str] = None,
    user_id: Optional[str] = None,
    doc_filter: Optional[str] = None,
) -> ChatThread:
    """Create a new chat thread, stamped with tenant_id + org_unit_id, and return it."""
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        thread = ChatThread(
            tenant_id = tenant_id,
            org_unit_id = org_unit_id,
            title     = title,
            user_id   = user_id,
            doc_filter= doc_filter,
        )
        session.add(thread)
        session.flush()
        session.expunge(thread)
        make_transient(thread)
        logger.info("Created thread: %s (tenant=%s org_unit=%s)", thread.id, tenant_id, org_unit_id)
        return thread


def update_thread_title(thread_id: str, tenant_id: str, org_unit_id: str, title: str) -> None:
    """Update thread title — called after first message. Scoped to tenant+org_unit."""
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        thread = session.query(ChatThread).filter(
            ChatThread.id == uuid.UUID(thread_id),
            ChatThread.tenant_id == tenant_id,
            ChatThread.org_unit_id == org_unit_id,
        ).first()
        if thread:
            thread.title = title
            thread.updated_at = datetime.now(timezone.utc)
            session.add(thread)
            logger.info("Updated thread title: %s -> %s", thread_id[:8], title)


def increment_message_count(thread_id: str, tenant_id: str, org_unit_id: str) -> None:
    """Increment message counter on thread. Scoped to tenant+org_unit."""
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        thread = session.query(ChatThread).filter(
            ChatThread.id == uuid.UUID(thread_id),
            ChatThread.tenant_id == tenant_id,
            ChatThread.org_unit_id == org_unit_id,
        ).first()
        if thread:
            thread.message_count = (thread.message_count or 0) + 1
            thread.updated_at = datetime.now(timezone.utc)
            session.add(thread)


def save_human_message(
    thread_id: str,
    tenant_id: str,
    org_unit_id: str,
    content: str,
) -> ChatMessage:
    """
    Save human message as its own row. Indexed into Qdrant for
    GET /chat/search (see _index_message_in_qdrant).
    """
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        msg = ChatMessage(
            tenant_id       = tenant_id,
            org_unit_id     = org_unit_id,
            thread_id       = uuid.UUID(thread_id),
            role            = "human",
            content         = content,
        )
        session.add(msg)
        session.flush()
        message_id = msg.id
        session.expunge(msg)
        make_transient(msg)

    _index_message_in_qdrant(message_id, tenant_id, org_unit_id, thread_id, "human", content)
    logger.info("Saved human message to thread %s", thread_id[:8])
    return msg


def save_ai_message(
    thread_id: str,
    tenant_id: str,
    org_unit_id: str,
    content: str,
    sources: Optional[list] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    has_image: bool = False,
    image_caption: Optional[str] = None,
) -> ChatMessage:
    """
    Save AI message as its own row — separate from human message.
    Stores sources (which chunks were used) and token usage.
    Generates and stores embedding.

    has_image / image_caption: set when this turn was answered using
    a user-attached image (multimodal chat). image_caption is the GPT-4o
    Vision caption of that image, kept for thread history / display.
    """
    with get_db_session_context(tenant_id=tenant_id, org_unit_id=org_unit_id) as session:
        msg = ChatMessage(
            tenant_id         = tenant_id,
            org_unit_id       = org_unit_id,
            thread_id         = uuid.UUID(thread_id),
            role              = "ai",
            content           = content,
            sources           = sources or [],
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            has_image         = has_image,
            image_caption     = image_caption,
        )
        session.add(msg)
        session.flush()
        message_id = msg.id
        session.expunge(msg)
        make_transient(msg)

    _index_message_in_qdrant(message_id, tenant_id, org_unit_id, thread_id, "ai", content)
    logger.info("Saved AI message to thread %s | sources=%d | has_image=%s",
               thread_id[:8], len(sources or []), has_image)
    return msg


def get_thread(thread_id: str, tenant_id: str, org_unit_id: str, db: Session) -> Optional[ChatThread]:
    """
    Get thread by ID, scoped to tenant_id + org_unit_id. Returns None if
    the thread doesn't exist OR belongs to a different tenant/department —
    callers should treat all cases identically (404), never distinguish
    them in the response, to avoid leaking whether a given thread_id
    exists at all.
    """
    return db.query(ChatThread).filter(
        ChatThread.id == uuid.UUID(thread_id),
        ChatThread.tenant_id == tenant_id,
        ChatThread.org_unit_id == org_unit_id,
    ).first()


def get_thread_messages(thread_id: str, tenant_id: str, org_unit_id: str, db: Session) -> list[ChatMessage]:
    """Get all messages for a thread ordered by time, scoped to tenant_id + org_unit_id."""
    return (
        db.query(ChatMessage)
        .filter(
            ChatMessage.thread_id == uuid.UUID(thread_id),
            ChatMessage.tenant_id == tenant_id,
            ChatMessage.org_unit_id == org_unit_id,
        )
        .order_by(ChatMessage.created_at.asc())
        .all()
    )