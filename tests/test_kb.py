import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select

from main import app
from db.models import KBDocument, KBChatHistory
from db.database import get_async_db_session_fastapi

client = TestClient(app)
DEFAULT_KB_HEADERS = {
    "X-Tenant-ID": "test-tenant",
    "X-Org-Unit-ID": "test-org",
}


@pytest.fixture
def mock_db_session():
    session = MagicMock()
    # Mock execute/scalars etc. if needed
    return session


@patch("api.routes.kb.ingest_document")
def test_upload_kb_document_endpoint(mock_ingest):
    """Upload endpoint creates a DB record and returns 202 with processing status."""
    mock_ingest.return_value = AsyncMock()

    # Mock async DB session so no real DB connection is needed
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    async def override_dep():
        yield mock_session

    app.dependency_overrides[get_async_db_session_fastapi] = override_dep

    file_content = b"This is some test content for the knowledge base."
    files = {"file": ("test_doc.txt", file_content, "text/plain")}
    data = {"owner_id": "test_owner_123"}

    response = client.post("/kb/upload", files=files, data=data, headers=DEFAULT_KB_HEADERS)

    app.dependency_overrides.clear()

    assert response.status_code == 202
    json_data = response.json()
    assert "kbDocumentId" in json_data
    assert json_data["status"] == "processing"


@patch("api.routes.kb.retrieve_relevant_context")
def test_retrieve_context_endpoint(mock_retrieve):
    mock_retrieve.return_value = ["Relevant context sentence 1", "Relevant context sentence 2"]

    payload = {
        "owner_id": "test_owner_123",
        "query": "test query",
        "filters": {}
    }

    response = client.post("/kb/retrieve", json=payload, headers=DEFAULT_KB_HEADERS)
    assert response.status_code == 200
    assert response.json() == ["Relevant context sentence 1", "Relevant context sentence 2"]
    mock_retrieve.assert_called_once_with("test_owner_123", "test query")


@patch("api.routes.kb.chat_with_kb")
def test_chat_kb_endpoint(mock_chat):
    mock_chat.return_value = "This is the AI response based on the context."

    payload = {
        "owner_id": "test_owner_123",
        "message": "Hello AI",
        "conversation_id": "conv_abc"
    }

    response = client.post("/kb/chat", json=payload, headers=DEFAULT_KB_HEADERS)
    assert response.status_code == 200
    assert response.json() == {"response": "This is the AI response based on the context."}


# ─────────────────────────────────────────────────────────────────────────────
# Status endpoint tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_kb_doc(doc_id: uuid.UUID, status: str, error_message: str = None) -> KBDocument:
    """Helper to create a mock KBDocument ORM object."""
    doc = KBDocument(
        id=doc_id,
        owner_id="test_owner_123",
        file_name="test.txt",
        status=status,
        error_message=error_message,
        created_at=datetime.now(timezone.utc),
    )
    return doc


@patch("api.routes.kb.get_async_db_session_fastapi")
def test_get_kb_status_ready(mock_dep):
    """Status endpoint returns correct data for a ready document."""
    doc_id = uuid.uuid4()
    mock_doc = _make_kb_doc(doc_id, status="ready")

    # Build async mock session
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_doc

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    # Override FastAPI dependency
    async def override_dep():
        yield mock_session

    app.dependency_overrides[get_async_db_session_fastapi] = override_dep

    response = client.get(f"/kb/status/{doc_id}", headers=DEFAULT_KB_HEADERS)

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["kbDocumentId"] == str(doc_id)
    assert data["status"] == "ready"
    assert data["error_message"] is None


@patch("api.routes.kb.get_async_db_session_fastapi")
def test_get_kb_status_failed(mock_dep):
    """Status endpoint surfaces error_message for a failed document."""
    doc_id = uuid.uuid4()
    mock_doc = _make_kb_doc(doc_id, status="failed", error_message="Ingestion error: file corrupt")

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_doc

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    async def override_dep():
        yield mock_session

    app.dependency_overrides[get_async_db_session_fastapi] = override_dep

    response = client.get(f"/kb/status/{doc_id}", headers=DEFAULT_KB_HEADERS)

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["kbDocumentId"] == str(doc_id)
    assert data["status"] == "failed"
    assert data["error_message"] == "Ingestion error: file corrupt"


@patch("api.routes.kb.get_async_db_session_fastapi")
def test_get_kb_status_processing(mock_dep):
    """Status endpoint returns processing for an in-flight document."""
    doc_id = uuid.uuid4()
    mock_doc = _make_kb_doc(doc_id, status="processing")

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_doc

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    async def override_dep():
        yield mock_session

    app.dependency_overrides[get_async_db_session_fastapi] = override_dep

    response = client.get(f"/kb/status/{doc_id}", headers=DEFAULT_KB_HEADERS)

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["kbDocumentId"] == str(doc_id)
    assert data["status"] == "processing"


@patch("api.routes.kb.get_async_db_session_fastapi")
def test_get_kb_status_not_found(mock_dep):
    """Status endpoint returns 404 when document does not exist."""
    doc_id = uuid.uuid4()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    async def override_dep():
        yield mock_session

    app.dependency_overrides[get_async_db_session_fastapi] = override_dep

    response = client.get(f"/kb/status/{doc_id}", headers=DEFAULT_KB_HEADERS)

    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["detail"] == "KB Document not found"
