"""
tests/test_api.py
------------------
API-level tests via FastAPI's TestClient — the file the plan
(IMPLEMENTATION_PLAN (2).md §5 step 9) calls for and that got missed in
the first docs/tests pass (only tests/test_pipeline.py was rewritten).

Deliberately does NOT use `with TestClient(app) as client:` — that form
triggers main.py's lifespan (init_db(), the OPENAI_API_KEY startup check,
ensure_collections()), all of which need real Postgres/Qdrant/OpenAI
credentials this test environment doesn't have. Plain `TestClient(app)`
skips lifespan entirely; routes that touch Postgres/Qdrant/OpenAI have
their dependencies/functions mocked per-test instead, matching the plan's
"TestClient, in-memory Qdrant, mocked OpenAI/job" description — the
"in-memory Qdrant" part specifically is exercised in
tests/test_pipeline.py's build_filter()/chunk_point_id() tests instead,
since those don't need the full app wired up.

Like tests/test_pipeline.py, this needs the real dependency stack
installed (`pip install -r requirements.txt`) to actually run — not
available in every sandbox.
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from main import app
from db.database import get_db_session_fastapi

client = TestClient(app)

HEADERS = {"X-Tenant-ID": "test-tenant", "X-Org-Unit-ID": "test-org"}


@pytest.fixture(autouse=True)
def _override_db_session():
    """Every DB-touching route gets a MagicMock session by default — individual tests further mock whatever query methods they need."""
    app.dependency_overrides[get_db_session_fastapi] = lambda: MagicMock()
    yield
    app.dependency_overrides.pop(get_db_session_fastapi, None)


# ─────────────────────────────────────────────────────────────────────────────
# Tenant/org header enforcement (TenantGuardMiddleware) — no mocking needed,
# this rejects before any route/DB/Qdrant code runs.
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_tenant_header_rejected():
    resp = client.get("/documents/", headers={"X-Org-Unit-ID": "test-org"})
    assert resp.status_code == 400
    assert "X-Tenant-ID" in resp.json()["detail"]


def test_missing_org_unit_header_rejected():
    resp = client.get("/documents/", headers={"X-Tenant-ID": "test-tenant"})
    assert resp.status_code == 400
    assert "X-Org-Unit-ID" in resp.json()["detail"]


def test_health_exempt_from_tenant_headers():
    resp = client.get("/health/")
    assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# /search/ — request validation happens before any DB/Qdrant call
# ─────────────────────────────────────────────────────────────────────────────

def test_search_rejects_invalid_search_mode():
    resp = client.post("/search/", headers=HEADERS, json={"query": "test", "search_mode": "not_a_real_mode"})
    assert resp.status_code == 422


def test_search_rejects_invalid_role():
    resp = client.post("/search/", headers=HEADERS, json={"query": "test", "role": "not_text_or_image"})
    assert resp.status_code == 422


def test_search_rejects_unknown_metadata_filter_key():
    resp = client.post("/search/", headers=HEADERS, json={"query": "test", "filters": {"not_allowlisted": ["x"]}})
    assert resp.status_code == 422


def test_search_happy_path_mocked_retrieval():
    fake_hit = {
        "id": str(uuid4()), "doc_name": "policy.pdf", "chunk_index": 0, "chunk_text": "some text",
        "chunk_size": 9, "page_number": 1, "role": "text", "similarity": 0.9, "score": 0.9,
    }
    with patch("api.routes.search.retriever_search", return_value=[fake_hit]) as mock_search:
        resp = client.post("/search/", headers=HEADERS, json={"query": "policy", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["results"][0]["doc_name"] == "policy.pdf"
    mock_search.assert_called_once()


def test_search_chunks_404_when_document_has_no_chunks():
    with patch("api.routes.search.vector_store.scroll_document_chunks", return_value=[]):
        resp = client.get(f"/search/chunks/{uuid4()}", headers=HEADERS)
    assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# /documents/upload — validation + dedup path, without touching real infra
# ─────────────────────────────────────────────────────────────────────────────

def test_upload_rejects_unsupported_extension():
    resp = client.post(
        "/documents/upload", headers=HEADERS,
        files={"file": ("virus.exe", b"not a real file", "application/octet-stream")},
    )
    assert resp.status_code == 422


def test_upload_rejects_invalid_metadata_json():
    resp = client.post(
        "/documents/upload", headers=HEADERS,
        files={"file": ("doc.txt", b"hello world", "text/plain")},
        data={"metadata": "{not valid json"},
    )
    assert resp.status_code == 422


def test_upload_duplicate_returns_cached_record_without_scheduling_ingest():
    fake_existing = MagicMock(
        id=uuid4(), doc_name="doc.txt", status="READY", summary_text="cached summary",
        key_topics=None, page_count=1, chunk_count=3, source_path="/x/doc.txt", language="English",
        model_used="gpt-4o-mini", doc_hash="abc123", tenant_id="test-tenant", org_unit_id="test-org",
        category=None, effective_from=None, effective_to=None, is_ground_truth=False,
        title=None, author=None, document_type=None, domain=None, key_entities=None, mentioned_dates=None,
        target_audience=None, sentiment=None, confidentiality_level=None, metadata_status="pending",
        image_count=0, status_detail=None, metadata_json=None, created_at="2026-08-21T00:00:00Z", updated_at=None,
        stored_path=None,
    )
    # check_duplicate is imported LOCALLY inside upload_document() (a
    # from-import evaluated at call time, not at module load) — patch it
    # at its source module (pipeline.storage), not api.routes.documents,
    # or the patch silently has no effect.
    with patch("pipeline.storage.check_duplicate", return_value=fake_existing), \
         patch("pipeline.ingest.process_document") as mock_ingest:
        resp = client.post(
            "/documents/upload", headers=HEADERS,
            files={"file": ("doc.txt", b"hello world", "text/plain")},
        )
    assert resp.status_code == 201
    assert resp.json()["doc_name"] == "doc.txt"
    # existing.status == "READY", not "FAILED" — no reprocessing should be scheduled.
    mock_ingest.assert_not_called()


def test_document_not_found_returns_404():
    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = None
    app.dependency_overrides[get_db_session_fastapi] = lambda: mock_session

    resp = client.get(f"/documents/{uuid4()}", headers=HEADERS)
    assert resp.status_code == 404
