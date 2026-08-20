# Knowledge Base (KB) Microservice

## Overview

The KB microservice adds a tenant-isolated, conversational knowledge base on top of the existing Avabodh API. Users can upload documents, check processing status, retrieve compressed context, and have stateful multi-turn conversations grounded in their documents.

---

## What Was Added

### Database Layer — `db/models.py`

Two new ORM tables registered on the shared `Base` class (auto-created at startup via `init_db()`):

| Table | Purpose |
|-------|---------|
| `kb_documents` | Tracks uploaded documents and their ingestion status (`processing` → `ready` / `failed`) |
| `kb_chat_history` | Stores per-conversation chat turns (human + ai) for stateful memory |

Both `status` and `role` columns use **SQLAlchemy `Enum` types** for strict value enforcement at the database level:
- `kb_document_status`: `processing`, `ready`, `failed`
- `kb_chat_role`: `human`, `ai`

No existing tables are modified.

### Database Layer — `db/database.py`

Added async database support alongside the existing sync engine:

- `async_engine` — created via `create_async_engine` using `settings.async_db_url` (`postgresql+asyncpg://...`)
- `AsyncSessionFactory` — via `async_sessionmaker`
- `get_async_db_session_fastapi()` — FastAPI dependency using `async with AsyncSessionFactory()` for proper session lifecycle (commit on success, rollback on exception, auto-close via context manager)

### Application — `main.py` (LangSmith env propagation)

LangSmith tracing variables from `.env`/Settings are pushed into `os.environ` at startup via `_apply_langsmith_env()` — called before any LangChain imports execute. This is required because LangSmith reads `LANGCHAIN_TRACING_V2` from the process environment at import time, not from Pydantic Settings.

### Pipeline Layer — `pipeline/kb_pipeline.py` *(new file)*

Three core async functions:

**`ingest_document(file_path, owner_id, document_id, db)`**
1. Loads the file using `load_single_document`
2. Splits into semantic chunks via `SemanticChunker` + `OpenAIEmbeddings(text-embedding-3-small)`
3. Injects `owner_id` and `document_id` into every chunk's metadata (tenant isolation)
4. Stores chunk vectors into a `PGVectorStore` (`langchain-postgres`) table named `kb_vectors`
5. Updates `kb_documents.status` → `ready` on success, `failed` + error message on exception

**`retrieve_relevant_context(owner_id, query)`**
1. Builds a `PGVectorStore` retriever filtered by `owner_id`
2. Wraps it in a `ContextualCompressionRetriever` using `LLMChainExtractor` (gpt-4o-mini)
3. Returns a list of compressed, relevant text chunks

**`chat_with_kb(owner_id, message, conversation_id, db)`**
1. Fetches the last 10 messages (5 turns) for the `conversation_id` **and `owner_id`** from `kb_chat_history` (both filters required — prevents cross-tenant history leakage)
2. Loads them into `ConversationBufferWindowMemory(k=5)`
3. Runs an LCEL chain: `prompt | ChatOpenAI(gpt-4o-mini) | StrOutputParser` with context + history
4. Saves the new human and AI messages back to `kb_chat_history`

### API Layer — `api/routes/kb.py` *(new file)*

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/kb/upload` | Upload a document (`multipart/form-data`). Returns `kbDocumentId` and `status: processing` |
| `GET` | `/kb/status/{kbDocumentId}` | Poll ingestion status. Returns `processing`, `ready`, or `failed` + error |
| `POST` | `/kb/retrieve` | Run retrieval + contextual compression. Returns list of relevant text chunks |
| `POST` | `/kb/chat` | Stateful conversational Q&A. Returns AI response grounded in KB documents |

### API Layer — `api/schemas/kb_schemas.py` *(new file)*

Pydantic v2 models for all KB request/response payloads:
- `KBUploadResponse`, `KBStatusResponse` — use `ConfigDict(from_attributes=True)` for ORM → Pydantic serialisation
- `KBRetrieveRequest`
- `KBChatRequest`, `KBChatResponse`

### Application — `main.py`

KB router mounted at `/kb`:
```python
from api.routes import kb
app.include_router(kb.router, prefix="/kb", tags=["Knowledge Base"])
```

### Deployment — `docker-compose.yml`

All required environment variables are exposed to the `api` service:

```yaml
OPENAI_API_KEY:       ${OPENAI_API_KEY}
LANGCHAIN_TRACING_V2: ${LANGCHAIN_TRACING_V2:-false}
LANGCHAIN_ENDPOINT:   ${LANGCHAIN_ENDPOINT:-https://api.smith.langchain.com}
LANGCHAIN_API_KEY:    ${LANGCHAIN_API_KEY}
LANGCHAIN_PROJECT:    ${LANGCHAIN_PROJECT:-Avabodh Project}
```

The `kb_uploaded_files` directory is mounted as a named volume so uploaded files survive container restarts during background ingestion:

```yaml
volumes:
  - kb_uploaded_files:/app/kb_uploaded_files
```

---

## Environment Setup

### Required Environment Variables

Create a `.env` file in the project root :

```env
# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=avabodh
DB_USER=postgres
DB_PASSWORD=your_password_here

# OpenAI (required for embeddings and LLM)
OPENAI_API_KEY=sk-...

# LangSmith tracing (optional — set to true to enable)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=Avabodh Project
```

---

## Running the Service

### Option 1 — Docker Compose (recommended)

```bash
# Copy and fill in your secrets
cp .env.example .env   # edit .env with real values

# Start Postgres (pgvector) + API
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

### Option 2 — Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Start Postgres separately (must have pgvector extension)
# Then run:
uvicorn main:app --reload --port 8000
```

The `pgvector` extension and all tables are created automatically at startup.

---

## Running the Tests

### Unit Tests (no database required)

All KB tests use mocked DB sessions — no live Postgres needed.

```bash
# Run only KB tests
pytest tests/test_kb.py -v

# Run all tests
pytest tests/ -v
```

Expected output:
```
tests/test_kb.py::test_upload_kb_document_endpoint     PASSED
tests/test_kb.py::test_retrieve_context_endpoint       PASSED
tests/test_kb.py::test_chat_kb_endpoint                PASSED
tests/test_kb.py::test_get_kb_status_ready             PASSED
tests/test_kb.py::test_get_kb_status_failed            PASSED
tests/test_kb.py::test_get_kb_status_processing        PASSED
tests/test_kb.py::test_get_kb_status_not_found         PASSED

7 passed
```

### What Each Test Covers

| Test | Endpoint | What it verifies |
|------|----------|-----------------|
| `test_upload_kb_document_endpoint` | `POST /kb/upload` | Returns 202, `kbDocumentId` UUID, `status: processing` |
| `test_retrieve_context_endpoint` | `POST /kb/retrieve` | Returns list of context strings, calls pipeline with correct args |
| `test_chat_kb_endpoint` | `POST /kb/chat` | Returns `{"response": "..."}` from mocked pipeline |
| `test_get_kb_status_ready` | `GET /kb/status/{id}` | Returns `status: ready`, no error message |
| `test_get_kb_status_failed` | `GET /kb/status/{id}` | Returns `status: failed` with `error_message` |
| `test_get_kb_status_processing` | `GET /kb/status/{id}` | Returns `status: processing` |
| `test_get_kb_status_not_found` | `GET /kb/status/{id}` | Returns HTTP 404 for unknown document ID |

---

## Manual Integration Testing (live service)

With the service running and a valid `OPENAI_API_KEY` configured:

### 1. Upload a Document

```bash
curl -X POST http://localhost:8000/kb/upload \
  -F "file=@/path/to/your/document.txt" \
  -F "owner_id=user_001"
```

Response:
```json
{
  "kbDocumentId": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "processing"
}
```

### 2. Poll Until Ready

```bash
curl http://localhost:8000/kb/status/3fa85f64-5717-4562-b3fc-2c963f66afa6
```

Response when done:
```json
{
  "kbDocumentId": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "ready",
  "error_message": null
}
```

Possible statuses: `processing` → `ready` or `failed`.

### 3. Retrieve Context

```bash
curl -X POST http://localhost:8000/kb/retrieve \
  -H "Content-Type: application/json" \
  -d '{"owner_id": "user_001", "query": "What is the main topic of the document?"}'
```

Response:
```json
[
  "The document discusses...",
  "Key finding: ..."
]
```

### 4. Chat with the Knowledge Base

```bash
curl -X POST http://localhost:8000/kb/chat \
  -H "Content-Type: application/json" \
  -d '{
    "owner_id": "user_001",
    "message": "Summarise the document for me",
    "conversation_id": "conv_001"
  }'
```

Response:
```json
{
  "response": "Based on the document, ..."
}
```

To verify memory, send a follow-up in the **same** `conversation_id`:

```bash
curl -X POST http://localhost:8000/kb/chat \
  -H "Content-Type: application/json" \
  -d '{
    "owner_id": "user_001",
    "message": "Can you elaborate on the first point?",
    "conversation_id": "conv_001"
  }'
```

The model will reference the previous exchange.

### 5. Swagger UI

All endpoints are also available interactively at:
```
http://localhost:8000/docs
```

---

## Tenant Isolation

Every document chunk stored in the `kb_vectors` table has `owner_id` injected into its metadata. All retrieval and chat queries filter by `owner_id`, so users only ever see their own documents. This is enforced in `pipeline/kb_pipeline.py`:

```python
# Ingestion — metadata injected per chunk
chunk.metadata["owner_id"] = owner_id
chunk.metadata["document_id"] = document_id

# Retrieval — filtered at query time
base_retriever = vector_store.as_retriever(
    search_kwargs={"filter": {"owner_id": owner_id}}
)
```

To verify isolation in the database directly:
```sql
-- Check kb_documents for a specific owner
SELECT id, file_name, status FROM kb_documents WHERE owner_id = 'user_001';

-- Check kb_chat_history for a conversation
SELECT role, content, created_at FROM kb_chat_history
WHERE conversation_id = 'conv_001'
ORDER BY created_at;
```

---

## Architecture Diagram

```
POST /kb/upload
      │
      ├── Save file to kb_uploaded_files/
      ├── INSERT kb_documents (status=processing)
      └── BackgroundTask ──► ingest_document()
                                   │
                                   ├── load_single_document()
                                   ├── SemanticChunker (OpenAI embeddings)
                                   ├── inject owner_id + document_id metadata
                                   ├── PGVectorStore.aadd_documents()
                                   └── UPDATE kb_documents (status=ready|failed)

POST /kb/retrieve
      └── ContextualCompressionRetriever
              ├── PGVectorStore (filter: owner_id)
              └── LLMChainExtractor (gpt-4o-mini)

POST /kb/chat
      ├── SELECT last 10 rows from kb_chat_history (conversation_id)
      ├── ConversationBufferWindowMemory (k=5)
      ├── ContextualCompressionRetriever (same as /retrieve)
      ├── LCEL: prompt | gpt-4o-mini | StrOutputParser
      └── INSERT human + ai messages into kb_chat_history
```
