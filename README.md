## What is Avabodh?

Avabodh is a production-grade Document Intelligence Platform built at MediaGuru Consultants. It takes any document (PDF, TXT, DOCX, CSV), generates a structured AI summary, stores semantic embeddings for retrieval, and lets users have a contextual conversation with their documents through a RAG-powered chatbot — all served through a REST API with full Pydantic validation, PostgreSQL persistence, and Docker deployment.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [System Flow](#system-flow)
- [Database Schema](#database-schema)
- [API Reference](#api-reference)
- [Setup & Installation](#setup--installation)
- [Running with Docker](#running-with-docker)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [Known Issues & Production Fixes Applied](#known-issues--production-fixes-applied)
- [Roadmap — In Progress](#roadmap--in-progress)
- [For New Contributors](#for-new-contributors)
- [License](#license)

---

## Architecture Overview

The system has 3 major pipelines that all share the same PostgreSQL + pgvector backend:

```
┌─────────────────────────────────────────────────────────────┐
│                     1. DOCUMENT INGESTION                    │
│  Upload (PDF/TXT/DOCX/CSV)                                    │
│       → Load → Clean → Split (SemanticChunker)                │
│       → Parallel Summarise (Map-Reduce, OpenAI)                │
│       → Pydantic Validate → Store Summary (PostgreSQL)         │
│       → Generate + Store Chunk Embeddings (pgvector)           │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  2. RAG CHATBOT (Q&A)                          │
│  User Query → Pydantic Validate                                │
│       → Load Conversation Memory (last 5 turns from DB)        │
│       → Embed Query → pgvector Similarity Search                │
│       → Contextual Compression (filter to relevant content)    │
│       → Build Prompt (history + context + query)               │
│       → LLM Call → Pydantic Output Parser                      │
│       → Store human + AI messages (separate rows)              │
│       → Auto-generate thread title (first message only)        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│              3. WEB SCRAPING (IN PROGRESS)                     │
│  URL → Scrape (playwright, being evaluated) → Markdown            │
│       → Same pipeline as Document Ingestion from here           │
└─────────────────────────────────────────────────────────────┘
```

**Key design decision:** Web scraping does NOT replace document upload — it is an additional ingestion source feeding into the exact same summarisation + RAG pipeline.

---

## Tech Stack

| Layer | Technology | Why chosen |
|---|---|---|
| API Framework | FastAPI 0.115 | Async, auto Swagger docs, native Pydantic integration |
| LLM (Summarise + Chat) | OpenAI GPT-4o-mini | Best cost/quality ratio for production |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) | Fast, cheap, high quality |
| Text Splitting | SemanticChunker (LangChain Experimental) | Splits by meaning, not fixed character count — produces coherent chunks |
| Vector Store | pgvector (PostgreSQL extension) | No separate vector DB needed — same Postgres instance |
| Database | PostgreSQL 16 | Relational + vector in one engine |
| ORM | SQLAlchemy 2.0 | Type-safe queries, session management |
| Validation | Pydantic v2 | Validates every request, response, and LLM output |
| Memory | ConversationBufferWindowMemory (LangChain) | Last 5 turns loaded fresh from DB on every request — stateless server, persistent memory |
| Retrieval | ContextualCompressionRetriever | Filters retrieved chunks to only query-relevant sentences before sending to LLM |
| Containerisation | Docker + Docker Compose | One-command deployment for the whole team |
| Tracing | LangSmith | Every LLM call traceable for debugging/cost monitoring |
| Tested via | Postman + Swagger UI (`/docs`) | Manual + automated endpoint testing |

---

## Project Structure

```
AvabodhAI/
├── main.py                       # FastAPI app entry point — registers all routers
├── requirements.txt
├── Dockerfile
├── docker-compose.yml            # Runs API + pgvector-enabled PostgreSQL together
├── .dockerignore
├── .env.example                  # Template — copy to .env and fill real values
├── Avabodh_API.postman_collection.json   # Import into Postman for manual testing
│
├── config/
│   └── settings.py               # Pydantic Settings — loads all env vars with defaults
│
├── db/
│   ├── database.py               # SQLAlchemy engine, session factory, pgvector extension init
│   └── models.py                 # ORM tables: DocumentSummary, DocumentChunk, ChatThread, ChatMessage
│
├── pipeline/
│   ├── loader.py                  # Step 1 — Document loading (PDFPlumberLoader, TextLoader, etc.)
│   ├── splitter.py                 # Step 2 — Cleaning + SemanticChunker + noise filtering
│   ├── summariser.py               # Steps 3-4 — Parallel Map-Reduce summarisation (ThreadPoolExecutor)
│   ├── storage.py                  # Steps 5-7 — Pydantic validation + PostgreSQL save (summaries)
│   ├── embedder.py                 # NEW — Generates + stores chunk embeddings in pgvector
│   ├── retriever.py                # RAG — Vector search + contextual compression
│   ├── memory.py                   # RAG — Loads conversation history from DB into LangChain memory
│   ├── chat.py                     # RAG — LLM call, thread title generation
│   ├── chat_storage.py             # RAG — Saves human/AI messages as separate rows + embeddings
│   └── orchestrator.py             # CLI orchestrator — wires loader→splitter→summariser→storage→embedder
│
├── api/
│   ├── routes/
│   │   ├── health.py               # GET /health/, /health/db
│   │   ├── documents.py            # Document CRUD + upload endpoint
│   │   ├── chat.py                 # Chat/RAG endpoints + thread management
│   │   └── search.py               # Semantic search across document chunks
│   ├── schemas/
│   │   ├── document.py             # Pydantic request/response models for documents
│   │   └── chat.py                 # Pydantic request/response models for chat
│   └── middleware/
│       └── logging_middleware.py   # Logs every request: method, path, status, latency
│
├── utils/
│   └── logger.py                   # Rotating file logger (UTF-8 safe)
│
└── tests/
    ├── test_pipeline.py             # Unit tests — splitter, validation
    └── test_api.py                  # FastAPI TestClient endpoint tests
```

---

## System Flow

### Document Ingestion (implemented)
1. `POST /documents/upload` — file received, saved permanently (needed for future re-embedding)
2. **Load** — `PDFPlumberLoader`/`TextLoader`/etc. extracts raw text + SHA-256 hash for dedup
3. **Clean** — fixes PDF hyphenation, removes page-number noise, normalises whitespace
4. **Split** — `SemanticChunker` (threshold 0.85) creates topically coherent chunks; chunks under 200 chars or low alphabetic ratio are filtered out as noise
5. **Summarise** — all chunks sent to OpenAI **simultaneously** via `ThreadPoolExecutor(max_workers=10)` (Map step, ~10x faster than sequential), then one Reduce call combines them into a structured summary (Document Overview / Key Findings / Main Topics / Conclusion format)
6. **Validate** — Pydantic `DocumentSummaryOutput` schema enforces shape before saving
7. **Save Summary** — `document_summaries` table; duplicate file hash returns cached result instantly (no re-processing)
8. **Embed Chunks** — each chunk embedded (batched, 100/batch) and stored in `document_chunks` with vector(1536) column

### RAG Chatbot (implemented)
1. `POST /chat/message` or thread-based interaction
2. New thread created if no `thread_id` provided; title auto-generated by LLM after first message
3. Last 5 conversation turns loaded fresh from `chat_messages` table into `ConversationBufferWindowMemory` (server is stateless — memory is rebuilt from DB every request)
4. Query embedded → `pgvector` cosine similarity search on `document_chunks` (optionally filtered to one document)
5. **Contextual compression** — `LLMChainExtractor` strips retrieved chunks down to only the sentences relevant to the query, reducing noise before the final LLM call
6. Fallback response returned immediately if no relevant chunks found (saves an LLM call)
7. Prompt built from: system instructions + document context + conversation history + current query
8. LLM answers → Pydantic `ChatMessageResponse` validates shape
9. Human and AI messages saved as **separate rows** (never combined in one row) in `chat_messages`, each with its own embedding for future semantic search across chat history
10. `chat_threads.message_count` and `updated_at` incremented

### Web Scraping (in progress — not yet merged into main repo)
- Being tested in a separate repo
- Decision pending on token cost + output cleanliness comparison
- Once finalised, plugs into the pipeline as a new loader at Step 1 only — everything downstream (split → summarise → embed → chat) stays unchanged

---

## Database Schema

All 4 tables live in the same PostgreSQL database (`Avabodh`), pgvector extension enabled once via `CREATE EXTENSION IF NOT EXISTS vector;`.

### `document_summaries`
One row per uploaded document.

| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| doc_name | VARCHAR(512) | |
| summary_text | TEXT | Structured: Overview/Findings/Topics/Conclusion |
| key_topics | TEXT | |
| page_count, chunk_count | INTEGER | |
| source_path | VARCHAR(1024) | File kept permanently (not deleted after processing) |
| doc_hash | VARCHAR(64) | SHA-256 — used for dedup |
| embedding_status | VARCHAR(32) | `pending` / `completed` / `failed` |
| embedding_model, model_used | VARCHAR(128) | |
| created_at, updated_at | TIMESTAMPTZ | |

### `document_chunks`
One row per chunk — pgvector table.

| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| summary_id | UUID (FK → document_summaries, CASCADE delete) | |
| doc_hash, doc_name | VARCHAR | For fast lookup without join |
| chunk_index, total_chunks | INTEGER | Position tracking |
| chunk_text | TEXT | |
| embedding | **vector(1536)** | OpenAI text-embedding-3-small |
| page_number | INTEGER | |

### `chat_threads`
One row per conversation.

| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| title | VARCHAR(512) | Auto-generated by LLM from first message |
| doc_filter | VARCHAR(512) | Optional — restrict thread to one document |
| message_count | INTEGER | |
| updated_at | TIMESTAMPTZ | Used for sorting thread list |

### `chat_messages`
One row per message — **human and AI never share a row**.

| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| thread_id | UUID (FK → chat_threads, CASCADE delete) | |
| role | VARCHAR(16) | `human` or `ai` |
| content | TEXT | |
| embedding | vector(1536), nullable | Enables semantic search across chat history |
| sources | JSON | AI messages only — which chunks were cited |
| prompt_tokens, completion_tokens | INTEGER | Cost tracking |

---

## API Reference

Full interactive docs always available at **`/docs`** (Swagger UI) once running.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Root health check |
| GET | `/health/` | Basic health check |
| GET | `/health/db` | Database connectivity check |
| POST | `/documents/upload` | Upload file → summarise → embed → store |
| GET | `/documents/` | List summaries (paginated) |
| GET | `/documents/{id}` | Full summary by UUID |
| PATCH | `/documents/{id}` | Rename document |
| DELETE | `/documents/{id}` | Delete document + cascading chunks |
| POST | `/search/` | Semantic search across `document_chunks` |
| GET | `/search/chunks/{id}` | All chunks for one document |
| POST | `/chat/message` | Send message, get full JSON answer (with sources + thread_id) |
| POST | `/chat/threads` | Create thread manually |
| GET | `/chat/threads` | List all threads |
| GET | `/chat/threads/{id}` | Get one thread |
| PATCH | `/chat/threads/{id}` | Rename thread |
| DELETE | `/chat/threads/{id}` | Delete thread + all messages |
| GET | `/chat/threads/{id}/messages` | Full message history for a thread |
| GET | `/chat/search` | Semantic search across past chat messages |


A ready-to-import Postman collection is included: `Avabodh_API.postman_collection.json`.

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- Docker Desktop (recommended) — or local PostgreSQL 16+ with pgvector extension
- OpenAI API key

### Local (without Docker)

```bash
git clone https://github.com/bapurba893/AvabodhAI.git
cd AvabodhAI

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt

cp .env.example .env
# Edit .env — set DB_HOST=localhost, fill OPENAI_API_KEY, DB_PASSWORD

# Enable pgvector on your local PostgreSQL
psql -U postgres -d Avabodh -c "CREATE EXTENSION IF NOT EXISTS vector;"

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`

The API is also published on `http://localhost:4000` in Docker for compatibility with older client configs.

---

## Running with Docker (recommended)

```bash
cp .env.example .env
# Edit .env — set DB_HOST=db (NOT localhost — this is the Docker service name)

docker-compose up --build -d
docker-compose logs -f api      # watch startup logs
```

Tables (`document_summaries`, `document_chunks`, `chat_threads`, `chat_messages`) and the `vector` extension are created automatically on first startup.

**Pull the pre-built image instead of building locally:**
```bash
docker pull apurbapm/avabodh-api:latest
docker-compose up -d
```

### Common Docker commands
```bash
docker-compose down              # stop containers, keep data
docker-compose down -v           # stop + WIPE all data (use when schema changes)
docker ps -a                     # see all container statuses
docker-compose logs --tail=50 api   # last 50 log lines
docker exec -it avabodh_api python -c "from db.models import DocumentChunk; print('OK')"  # sanity check imports inside container
```

> ⚠️ **Important:** `DB_HOST` value depends on how you're running it:
> - `uvicorn main:app` locally → `DB_HOST=localhost`
> - `docker-compose up` → `DB_HOST=db`
>
> This is the single most common setup mistake during onboarding.

---

## Environment Variables

See `.env.example` for the full template. Key variables:

```env
OPENAI_API_KEY=sk-...
MAP_MODEL=gpt-4o-mini
REDUCE_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

DB_HOST=db                  # 'localhost' if running outside Docker
DB_PORT=5432
DB_NAME=Avabodh
DB_USER=postgres
DB_PASSWORD=...

MIN_CHUNK_SIZE=200           # chunks below this are discarded as noise
MAX_CHUNK_SIZE=3000
CHUNK_BREAKPOINT_THRESHOLD=0.85   # SemanticChunker sensitivity — lower = fewer, larger chunks

LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=Avabodh Project
```

**LLM provider is swappable** — the codebase has been tested with OpenAI (current/production), Groq (free, fast), HuggingFace Inference API, and Ollama (fully local, zero cost). Only `summariser.py`'s `_build_llm()` function and the relevant `.env` values change.

---

## Testing

```bash
pytest tests/ -v
```

Manual testing:
- **Swagger UI** — `http://localhost:8000/docs` — click any endpoint → Try it out
- **Postman** — import `Avabodh_API.postman_collection.json`, set `base_url` variable

---

## Known Issues & Production Fixes Applied

These were real issues hit during development — documented so the team doesn't re-debug them:

| Issue | Root Cause | Fix Applied |
|---|---|---|
| Chunks contain split words like `"Aper- ture"` | PDF hyphenation not cleaned before chunking | `clean_extracted_text()` in `splitter.py` fixes line-break hyphenation via regex before splitting |
| Meaningless chunks (`"1."`, `"Fig."`, page numbers) | `MIN_CHUNK_SIZE` was too low (10) | Raised to 200; added `is_meaningful_chunk()` filter (rejects <5 real words or <50% alphabetic content) |
| Too many tiny fragmented chunks | `CHUNK_BREAKPOINT_THRESHOLD` was too aggressive (0.95) | Lowered to 0.85 — production-tested sweet spot |
| `DetachedInstanceError` when returning ORM objects from FastAPI routes | SQLAlchemy session closed before object accessed outside `with` block | All storage functions call `session.expunge()` + `make_transient()` before returning |
| `ModuleNotFoundError` for various LangChain imports | Rapid LangChain version churn — `langchain.chains`, `langchain.retrievers`, `langchain.memory` moved between `langchain`, `langchain_community`, and `langchain_classic` across versions | Pinned working imports; see `requirements.txt` (generated via `pip freeze`, not hand-typed versions) |
| `UnicodeEncodeError` on Windows console (`cp1252` codec) | Windows terminal default encoding can't print Unicode box-drawing characters in log messages | Replaced all `━`, `→`, `✓`, `✗` characters in log strings with plain ASCII (`---`, `->`, etc.) |
| Docker container using stale code after rebuild | `docker-compose build` reused cached layers despite file changes | Use `docker-compose build --no-cache` after significant pipeline changes, or `docker-compose down -v && docker-compose up --build -d` to force full recreation including DB schema |
| `document_chunks` table not appearing after model changes | New ORM table added to `models.py` but old DB volume already existed with old schema | `init_db()` only runs `CREATE TABLE IF NOT EXISTS` — it does NOT alter existing tables. Schema changes require `docker-compose down -v` (deletes data) or a manual `ALTER TABLE` / Alembic migration |
| Sequential chunk summarisation was very slow (10–15 min for ~300 chunks) | One LLM call per chunk, one after another | Rewrote Map step using `ThreadPoolExecutor(max_workers=10)` — all chunks sent to OpenAI in parallel batches, ~10x speed improvement |

---

## Roadmap — In Progress

- [ ] **Web scraping ingestion** — currently evaluating Crawl4AI (Apache 2.0) vs Spider/spider-rs (MIT) in a separate test repo for markdown quality + token cost before merging into main pipeline
- [ ] Image upload + query support (multimodal RAG) — planned per lead architect direction
- [ ] Scheduled re-scraping for URL-based sources (content freshness)
- [ ] Migrate schema management from `create_all()` to Alembic migrations (current approach can't handle column changes on existing tables)
- [ ] Authentication layer (currently no auth on any endpoint — internal use only)

---

## For New Contributors

**Start here, in this order:**
1. Read [System Flow](#system-flow) above to understand the 3 pipelines conceptually
2. Run locally with Docker first (`docker-compose up --build -d`) — it's the fastest path to a working environment
3. Open `/docs` and manually trigger `POST /documents/upload` with a sample PDF — watch `docker-compose logs -f api` to see each pipeline step execute live
4. Then trigger `POST /chat/message` with a question about that document
5. Check `pgAdmin` (or any Postgres client) connected to port `5433` (Docker) to see the actual rows created in all 4 tables
6. Read [Known Issues & Production Fixes Applied](#known-issues--production-fixes-applied) before changing `splitter.py`, `models.py`, or any LangChain imports — these have already been debugged once

**Before pushing any code:**
- Never commit `.env` — it's gitignored; only `.env.example` should ever be in the repo
- Search for hardcoded secrets (`sk-`, `lsv2_pt`, DB passwords) before every commit
- If you change `db/models.py`, you must run `docker-compose down -v` to force schema recreation (no migration tool is in place yet)

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

Apache 2.0 was selected because the entire dependency stack (LangChain, FastAPI, pgvector, Docker) is MIT/Apache-2.0-licensed and compatible, and it provides explicit patent protection that MIT does not — the standard choice for production AI/ML platforms (used by TensorFlow, Kubernetes, Docker itself).

---

*Built by Apurba — Product Management Intern, MediaGuru Consultants Pvt Ltd*
