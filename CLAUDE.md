# CLAUDE.md — AvabodhAI-new

> **Verification pass (deep recheck):** every file:line citation below was re-opened and confirmed against the current code. No inaccuracies found in the prior version — all fixes below are additions/resolutions of previously-flagged gaps, not corrections of wrong claims. Headline result: **this repo is confirmed to be the exact code deployed as `avabodh_api` in `clariona-core`'s MDS stack** (see box below) — this was previously an open question, now closed.

Guidance for working in this repo. This documents the **actual code as it exists now**, not the README (`README.md` is stale — it describes an earlier, single-tenant, no-RLS, no-image, no-hybrid-search version of this app; see "README vs reality" at the bottom).

This is the repo referred to elsewhere in the Clariona workspace as "AvabodhAI-new" — the newer of two Avabodh document-intelligence services being compared for a possible migration (see sibling repo `D:\clariona\AvabodhAI`, documented separately in its own `CLAUDE.md`).

> [!IMPORTANT]
> **CONFIRMED: this repo IS the `avabodh_api` container clariona-core already runs in production/dev.** Definitive evidence, not name-overlap inference:
> - This repo's own `docker-compose.yml:20` builds and tags itself `image: apurbapm/avabodhai-api:latest`.
> - `clariona-core/docker-compose.mds.yml:59` runs `avabodh_api` from the **identical image tag**: `image: apurbapm/avabodhai-api:latest`.
> - `clariona-core/docker-compose.mds.yml:67-71` sets `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD` — exactly this repo's own config var names (`config/settings.py`), not a different naming scheme.
> - `clariona-core/docker-compose.mds.yml:76-77` mounts volumes named `uploaded_files:/app/uploaded_files` and `documents:/app/documents` — matching this repo's `Dockerfile` (`RUN mkdir -p uploaded_files documents`) exactly.
> - `clariona-core/docker-compose.mds.yml:208-210` sets `AVABODH_TABLE=document_chunks`, `AVABODH_EMBEDDING_COL=embedding`, `AVABODH_TEXT_COL=chunk_text` as defaults — these are the literal table/column names in `db/models.py:203-227` of this repo.
> - This repo's git `upstream` remote is `github.com/bapurba893/AvabodhAI.git` and the Dockerfile's `LABEL maintainer="Apurba"` (`Dockerfile:7`) — matching the `apurbapm` image-tag author.
>
> **Practical implication for the migration decision**: this is not a choice between two candidate backends in the abstract — `AvabodhAI-new` is the knowledge-base backend `clariona-core` is *currently* wired to (`KNOWLEDGE_BASE_API_URL` in `clariona-core`'s `.env`, port 8002 per `docker-compose.mds.yml:61`, mapped from this container's internal port 8000). Any migration plan that moves capability *out of* this repo and *into* `AvabodhAI` would mean re-pointing `clariona-core` at a different, currently-inactive service — a cutover, not a green-field choice. Flag this explicitly to Vijay if it wasn't already the assumption.

## Architecture Overview

FastAPI service. Three ingestion sources (PDF/TXT/DOCX/CSV upload, single/batch/full-site web scraping via Playwright) feed one shared pipeline: load → split (semantic chunking) → summarise + extract metadata (OpenAI, parallel Map-Reduce) → save summary → embed chunks into pgvector → (PDF/web only) extract + caption + embed images via GPT-4o Vision. A separate RAG chat pipeline does hybrid (vector + keyword) retrieval with Reciprocal-Rank-Fusion merging, contextual compression, and multimodal (image-in-chat) support.

**Multi-tenancy is two-level and mandatory everywhere**: every table, every query, every pipeline function takes both `tenant_id` and `org_unit_id` (department within tenant) and treats them as **hard boundaries applied together, never independently** (`db/models.py:130-136`, `191-195`). Enforcement is defense-in-depth, three layers deep:
1. **Middleware** (`api/middleware/tenant_guard_middleware.py`) — rejects any non-exempt request missing either `X-Tenant-ID` or `X-Org-Unit-ID` header, at 400, before routing.
2. **Per-route dependency** (`api/dependencies.py:28-63`) — `get_tenant_id`/`get_org_unit_id` FastAPI dependencies thread the values into every route function and every downstream query.
3. **Postgres Row-Level Security** (`db/database.py:93-137`) — `FORCE ROW LEVEL SECURITY` on all 4 core tables, backstop against ad-hoc/forgotten-filter reads. The app connects as a **non-superuser restricted role** (`avabodh_app`, not the bootstrap admin role) specifically so RLS actually applies (`db/database.py:1-22`).

**Trust boundary**: `X-Tenant-ID`/`X-Org-Unit-ID` headers are trusted as coming from a gateway/trusted internal caller, NOT verified as end-user input (`api/dependencies.py:13-19`). If this service is ever exposed directly to end users, that assumption breaks and needs a real auth layer (JWT claim, session lookup) instead.

## 1. Document Ingestion Pipeline

**Trigger points**: `POST /documents/upload` (PDF/TXT/DOCX/CSV, `api/routes/documents.py:118`), `POST /web/scrape` (single/batch/full-site URLs, `api/routes/web.py:370`), CLI (`cli.py`, `pipeline/orchestrator.py`).

**PDF text extraction**: `PyMuPDFLoader` (LangChain community loader, `pipeline/loader.py:47`) for `.pdf`; `TextLoader` for `.txt`; `UnstructuredWordDocumentLoader` for `.docx`; `CSVLoader` for `.csv` (`pipeline/loader.py:46-51`).

**Website scraping**: Playwright (Chromium, headless) — `pipeline/scraper.py`. Two modes: `scrape_url_async` (single page, `scraper.py:269`) and `scrape_website_async` (BFS crawl with concurrent batch fetching, semaphore-bounded, `scraper.py:377`). HTML → Markdown via BeautifulSoup + `markdownify`, with noise stripping (nav/footer/cookie banners/ads via CSS-selector denylist, `scraper.py:59-76`), math preservation (MathML/KaTeX/MathJax → LaTeX, `scraper.py:115-143`), and code-block language tagging. Raw HTML is kept in `Document.metadata["raw_html"]` for later image extraction.

**End-to-end flow** (both sources converge here): Load → `clean_extracted_text()` (fixes PDF hyphenation, strips page-number noise, `pipeline/splitter.py:32-56`) → `SemanticChunker` split (embedding-based breakpoint detection, threshold configurable, `pipeline/splitter.py:104-173`) → noise-filter chunks (`is_meaningful_chunk()`, min 200 chars / 5 real words / 50% alphabetic, `splitter.py:59-84`) → parallel Map-Reduce summarisation + metadata extraction (`pipeline/summariser.py`) → Pydantic validation (`pipeline/storage.py:36-58`) → save `document_summaries` row (dedup by SHA-256 hash + tenant + org_unit, `storage.py:115-237`) → embed chunks into `document_chunks` (`pipeline/embedder.py:124-263`) → **PDF/web only**: extract + caption + embed images (below).

## 2. Embedding Generation

- **Model**: OpenAI `text-embedding-3-small` (1536-dim) — `config/settings.py:22-26`, `EMBEDDING_MODEL` env var. Vector column is hardcoded `Vector(1536)` in `db/models.py:227,356` regardless of the configured model (mismatch risk if `EMBEDDING_MODEL` is ever changed to a different-dimension model — not guarded against).
- **What's chunked**: `SemanticChunker` (LangChain Experimental) splits by semantic breakpoint, not fixed size — `pipeline/splitter.py:116-138`. Also calls OpenAI embeddings itself to compute breakpoints (`splitter.py:109-119`) — this is a **real, billed OpenAI call**, separate from the final chunk-embedding step.
- **Chunk size bounds**: `MIN_CHUNK_SIZE` (default 100, `.env`-configurable), `MAX_CHUNK_SIZE` (default 3000, hard-split via `RecursiveCharacterTextSplitter` if exceeded, `splitter.py:87-101`), `CHUNK_BREAKPOINT_THRESHOLD` (default 0.85, percentile-based sensitivity).
- **Batching**: 100 chunks/batch to OpenAI embeddings API (`pipeline/embedder.py:193,342`).
- **Where**: `pipeline/embedder.py` — `store_chunk_embeddings()` (text, `:124`) and `embed_and_store_images()` (images, `:298`).

## 3. Storage — Postgres + pgvector

Confirmed: `pgvector/pgvector:pg16` image (`docker-compose.yml:3`), extension enabled via `CREATE EXTENSION IF NOT EXISTS vector` in `init_db()` (`db/database.py:177`).

**No separate migration tool** — schema is `Base.metadata.create_all()` (SQLAlchemy declarative, `db/database.py:181`), run idempotently on every startup. **Known limitation, self-documented**: this cannot alter existing tables — schema changes to an existing deployment require manual `ALTER TABLE` or a real migration tool (Alembic is a listed dependency in `requirements.txt` but not actually wired up anywhere in the code searched).

**Index type**: Similarity search uses the pgvector `<=>` cosine-distance operator directly in raw SQL (`pipeline/retriever.py:137,192`) with `ORDER BY ... LIMIT`. **No explicit ivfflat/HNSW index found** in `db/models.py` or `db/database.py` — searches would be doing exact (brute-force) nearest-neighbor scans, not approximate index-accelerated ones. This is a real gap worth flagging for scale.

### Schema — 4 tables, all with `tenant_id` + `org_unit_id`

**`document_summaries`** (`db/models.py:124-196`) — one row per document. Key columns: `id` (UUID PK), `tenant_id`, `org_unit_id` (both `String(128)`, indexed, `nullable=False`), `doc_name`, `summary_text`, `doc_hash` (SHA-256, dedup), `embedding_status` (pending/completed), plus a large block of LLM-extracted document metadata (`title`, `author`, `document_type`, `domain`, `key_entities` array, `sentiment`, `confidentiality_level`, etc.) and client-supplied fields (`category`, `effective_from/to` — metadata only, no retrieval effect; `is_ground_truth` — **gates retrieval**, see below). Composite indexes are explicitly `(tenant_id, org_unit_id, X)`, never `org_unit_id` alone — code comments repeatedly warn that a department code could collide across two different tenants.

**`document_chunks`** (`:203-294`) — the pgvector table, **M2O to `document_summaries`** via `summary_id` FK (`ON DELETE CASCADE`). Denormalizes `tenant_id`/`org_unit_id`/`is_ground_truth` onto every row (not joined) specifically so the hot-path similarity query stays a single-table scan with a `WHERE` clause, no join. `embedding` is `Vector(1536)`. Also carries **image-specific columns** (`role` = `"text"` or `"image"`, `image_caption`, `image_type`, `image_width/height`, `vision_confidence`, etc. — image chunks live in the *same table* as text chunks, distinguished by `role`) and a `chunk_tsvector` (`TSVECTOR`, GIN-indexed, auto-populated by a Postgres trigger — see Full-text search below).

**`chat_threads`** (`:301-332`) — one row per conversation, `tenant_id`+`org_unit_id`, `doc_filter` (optional scoping to one document).

**`chat_messages`** (`:339-380`) — one row per message, **human and AI always separate rows** (confirms user's understanding). `embedding` nullable `Vector(1536)` (every message gets embedded for `/chat/search`). `sources` JSON column (which chunks cited, AI messages only). `has_image` + `image_caption` (multimodal chat tracking).

### Full-text search (hybrid search)

Not mentioned in the README at all. A Postgres trigger (`tsvector_update_trigger`, built-in function, `db/database.py:140-164`) auto-populates `document_chunks.chunk_tsvector` from `chunk_text` on every INSERT/UPDATE, GIN-indexed. `pipeline/retriever.py` runs semantic (pgvector) and keyword (`websearch_to_tsquery`) search independently and merges via **Reciprocal Rank Fusion** (`retriever.py:216-254`), not a weighted score blend — deliberate, since cosine similarity and `ts_rank` aren't on comparable scales.

## 4. Tenant/Org-Unit Isolation — CONFIRMED, real and enforced

Every one of the 4 tables above carries both `tenant_id` and `org_unit_id`, `nullable=False`, indexed, always filtered **together** (never `org_unit_id` alone — this is called out in comments at ~15 separate call sites as a deliberate anti-cross-tenant-collision measure). Enforcement is the 3-layer defense-in-depth described in Architecture Overview above (middleware header check → per-route dependency → Postgres RLS with `FORCE` + non-superuser app role). RLS session variables (`app.tenant_id`, `app.org_unit_id`) are set per-request via `set_config(..., true)` — transaction-local, cannot leak across pooled connections (`db/database.py:250-266`).

One documented, deliberate gap: RLS `WITH CHECK` is `true` (doesn't restrict INSERT/UPDATE) — writes rely entirely on the Python-level code always setting `tenant_id`/`org_unit_id` correctly, not on RLS re-validating them (`db/database.py:108-123`, explicit tradeoff discussion in the docstring).

## 5. Chat Flow

**Storage**: Yes, in Postgres — `chat_threads` + `chat_messages`, as above. Every human/AI turn is 2 separate `ChatMessage` rows (`pipeline/chat_storage.py:103-176`).

**Images in chat**: **Confirmed present — contradicts the "AvabodhAI-new doesn't support images" assumption.** `POST /chat/message` accepts optional `image_base64` + `image_media_type` (`api/schemas/chat.py:41-52`). When attached: (1) GPT-4o Vision captions the image (`api/routes/chat.py:74-151`), the caption is embedded and used for pgvector search to find related chunks, AND (2) the final LLM call sends the **actual image bytes** to GPT-4o alongside retrieved text context — the model literally sees the pixels, not just a caption (`api/routes/chat.py:158-202`). Chat images are ephemeral (not stored as `document_chunks` — no dedicated image-storage path for chat-attached images was found); only the resulting caption is persisted on the `ChatMessage` row (`has_image`, `image_caption` columns).

**PDF summary generation/storage**: Summary text (4-paragraph Overview/Key-Findings/Main-Topics/Conclusion structure, `pipeline/summariser.py:69-109`) is generated once per document (Map-Reduce over chunks) and stored in `document_summaries.summary_text`. This is separate from the raw extracted text, which is **not stored** in the DB at all — only chunk-level `chunk_text` (in `document_chunks`) and the summary persist; the original full raw text lives only transiently in memory during processing (the original uploaded *file* itself is kept on disk under `uploaded_files/`, separately, and is what the signed preview link serves).

**Retrieval pipeline** (`pipeline/retriever.py:378-410`): hybrid search (vector + keyword via RRF) → contextual compression (`LLMChainExtractor`, strips chunks to query-relevant sentences; image chunks bypass compression untouched since compression tends to mangle dense captions, `retriever.py:309-317`) → top-k.

**Memory**: `ConversationBufferWindowMemory`, last 5 turns, rebuilt from DB on every request (server is stateless) — `pipeline/memory.py`.

**Retrieval gating**: `is_ground_truth` is **unconditionally enforced** at the SQL `WHERE` level on every search path (vector, keyword, chat) — only documents explicitly marked ground-truth at upload/scrape time are ever retrievable (`pipeline/retriever.py:71`, `db/models.py:258-263`). Default is `False` (opt-in).

## 6. API Surface

All routes require `X-Tenant-ID` + `X-Org-Unit-ID` headers except `/`, `/health*`, `/docs`, `/redoc`, `/openapi.json`, and `GET /documents/{id}/file` (token-authorized instead).

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Root info |
| GET | `/health/`, `/health/db` | Health checks |
| POST | `/documents/upload` | Upload file, full pipeline (`documents.py:118`) |
| GET | `/documents/` | List (paginated, filterable by type/domain/category) |
| GET | `/documents/{id}` | Full detail (also serves as preview payload) |
| GET | `/documents/{id}/preview-url` | Fresh signed preview link |
| GET | `/documents/{id}/file` | Stream original file (signed-token auth, not headers) |
| PATCH | `/documents/{id}` | Rename |
| DELETE | `/documents/{id}` | Delete (cascades chunks) |
| POST | `/search/` | Semantic/keyword/hybrid search, role-filterable (text/image) |
| GET | `/search/chunks/{doc_id}` | All chunks for a document |
| POST | `/chat/message` | Send message (+ optional image), full JSON response |
| POST | `/chat/threads`, GET/PATCH/DELETE `/chat/threads/{id}` | Thread CRUD |
| GET | `/chat/threads/{id}/messages` | Message history |
| GET | `/chat/search` | Semantic search over chat history |
| POST | `/web/scrape` | Single/batch/full-site scraping (`web.py:370`) |

Note: `main.py` registers `documents`, `chat`, `search`, `web`, `health` routers — matches the above. `app.py` is a **separate Streamlit UI** (not part of the FastAPI app), run independently (`streamlit run app.py`) for manual document-summarisation testing; it has its own tenant/org-unit text-input fields since it has no gateway in front of it.

## 7. Auth

**No user-level authentication on the API itself.** Isolation is entirely tenant/org-unit header-based, trusted from an assumed upstream gateway (see Trust boundary note above). No JWT, no API key, no session mechanism found anywhere in `api/`. The one exception is the HMAC-signed token for `GET /documents/{id}/file` (`utils/signed_link.py`) — not user auth, just a way to authorize a plain link/iframe that can't carry custom headers. Tokens **do not expire by default** (`FILE_LINK_TTL_SECONDS = None`, deliberate product decision per code comments) — a leaked link is valid forever.

## 8. Config / Env Vars

Full list in `config/settings.py:14-129`. Highlights:

| Var | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | `""` | Required for real use |
| `MAP_MODEL` / `REDUCE_MODEL` | `"llama3.2"` (stale default — actual production usage is `gpt-4o-mini` per docker-compose.yml default) | Summarisation LLM |
| `EMBEDDING_MODEL` | `"nomic-embed-text"` (stale default, same note) | Overridden to `text-embedding-3-small` in docker-compose |
| `EMBEDDING_DIMENSIONS` | 1536 | **Not actually wired to the DB column** — `Vector(1536)` is hardcoded in `db/models.py`, this setting is decorative |
| `VISION_MODEL` | `"gpt-4o"` | Image captioning |
| `DB_HOST/PORT/NAME/USER/PASSWORD` | — | **Admin/bootstrap connection only** |
| `APP_DB_USER` / `APP_DB_PASSWORD` | `avabodh_app` / `CHANGE-ME-app-role-password` | **Restricted role — what the app actually queries with.** Must be changed in production (loud startup warning if left default, `app.py:38-57`) |
| `FTS_LANGUAGE` | `"english"` | Fixed regconfig for the FTS trigger — known limitation for non-English content |
| `HYBRID_RRF_K` | 60 | RRF constant (same as Elasticsearch default) |
| `MIN_CHUNK_SIZE` / `MAX_CHUNK_SIZE` / `CHUNK_BREAKPOINT_THRESHOLD` | 100 / 3000 / 0.85 | Chunking tuning |
| `IMAGE_MIN_SIZE_BYTES` / `IMAGE_MAX_DIMENSION` / `IMAGE_MAX_WORKERS` | 5000 / 1024 / 4 | Image pipeline tuning |
| `SECRET_KEY` | `"dev-only-CHANGE-ME-in-production"` | Signs preview file tokens — loud startup warning if left default |
| `FILE_LINK_TTL_SECONDS` | `None` | Preview link expiry — off by design |
| `PUBLIC_BASE_URL` | `""` | Set when behind a reverse proxy/gateway, else auto-detected from request |
| `LANGCHAIN_TRACING_V2/ENDPOINT/API_KEY/PROJECT` | — | LangSmith tracing, optional |

**Cross-check against clariona-core's `.env.mds` / `docker-compose.mds.yml` vars**: `MAP_MODEL`/`REDUCE_MODEL`/`OPENAI_API_KEY`/`DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD` all match this repo's own var names exactly — because, per the confirmed-link box under the title, `clariona-core/docker-compose.mds.yml` runs this exact repo's image (`apurbapm/avabodhai-api:latest`) as its `avabodh_api` service. `AVABODH_DB_PASSWORD`/`PIPELINE_DB_PASSWORD` (seen elsewhere in the clariona-core `.env.mds`) belong to the separate `misinfo_detection_engine`/`pipeline_db` services in that same compose file, not to this repo.

## 9. Data Models Summary

See Section 3 for full schema. Five Pydantic (non-table) models also matter: `DocumentSummaryOutput` (LLM summary output validator), `DocumentMetadataOutput` (title/author/type/domain/etc, one extra LLM call per document), `ChunkMetadataOutput` (section/topic/entities per chunk, extracted in the *same* call as the chunk summary — no extra cost), `ImageCaptionOutput` (GPT-4o Vision structured output), `ChunkEmbeddingInput`/`ImageEmbeddingInput` (pre-save validation shapes in `pipeline/embedder.py`).

## 10. Deployment

`docker-compose.yml`: 2 services — `db` (`pgvector/pgvector:pg16`, port `5433→5432` host mapping, healthcheck-gated) and `api` (built from local `Dockerfile`, port `8000`, depends on `db` healthy). No message queue, no separate worker process — everything runs in-process in the FastAPI app (image captioning, embedding, summarisation all happen synchronously within the request handler for `/documents/upload` and `/web/scrape`, no background task queue).

`Dockerfile`: Python 3.11-slim base; installs `poppler-utils`, `tesseract-ocr`, `libmagic1` (system deps for PDF/OCR/type-sniffing via `unstructured`); pre-downloads NLTK data at build time (avoids a first-request network dependency); installs Playwright + Chromium; runs `uvicorn main:app --workers 1`. **Single worker** — no horizontal process scaling within the container as shipped.

**Migration tooling**: None wired up despite `alembic` being in `requirements.txt`. Schema is `create_all()`-only — see Storage section limitation above.

**Bootstrap on every startup** (`init_db()`, `db/database.py:167-192`): enable pgvector extension → `create_all()` → create/update FTS trigger → create/update restricted app role + password → enable+force RLS + policies. All idempotent, safe to re-run.

---

## Known Gaps / Ambiguities

Resolved by the verification pass (were previously listed here, now closed):
- ~~Link to clariona-core's MDS/Avabodh containers is unverified~~ — **CONFIRMED**, see the box under the title. Exact image-tag match, exact volume-name match, exact env-var-name match, exact table/column-name match.
- ~~Whether the deployment topology actually has a trusted gateway in front of this service~~ — `api/dependencies.py:13-19`'s own docstring names the expected caller explicitly: **"the Security Gateway (or an equivalent internal caller)"** — this is a named, intended component, not a hypothetical. Still not verifiable from *this* repo alone whether that gateway is actually deployed in front of every environment (that's `clariona-core`'s side to confirm), so the underlying risk (headers become attacker-controlled if it's ever missing) stands — just no longer an open question about intent.

Still open (genuinely unresolvable from this repo alone):
- **No approximate vector index** (ivfflat/HNSW) found on `document_chunks.embedding` — re-confirmed by re-reading `db/models.py:203-294` and `db/database.py` in full; no `CREATE INDEX ... USING ivfflat/hnsw` anywhere. Similarity search is exact/brute-force. Will not scale well past a moderate row count; worth confirming this is intentional (dataset still small) vs. an oversight.
- **`EMBEDDING_DIMENSIONS` setting is disconnected from the actual schema** — changing `EMBEDDING_MODEL` to a different-dimension model (e.g. `text-embedding-3-large`, 3072-dim) would silently break inserts against the hardcoded `Vector(1536)` columns (`db/models.py:227,356`), not raise a clear config error. Re-confirmed: no dynamic-dimension logic found anywhere in `db/`, `pipeline/`, or `config/`.
- **No background job queue** — long-running work (scraping, summarisation, embedding, image captioning) all happens synchronously inside the HTTP request. Large PDFs or full-site crawls could produce long-held connections/timeouts; no evidence of async task offloading (Celery, RQ, etc.).
- **Migration tooling absent in practice** — `alembic` is a listed dependency but no `alembic.ini`/`versions/` directory found; schema changes to a live DB require manual intervention.
- **No image storage/dedup path found for chat-attached images specifically** (as opposed to PDF/web images, which are persisted as `document_chunks` rows with `role='image'`) — chat images appear to be process-once, caption-and-discard. Re-confirmed via full re-trace of `api/routes/chat.py` — chat-attached images are decoded, captioned, embedded for retrieval, and sent raw to GPT-4o, but never written to `document_chunks` or any other persistent image store.
- **Authentication model is entirely trust-based** on the `X-Tenant-ID`/`X-Org-Unit-ID` headers with no cryptographic binding to a verified identity — acceptable only if this service is truly unreachable except via a trusted internal gateway (see "Resolved" note above — a gateway is clearly the intended architecture, but this repo can't confirm it's actually in place in every environment).

## README vs Reality

`README.md` describes an earlier/simpler version of this app: no RLS, no `org_unit_id` (tenant-only), no image pipeline, no hybrid/full-text search, no signed preview links, Ollama/local-model defaults instead of OpenAI-first. The actual code has all of the above. **Treat the README as historical/aspirational, not authoritative — trust the code (this document) instead.**
