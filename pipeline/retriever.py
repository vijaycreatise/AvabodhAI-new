"""
pipeline/retriever.py
---------------------
Retrieval pipeline — now hybrid:
1. Embed query using OpenAI                        → vector_search()  (semantic)
2. Parse query into a tsquery                       → keyword_search() (exact keyword)
3. Merge both ranked lists via Reciprocal Rank Fusion → hybrid_search()
4. Contextual compression — filters chunks to query-relevant content only

Why hybrid, not vector-only: semantic search is good at "what does this
mean" queries (different wording, same idea) and structurally weak at
exact-term lookups (a specific order number, an acronym, a case ID) —
short/distinctive strings often don't embed distinctly enough to rank
highly by cosine similarity even when the exact text is right there in
the chunk. Keyword search is the opposite: exact, fast, no semantic
understanding at all. Combining them covers both failure modes.

Why Reciprocal Rank Fusion, not a weighted score blend: vector cosine
similarity and ts_rank are on completely different, incomparable scales
— there's no principled way to decide "how much" of a 0.83 similarity
score should outweigh a keyword rank without real tuning data. RRF
sidesteps that by only using each result's RANK POSITION in its own
list, never the raw scores — the same well-established approach used by
Elasticsearch and most production hybrid-search systems.

Contextual compression is better than raw retrieval because:
- Raw retrieval returns full chunks (may have irrelevant parts)
- Compression extracts only the sentences relevant to the query
- LLM gets cleaner, more focused context → better answers
"""

from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def embed_query(query: str) -> list[float]:
    """Convert query text to embedding vector."""
    client = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model=settings.EMBEDDING_MODEL,
    )
    return client.embed_query(query)


def _build_where(
    tenant_id: str,
    org_unit_id: str,
    doc_filter: Optional[str],
    role_filter: Optional[str],
    params: dict,
) -> str:
    """
    Shared WHERE-clause builder for both vector_search() and
    keyword_search() — the two search paths must apply IDENTICAL
    isolation and ground-truth filtering, or hybrid results could leak
    a chunk through one path that the other correctly excludes. One
    function, both callers, so this can't drift apart by accident.
    """
    clauses = ["tenant_id = :tenant_id", "org_unit_id = :org_unit_id", "is_ground_truth = true"]
    params["tenant_id"] = tenant_id
    params["org_unit_id"] = org_unit_id

    if doc_filter:
        clauses.append("doc_name = :doc_filter")
        params["doc_filter"] = doc_filter

    if role_filter:
        clauses.append("role = :role_filter")
        params["role_filter"] = role_filter

    return "WHERE " + " AND ".join(clauses)


def _row_to_chunk(r) -> dict:
    """Shared row->dict mapping so both search paths produce identically-shaped results."""
    return {
        "id":              str(r.id),
        "doc_name":        r.doc_name,
        "chunk_index":     r.chunk_index,
        "chunk_text":      r.chunk_text,
        "chunk_size":      r.chunk_size,
        "page_number":     r.page_number,
        "doc_hash":        r.doc_hash,
        "role":            r.role or "text",
        "image_type":      r.image_type,
        "image_caption":   r.image_caption,
        "image_url":       r.image_url,
        "section_heading": r.section_heading,
        "topic":           r.topic,
        "chunk_type":      r.chunk_type,
    }


def vector_search(
    query_vector: list[float],
    db: Session,
    tenant_id: str,
    org_unit_id: str,
    top_k: int = 10,
    doc_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
) -> list[dict]:
    """
    Semantic search via pgvector cosine similarity. Returns top_k most
    similar chunks with a `similarity` score (0-1, higher = closer).

    tenant_id and org_unit_id are BOTH required (no defaults) and always
    applied together — org_unit_id is a hard boundary WITHIN a tenant
    (department-level), so a department code that happens to collide
    across two different tenants must never leak across companies. Never
    filter on org_unit_id alone. is_ground_truth = true is always
    applied too, unconditionally — no query-time toggle by design.
    """
    params: dict = {"vector": str(query_vector), "top_k": top_k}
    where_sql = _build_where(tenant_id, org_unit_id, doc_filter, role_filter, params)

    query = text(f"""
        SELECT id, doc_name, chunk_index, chunk_text,
               chunk_size, page_number, doc_hash,
               role, image_type, image_caption, image_url,
               section_heading, topic, chunk_type,
               1 - (embedding <=> CAST(:vector AS vector)) AS similarity
        FROM document_chunks
        {where_sql}
        ORDER BY embedding <=> CAST(:vector AS vector)
        LIMIT :top_k
    """)

    try:
        results = db.execute(query, params).fetchall()
        out = []
        for r in results:
            c = _row_to_chunk(r)
            c["similarity"] = round(float(r.similarity), 4)
            c["search_type"] = "semantic"
            out.append(c)
        return out
    except Exception as e:
        logger.error("Vector search failed: %s", e)
        return []


def keyword_search(
    query_text: str,
    db: Session,
    tenant_id: str,
    org_unit_id: str,
    top_k: int = 10,
    doc_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
) -> list[dict]:
    """
    Exact keyword search via chunk_tsvector (populated automatically by
    the Postgres trigger set up in db/database.py — nothing here writes
    to that column). Uses websearch_to_tsquery(), not the stricter
    to_tsquery() — it accepts plain typed phrases ("annual leave
    policy"), quoted exact phrases, and -exclude syntax the way a person
    actually types into a search box, instead of throwing a syntax error
    on anything that isn't pre-formatted with & / | operators.

    Same isolation + ground-truth filtering as vector_search() via the
    shared _build_where() — see that function's docstring.

    An empty or stopword-only query_text produces an empty tsquery that
    matches nothing — that's correct, safe behavior (falls through to
    whatever the vector side found), not an error.
    """
    if not query_text or not query_text.strip():
        return []

    params: dict = {"query_text": query_text, "top_k": top_k, "lang": settings.FTS_LANGUAGE}
    where_sql = _build_where(tenant_id, org_unit_id, doc_filter, role_filter, params)
    where_sql += " AND chunk_tsvector @@ websearch_to_tsquery(:lang, :query_text)"

    query = text(f"""
        SELECT id, doc_name, chunk_index, chunk_text,
               chunk_size, page_number, doc_hash,
               role, image_type, image_caption, image_url,
               section_heading, topic, chunk_type,
               ts_rank(chunk_tsvector, websearch_to_tsquery(:lang, :query_text)) AS similarity
        FROM document_chunks
        {where_sql}
        ORDER BY similarity DESC
        LIMIT :top_k
    """)

    try:
        results = db.execute(query, params).fetchall()
        out = []
        for r in results:
            c = _row_to_chunk(r)
            c["similarity"] = round(float(r.similarity), 4)   # ts_rank score — NOT comparable to vector similarity, see RRF note below
            c["search_type"] = "keyword"
            out.append(c)
        return out
    except Exception as e:
        # Errors here (e.g. FTS_LANGUAGE misconfigured, or the trigger/
        # column somehow missing on an un-migrated database) should not
        # take down the whole search — hybrid degrades to vector-only.
        logger.warning("Keyword search failed — continuing with vector-only results: %s", e)
        return []


def _reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    top_k: int,
    k: Optional[int] = None,
) -> list[dict]:
    """
    Merge multiple ranked result lists into one, using RRF:
        score(chunk) = sum over lists of  1 / (k + rank_in_that_list)
    Only rank POSITION is used, never the raw similarity/ts_rank scores
    — that's deliberate, see the module docstring for why. A chunk that
    appears near the top of either list scores well; a chunk that
    appears in both scores best of all.

    k defaults to settings.HYBRID_RRF_K (60 — the standard constant used
    by Elasticsearch and the original RRF paper). Higher k flattens the
    influence of exact rank position; lower k makes top ranks matter more.
    """
    k = k if k is not None else settings.HYBRID_RRF_K
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict] = {}
    search_types_by_id: dict[str, set] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunks_by_id.setdefault(cid, chunk)
            search_types_by_id.setdefault(cid, set()).add(chunk.get("search_type", "unknown"))

    merged = []
    for cid, chunk in chunks_by_id.items():
        out = dict(chunk)
        out["rrf_score"] = round(scores[cid], 6)
        out["similarity"] = out["rrf_score"]   # downstream code (memory.py, chat.py source display) reads `similarity` generically
        out["matched_by"] = sorted(search_types_by_id[cid])   # ["semantic"], ["keyword"], or ["keyword", "semantic"] — found by both
        merged.append(out)

    merged.sort(key=lambda c: c["rrf_score"], reverse=True)
    return merged[:top_k]


def hybrid_search(
    query_text: str,
    db: Session,
    tenant_id: str,
    org_unit_id: str,
    top_k: int = 10,
    doc_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
) -> list[dict]:
    """
    Runs semantic (vector) and keyword (FTS) search independently, then
    merges via Reciprocal Rank Fusion. This is the main entry point
    retrieve() now uses. Each side is independently fault-tolerant —
    vector_search()/keyword_search() already catch their own errors and
    return [] rather than raising, so if either backend has a bad day,
    hybrid_search() degrades to whichever side is still working rather
    than failing the whole request.
    """
    query_vector = embed_query(query_text)

    vector_results = vector_search(
        query_vector=query_vector, db=db, tenant_id=tenant_id, org_unit_id=org_unit_id,
        top_k=top_k * 2, doc_filter=doc_filter, role_filter=role_filter,
    )
    keyword_results = keyword_search(
        query_text=query_text, db=db, tenant_id=tenant_id, org_unit_id=org_unit_id,
        top_k=top_k * 2, doc_filter=doc_filter, role_filter=role_filter,
    )

    if not vector_results and not keyword_results:
        return []

    return _reciprocal_rank_fusion([vector_results, keyword_results], top_k=top_k)


def contextual_compression(
    query: str,
    chunks: list[dict],
) -> list[dict]:
    """
    Contextual compression retriever.
    Takes raw chunks and extracts only the parts relevant to the query.

    Why: A chunk may be 500 chars but only 2 sentences are relevant.
    Compression gives the LLM only those 2 sentences — cleaner context.

    Returns filtered chunks. If compression removes everything from a chunk,
    that chunk is dropped.
    """
    if not chunks:
        return []

    # Image chunks are already a dense, purpose-built caption — running them
    # through an LLM sentence-extractor tends to mangle or drop the caption,
    # so only text chunks go through compression. Image chunks pass straight
    # through untouched, keeping their full caption intact.
    text_chunks  = [c for c in chunks if c.get("role", "text") == "text"]
    image_chunks = [c for c in chunks if c.get("role") == "image"]

    if not text_chunks:
        return image_chunks

    # Convert to LangChain Documents for compression
    docs = [
        Document(
            page_content=c["chunk_text"],
            metadata={
                "doc_name":    c["doc_name"],
                "chunk_index": c["chunk_index"],
                "similarity":  c["similarity"],
                "id":          c["id"],
                "role":        c.get("role", "text"),
            }
        )
        for c in text_chunks
    ]

    try:
        llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.MAP_MODEL,
            temperature=0.0,
        )
        compressor = LLMChainExtractor.from_llm(llm)
        compressed_docs = compressor.compress_documents(docs, query)

        # Map compressed docs back to original chunk metadata.
        # Match by id (unique per row) — chunk_index alone isn't safe because
        # text and image chunks are each independently 0-indexed per document,
        # so a text chunk_index=0 and an image chunk_index=0 can collide.
        by_id = {c["id"]: c for c in text_chunks}
        compressed_chunks = []
        for doc in compressed_docs:
            if not doc.page_content.strip():
                continue
            original = by_id.get(doc.metadata.get("id"))
            if original:
                compressed_chunks.append({
                    **original,
                    "chunk_text": doc.page_content,  # compressed text
                    "compressed": True,
                })

        if not compressed_chunks:
            compressed_chunks = text_chunks[:3]

        # Merge image chunks back in untouched, re-sort by similarity
        merged = compressed_chunks + image_chunks
        merged.sort(key=lambda c: c.get("similarity", 0), reverse=True)

        logger.info(
            "Compression: %d chunks -> %d text (+%d image) after filtering",
            len(text_chunks), len(compressed_chunks), len(image_chunks),
        )
        return merged

    except Exception as e:
        logger.warning("Contextual compression failed — using raw chunks: %s", e)
        return (text_chunks[:5] + image_chunks)


def retrieve(
    query: str,
    db: Session,
    tenant_id: str,
    org_unit_id: str,
    top_k: int = 5,
    doc_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
) -> list[dict]:
    """
    Full retrieval pipeline:
    1. Hybrid search — semantic + keyword, merged via RRF — always
       scoped to tenant_id + org_unit_id, always ground-truth-only
    2. Contextual compression
    Returns final list of relevant chunks.
    """
    raw_chunks = hybrid_search(
        query_text=query,
        db=db,
        tenant_id=tenant_id,
        org_unit_id=org_unit_id,
        top_k=top_k * 2,    # get 2x so compression has room to filter
        doc_filter=doc_filter,
        role_filter=role_filter,
    )

    if not raw_chunks:
        logger.info("No chunks found for query: %s", query[:50])
        return []

    # Step: contextual compression
    compressed = contextual_compression(query, raw_chunks)
    return compressed[:top_k]
