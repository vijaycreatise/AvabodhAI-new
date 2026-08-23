"""
pipeline/retriever.py
---------------------
Rewritten 2026-08-21 (Qdrant migration). Retrieval is now:

1. build_filter() — the SINGLE isolation chokepoint. Qdrant has no RLS
   equivalent, so this function is the only thing standing between one
   tenant's search and another's data. EVERY read in this module goes
   through it. tenant_id is mandatory; org_unit_id defaults to [header
   org] if not explicitly widened by the caller.
2. search() — dense (OpenAI) + sparse (fastembed SPLADE) via
   pipeline/vector_store.py, fused server-side by Qdrant (RRF), or either
   leg alone for "semantic"/"keyword" mode.
3. rerank via fastembed cross-encoder (pipeline/embedder.py::rerank) over
   the fused candidates.
4. retrieve() — used by chat: same search() + rerank, plus the
   image-in-chat merge (unchanged behavior from before this rewrite).

is_ground_truth is no longer a hard gate (previously unconditionally
enforced) — it's now an OPTIONAL filter, per Vijay's decision
(IMPLEMENTATION_PLAN (2).md decisions table). Pass is_ground_truth=True
to restrict to ground-truth-only documents; omit it to search everything.
"""

from datetime import date
from typing import Optional

from qdrant_client import models

from pipeline import embedder, vector_store
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Metadata keys clients are allowed to filter by — mirrors
# vector_store.FILTERABLE_METADATA_KEYS (the set that actually gets
# indexed); kept here too so build_filter() can validate/422 without
# importing vector_store's index-creation concerns into this module.
FILTERABLE_METADATA_KEYS = vector_store.FILTERABLE_METADATA_KEYS


def build_filter(
    tenant_id: str,
    org_unit_id: str,
    org_ids: Optional[list[str]] = None,
    document_ids: Optional[list[str]] = None,
    doc_name: Optional[str] = None,
    role: Optional[str] = None,
    is_ground_truth: Optional[bool] = None,
    metadata: Optional[dict[str, list[str]]] = None,
    as_of: Optional[date] = None,
) -> models.Filter:
    """
    The single isolation + filtering chokepoint for every Qdrant read.
    tenant_id is always required and always applied. org_unit_id (from
    the X-Org-Unit-ID header) is the default scope; pass org_ids to widen
    to multiple departments within the SAME tenant — never across tenants.

    metadata keys not in FILTERABLE_METADATA_KEYS raise ValueError — the
    caller (api/routes/search.py) turns that into a 422, per architecture
    doc §10/§28 Rule 6 ("keep filterable metadata controlled").
    """
    must: list[models.Condition] = [
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
    ]

    scoped_orgs = org_ids if org_ids else [org_unit_id]
    must.append(models.FieldCondition(key="org_unit_id", match=models.MatchAny(any=scoped_orgs)))

    if document_ids:
        must.append(models.FieldCondition(key="document_id", match=models.MatchAny(any=[str(d) for d in document_ids])))
    if doc_name:
        must.append(models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_name)))
    if role:
        must.append(models.FieldCondition(key="role", match=models.MatchValue(value=role)))
    if is_ground_truth is not None:
        must.append(models.FieldCondition(key="is_ground_truth", match=models.MatchValue(value=is_ground_truth)))

    if metadata:
        for key, values in metadata.items():
            if key not in FILTERABLE_METADATA_KEYS:
                raise ValueError(f"'{key}' is not a filterable metadata key. Allowed: {sorted(FILTERABLE_METADATA_KEYS)}")
            if values:
                must.append(models.FieldCondition(key=f"meta_{key}", match=models.MatchAny(any=values)))

    if as_of is not None:
        as_of_str = as_of.isoformat()
        # (effective_from IS NULL OR effective_from <= as_of)
        must.append(models.Filter(should=[
            models.IsEmptyCondition(is_empty=models.PayloadField(key="effective_from")),
            models.FieldCondition(key="effective_from", range=models.DatetimeRange(lte=as_of_str)),
        ]))
        # (effective_to IS NULL OR effective_to >= as_of)
        must.append(models.Filter(should=[
            models.IsEmptyCondition(is_empty=models.PayloadField(key="effective_to")),
            models.FieldCondition(key="effective_to", range=models.DatetimeRange(gte=as_of_str)),
        ]))

    return models.Filter(must=must)


def _dynamic_search_candidates(query_filter: models.Filter) -> int:
    """
    How many pre-rerank candidates to fetch for THIS query's scope —
    min(actual chunk count in scope, SEARCH_CANDIDATES_MAX), so a small
    document doesn't waste time fetching more candidates than it has, and
    a large document isn't capped at a fixed count that only covered a
    fraction of it. Falls back to the static settings.SEARCH_CANDIDATES on
    any count failure (Qdrant hiccup, etc.) — this must never be a hard
    blocker for search working at all.
    """
    try:
        chunk_count = vector_store.count_chunks(query_filter)
        if chunk_count > 0:
            return min(chunk_count, settings.SEARCH_CANDIDATES_MAX)
    except Exception as e:
        logger.warning("Dynamic SEARCH_CANDIDATES sizing failed, falling back to static value: %s", e)
    return settings.SEARCH_CANDIDATES


def search(
    query: str,
    query_filter: models.Filter,
    mode: str = "hybrid",
    top_k: int = 10,
    do_rerank: bool = True,
    score_threshold: Optional[float] = None,
    query_point_id: Optional[str] = None,
    lookup_from_collection: Optional[str] = None,
) -> list[dict]:
    """
    mode: "hybrid" (dense+sparse, Qdrant-native RRF fusion — default),
    "semantic" (dense only), "keyword" (sparse only).

    When do_rerank, fetches settings.SEARCH_CANDIDATES candidates first
    (recall-oriented), then reranks with the fastembed cross-encoder and
    truncates to top_k (precision-oriented) — architecture doc §18.

    score_threshold: per-call override of settings.SEARCH_SCORE_THRESHOLD
    (.env-tunable) — None (default) just uses whatever's configured there.

    query_point_id + lookup_from_collection: bypass embedding `query`
    entirely and search using an EXISTING point's own stored vector
    instead — e.g. "find chunks similar to this chunk" (query_point_id
    from this same collection) or "find chunks similar to this past chat
    message" (query_point_id from avabodh_chat_messages,
    lookup_from_collection=settings.QDRANT_CHAT_COLLECTION). See
    pipeline/vector_store.py::search() for the full mechanics. `query`
    (the text arg) still needs to be passed but is ignored for embedding
    when query_point_id is set — only used for the rerank pass below.

    Each leg is independently fault-tolerant, same property the old
    pgvector-era hybrid_search() had (its own docstring: "if either
    backend has a bad day, hybrid_search() degrades to whichever side is
    still working"): if mode="hybrid" and ONE of dense/sparse embedding
    fails, this degrades to the other leg alone rather than failing the
    whole search. Only fails outright if the mode being asked for has no
    surviving leg (e.g. mode="semantic" and dense embedding itself fails).
    """
    dense_vector = sparse_indices = sparse_values = None

    if query_point_id is None:
        if mode in ("hybrid", "semantic"):
            try:
                dense_vector = embedder.embed_dense_query(query)
            except Exception as e:
                logger.warning("Dense embedding failed (mode=%s): %s", mode, e)
                if mode == "semantic":
                    return []
                mode = "keyword"   # degrade hybrid -> keyword-only
        if mode in ("hybrid", "keyword"):
            try:
                sparse_indices, sparse_values = embedder.embed_sparse_query(query)
            except Exception as e:
                logger.warning("Sparse embedding failed (mode=%s): %s", mode, e)
                if mode == "keyword":
                    return []
                mode = "semantic"   # degrade hybrid -> semantic-only

    fetch_limit = _dynamic_search_candidates(query_filter) if do_rerank else top_k

    try:
        hits = vector_store.search(
            query_filter=query_filter,
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            mode=mode,
            limit=fetch_limit,
            score_threshold=score_threshold,
            query_point_id=query_point_id,
            lookup_from_collection=lookup_from_collection,
        )
    except Exception as e:
        logger.error("Qdrant search failed (mode=%s): %s", mode, e)
        return []

    if not hits:
        return []

    if do_rerank and len(hits) > 1:
        try:
            scores = embedder.rerank(query, [h.get("chunk_text", "") for h in hits])
            for h, s in zip(hits, scores):
                h["rerank_score"] = float(s)
            hits.sort(key=lambda h: h["rerank_score"], reverse=True)
        except Exception as e:
            logger.warning("Reranking failed — falling back to fusion order: %s", e)

    out = []
    for h in hits[:top_k]:
        h = dict(h)
        h["chunk_id"] = h["id"]
        h["similarity"] = h.get("rerank_score", h.get("score", 0.0))
        h["search_type"] = mode
        out.append(h)
    return out


def retrieve(
    query: str,
    tenant_id: str,
    org_unit_id: str,
    top_k: int = 5,
    doc_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
    is_ground_truth: Optional[bool] = None,
) -> list[dict]:
    """
    Chat's retrieval entry point — hybrid search + rerank, scoped to one
    tenant/org. Kept as a thin wrapper (search() already does the real
    work) so chat.py's call site doesn't need to build a Filter itself.
    """
    query_filter = build_filter(
        tenant_id=tenant_id, org_unit_id=org_unit_id,
        doc_name=doc_filter, role=role_filter, is_ground_truth=is_ground_truth,
    )
    results = search(query=query, query_filter=query_filter, mode="hybrid", top_k=top_k)
    if not results:
        logger.info("No chunks found for query: %s", query[:50])
    return results


def multi_query_search(
    queries: list[str],
    query_filter: models.Filter,
    mode: str = "hybrid",
    top_k: int = 10,
) -> list[dict]:
    """
    Retrieves candidates for EACH query variant independently (no
    per-variant rerank — reranking happens once, on the merged union,
    below), merges by chunk id (keeping the best fusion score seen across
    variants), then reranks the union against queries[0] (the primary/
    condensed query — the clearest single statement of intent) before
    truncating to top_k.

    queries[0] is expected to be a standalone, context-complete query
    (see pipeline/chat.py::generate_search_queries) — the alternates
    exist purely to broaden recall, not as a replacement for a good
    primary query.

    2026-08-21: previously issued one full embed+search round trip PER
    query variant, serially (N dense-embed calls, N sparse-embed calls, N
    Qdrant calls for N queries). Both embedder.embed_dense_batch() and
    embed_sparse_batch() already batch multiple texts into one call each
    (built for chunk ingestion, reused here as-is), and
    vector_store.batch_search() sends all N hybrid queries to Qdrant in
    ONE query_batch_points() round trip — so an N-query multi-query search
    now costs exactly 1 dense-embed call + 1 sparse-embed call + 1 Qdrant
    call, regardless of N, instead of 3N. mode is currently always hybrid
    here in practice (the only caller passes "hybrid"); non-hybrid modes
    fall back to the old per-query search() loop since batch_search() only
    implements the hybrid path.
    """
    if len(queries) == 1:
        return search(query=queries[0], query_filter=query_filter, mode=mode, top_k=top_k, do_rerank=True)

    candidates_limit = _dynamic_search_candidates(query_filter)

    by_id: dict[str, dict] = {}
    batched_ok = False
    if mode == "hybrid":
        try:
            dense_vectors = embedder.embed_dense_batch(queries)
            sparse_vectors = embedder.embed_sparse_batch(queries)
            batches = vector_store.batch_search(
                query_filter=query_filter,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
                limit=candidates_limit,
            )
            for hits in batches:
                for h in hits:
                    existing = by_id.get(h["id"])
                    if existing is None or h.get("score", 0.0) > existing.get("score", 0.0):
                        by_id[h["id"]] = h
            batched_ok = True
        except Exception as e:
            logger.warning("Batched multi-query search failed — falling back to per-query search: %s", e)

    if not batched_ok:
        for q in queries:
            hits = search(query=q, query_filter=query_filter, mode=mode, top_k=candidates_limit, do_rerank=False)
            for h in hits:
                existing = by_id.get(h["id"])
                if existing is None or h.get("score", 0.0) > existing.get("score", 0.0):
                    by_id[h["id"]] = h

    candidates = list(by_id.values())
    if not candidates:
        return []

    try:
        scores = embedder.rerank(queries[0], [c.get("chunk_text", "") for c in candidates])
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    except Exception as e:
        logger.warning("Multi-query reranking failed — falling back to per-variant fusion order: %s", e)

    out = []
    for h in candidates[:top_k]:
        h = dict(h)
        h["chunk_id"] = h["id"]
        h["similarity"] = h.get("rerank_score", h.get("score", 0.0))
        h["search_type"] = mode
        out.append(h)
    return out


def retrieve_multi(
    queries: list[str],
    tenant_id: str,
    org_unit_id: str,
    top_k: int = 5,
    doc_filter: Optional[str] = None,
    role_filter: Optional[str] = None,
    is_ground_truth: Optional[bool] = None,
) -> list[dict]:
    """Multi-query variant of retrieve() — see multi_query_search()'s docstring for the merge/rerank strategy."""
    query_filter = build_filter(
        tenant_id=tenant_id, org_unit_id=org_unit_id,
        doc_name=doc_filter, role=role_filter, is_ground_truth=is_ground_truth,
    )
    results = multi_query_search(queries=queries, query_filter=query_filter, mode="hybrid", top_k=top_k)
    if not results:
        logger.info("No chunks found for multi-query: %s", queries[0][:50] if queries else "")
    return results
