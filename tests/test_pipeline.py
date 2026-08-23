"""
tests/test_pipeline.py
----------------------
Unit tests for the Qdrant-migration pipeline (2026-08-21 rewrite).
Rewritten because the previous version tested validate_summary()
(pipeline/storage.py — removed, chunk storage no longer needs a Pydantic
validator for pgvector rows) and split_documents() (pipeline/splitter.py —
deleted, replaced by pipeline/chunker.py). No network/DB/Qdrant connection
needed for any test below — everything here is pure-function logic.

Run with: pytest tests/ -v
"""

import pytest

from pipeline.chunker import clean_extracted_text, is_meaningful_chunk
from pipeline.retriever import build_filter, FILTERABLE_METADATA_KEYS
from pipeline.vector_store import chunk_point_id


# ── clean_extracted_text() ──────────────────────────────────────────────────

def test_clean_extracted_text_fixes_hyphenation():
    text = "This is a long word: Aper-\nture that got split."
    assert "Aperture" in clean_extracted_text(text)


def test_clean_extracted_text_removes_standalone_page_numbers():
    text = "Some real content.\n42\nMore real content."
    cleaned = clean_extracted_text(text)
    assert "\n42\n" not in cleaned


def test_clean_extracted_text_collapses_excess_newlines():
    text = "Paragraph one.\n\n\n\n\nParagraph two."
    assert "\n\n\n" not in clean_extracted_text(text)


# ── is_meaningful_chunk() ───────────────────────────────────────────────────

def test_is_meaningful_chunk_rejects_short_text():
    assert is_meaningful_chunk("Hi") is False


def test_is_meaningful_chunk_rejects_numbers_only():
    text = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20"
    assert is_meaningful_chunk(text) is False


def test_is_meaningful_chunk_accepts_real_prose():
    text = (
        "This is a properly sized chunk with enough real words and enough "
        "alphabetic content to pass every one of the noise filters that "
        "is_meaningful_chunk applies to a candidate chunk of text, and it "
        "has been padded out with a few more clauses so it clears the "
        "two-hundred-character minimum length check as well."
    )
    assert len(text) >= 200
    assert is_meaningful_chunk(text) is True


def test_is_meaningful_chunk_rejects_mostly_symbols():
    text = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~" * 10
    assert is_meaningful_chunk(text) is False


# ── pipeline/retriever.py::build_filter() — the tenant-isolation chokepoint ──

def test_build_filter_requires_tenant_and_defaults_org():
    f = build_filter(tenant_id="tenant-a", org_unit_id="dept-1")
    conditions = {c.key: c for c in f.must if hasattr(c, "key")}
    assert conditions["tenant_id"].match.value == "tenant-a"
    # org_unit_id defaults to [org_unit_id] when org_ids isn't supplied —
    # this is what stops a bare tenant match from also matching every
    # department within that tenant.
    assert conditions["org_unit_id"].match.any == ["dept-1"]


def test_build_filter_org_ids_widens_within_tenant_only():
    f = build_filter(tenant_id="tenant-a", org_unit_id="dept-1", org_ids=["dept-1", "dept-2"])
    conditions = {c.key: c for c in f.must if hasattr(c, "key")}
    assert set(conditions["org_unit_id"].match.any) == {"dept-1", "dept-2"}
    # tenant_id is never widened by org_ids — still exactly one tenant.
    assert conditions["tenant_id"].match.value == "tenant-a"


def test_build_filter_rejects_non_allowlisted_metadata_key():
    with pytest.raises(ValueError):
        build_filter(tenant_id="t", org_unit_id="o", metadata={"not_a_real_key": ["x"]})


def test_build_filter_accepts_allowlisted_metadata_key():
    key = next(iter(FILTERABLE_METADATA_KEYS))
    f = build_filter(tenant_id="t", org_unit_id="o", metadata={key: ["IN"]})
    conditions = {c.key: c for c in f.must if hasattr(c, "key")}
    assert conditions[f"meta_{key}"].match.any == ["IN"]


def test_build_filter_as_of_adds_effective_date_conditions():
    import datetime
    f = build_filter(tenant_id="t", org_unit_id="o", as_of=datetime.date(2026, 6, 1))
    # Two extra should-style Filter sub-conditions get appended (one for
    # effective_from, one for effective_to) beyond tenant_id + org_unit_id.
    assert len(f.must) == 4


# ── pipeline/vector_store.py::chunk_point_id() — idempotent upsert IDs ───────

def test_chunk_point_id_is_deterministic():
    id1 = chunk_point_id("doc-123", "text", 0)
    id2 = chunk_point_id("doc-123", "text", 0)
    assert id1 == id2


def test_chunk_point_id_differs_by_role_and_index():
    base = chunk_point_id("doc-123", "text", 0)
    assert chunk_point_id("doc-123", "image", 0) != base
    assert chunk_point_id("doc-123", "text", 1) != base
    assert chunk_point_id("doc-456", "text", 0) != base
