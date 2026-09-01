"""
api/routes/web.py
─────────────────────────────────────────────────────────────────
Rewritten 2026-08-21 (Qdrant migration). Same single endpoint, same three
modes, same response contract (additive: PageData gains `status`). Scraping
+ registration still happens INLINE (the page is fetched, saved, and a
Document row created synchronously — same as before); chunking/embedding/
summarising/image-captioning now happens in a BackgroundTasks job
(pipeline/ingest.py::process_web_document), matching documents.py's async
upload pattern and IMPLEMENTATION_PLAN (2).md's decision table
("Web scrape: scrape + register inline, chunk/embed/summarise/caption in
background").

Scraping itself is unchanged — pipeline/scraper.py still uses Playwright
(headless Chromium) to fetch pages, which already fully executes
JavaScript/renders dynamic content before handing HTML back here.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends

from api.dependencies import get_tenant_id, get_org_unit_id
from api.schemas.web import WebScrapeRequest, WebScrapeResponse, PageResult, PageData
from pipeline.scraper import scrape_url_async, scrape_website_async
from pipeline.storage import check_duplicate, register_document
from pipeline import ingest
from utils.ssrf_guard import is_safe_scrape_target
from config.settings import get_settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


def _to_utc_datetime(d: Optional[date]) -> Optional[datetime]:
    if d is None:
        return None
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    """datetime -> ISO 8601 string for Qdrant payload fields (see pipeline/ingest.py's effective_from/effective_to docstring)."""
    return dt.isoformat() if dt else None


def _web_storage_dir(tenant_id: str, org_unit_id: str) -> Path:
    d = Path(settings.UPLOAD_DIR) / tenant_id / org_unit_id / "web"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _page_data_from_record(record, url: str, message: Optional[str] = None) -> PageData:
    return PageData(
        id=record.id, doc_name=record.doc_name, url=url,
        summary_text=record.summary_text or "",
        key_topics=(record.key_topics.split(", ") if record.key_topics else None),
        page_count=record.page_count or 0, chunk_count=record.chunk_count or 0,
        source_path=record.source_path, language=record.language,
        model_used=record.model_used, doc_hash=record.doc_hash,
        tenant_id=record.tenant_id, org_unit_id=record.org_unit_id,
        category=record.category, effective_from=record.effective_from, effective_to=record.effective_to,
        is_ground_truth=record.is_ground_truth, preview_link=url, status=record.status,
        created_at=record.created_at, updated_at=record.updated_at,
        elapsed_sec=0.0, message=message,
    )


async def _register_and_schedule(
    url: str,
    background_tasks: BackgroundTasks,
    tenant_id: str, org_unit_id: str,
    category: Optional[str], effective_from: Optional[datetime], effective_to: Optional[datetime],
    is_ground_truth: bool,
    wait_for_selector: Optional[str], extra_wait_ms: int,
) -> tuple[str, PageData]:
    """Scrape one page (Playwright, already-rendered HTML) -> register -> schedule the background ingest job. Returns (status, PageData) — status is 'success'|'duplicate'|'error' for PageResult."""
    safe, reason = is_safe_scrape_target(url)
    if not safe:
        raise RuntimeError(f"Blocked: {reason}")

    try:
        docs = await scrape_url_async(url=url, wait_for_selector=wait_for_selector, extra_wait_ms=extra_wait_ms)
    except Exception as e:
        logger.exception("Scrape failed for %s: %s", url, e)
        raise RuntimeError("Scrape failed. Contact support with the X-Request-ID response header if this persists.")

    file_hash = docs[0].metadata.get("file_hash", "")
    page_title = docs[0].metadata.get("title") or url
    raw_html = docs[0].metadata.get("raw_html", "")
    markdown_text = docs[0].page_content

    # Phase H #6 — everything from here on (DB lookups, disk writes) could
    # raise something that leaks internal detail (a connection string, a
    # filesystem path) if left uncaught; wrap it so the caller always gets
    # a safe, generic message regardless of what actually failed.
    try:
        existing = check_duplicate(file_hash, tenant_id, org_unit_id) if file_hash else None
        if existing:
            if existing.status == "FAILED" and existing.stored_path:
                from pipeline.storage import set_status
                set_status(existing.id, tenant_id, org_unit_id, "PROCESSING")
                background_tasks.add_task(
                    ingest.process_web_document, str(existing.id), tenant_id, org_unit_id,
                    markdown_text, raw_html, page_title, file_hash, url, is_ground_truth, {},
                    _iso(existing.effective_from), _iso(existing.effective_to),
                    body.caption_images,
                )
            return "duplicate", _page_data_from_record(existing, url, message="Duplicate — returning cached summary")

        dest_dir = _web_storage_dir(tenant_id, org_unit_id)
        stem = uuid4()
        (dest_dir / f"{stem}.md").write_text(markdown_text, encoding="utf-8")
        if raw_html:
            (dest_dir / f"{stem}.html").write_text(raw_html, encoding="utf-8")

        record, _created = register_document(
            doc_name=page_title, file_hash=file_hash, tenant_id=tenant_id, org_unit_id=org_unit_id,
            file_type="md", file_size=len(markdown_text.encode("utf-8")), source="web",
            source_path=url, stored_path=str(dest_dir / f"{stem}.md"),
            category=category, effective_from=effective_from, effective_to=effective_to,
            is_ground_truth=is_ground_truth, metadata={},
        )

        background_tasks.add_task(
            ingest.process_web_document, str(record.id), tenant_id, org_unit_id,
            markdown_text, raw_html, page_title, file_hash, url, is_ground_truth, {},
            _iso(effective_from), _iso(effective_to),
            body.caption_images,
        )

        return "success", _page_data_from_record(record, url)

    except Exception as e:
        logger.exception("Registration failed for %s: %s", url, e)
        raise RuntimeError("Registration failed. Contact support with the X-Request-ID response header if this persists.")


@router.post(
    "/scrape",
    response_model=WebScrapeResponse,
    status_code=207,
    summary="Scrape URLs or crawl a full website",
    description="""
Single endpoint for all web scraping modes:

| Scenario | Request |
|---|---|
| Scrape 1 specific page | `urls: ["https://..."]` |
| Scrape 2–10 specific pages | `urls: ["https://...", "https://..."]` |
| Crawl entire website | `urls: ["https://..."], full_site: true` |

Pages are registered synchronously; text/image processing runs in the
background — poll `GET /documents/{id}` for `status`.
    """,
)
async def scrape(
    body: WebScrapeRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(get_tenant_id),
    org_unit_id: str = Depends(get_org_unit_id),
):
    results: List[PageResult] = []
    effective_from = _to_utc_datetime(body.effective_from)
    effective_to = _to_utc_datetime(body.effective_to)

    if body.full_site:
        url0 = str(body.urls[0])
        safe, reason = is_safe_scrape_target(url0)
        if not safe:
            raise HTTPException(status_code=422, detail=f"Blocked: {reason}")

        logger.info("Full site crawl: %s (max_pages=%d, tenant=%s org_unit=%s)", url0, body.max_pages, tenant_id, org_unit_id)
        try:
            urls = await scrape_website_async(
                url=url0, max_pages=body.max_pages, same_domain_only=body.same_domain_only,
                wait_for_selector=body.wait_for_selector, extra_wait_ms=body.extra_wait_ms,
                # scrape_website_async returns already-loaded Documents in
                # the pre-rewrite version — see note below.
            )
        except Exception as e:
            logger.exception("Crawl failed for %s: %s", url0, e)
            raise HTTPException(status_code=500, detail="Crawl failed. Contact support with the X-Request-ID response header if this persists.")

        if not urls:
            raise HTTPException(status_code=404, detail="No pages found at the given URL.")

        # scrape_website_async (pipeline/scraper.py, unchanged by this
        # rewrite) returns a list of already-scraped LangChain Documents,
        # not raw URLs — register/schedule each one directly instead of
        # re-scraping via _register_and_schedule (which scrapes by URL).
        for doc in urls:
            page_url = doc.metadata.get("source", "unknown")
            file_hash = doc.metadata.get("file_hash", "")
            page_title = doc.metadata.get("title") or page_url
            raw_html = doc.metadata.get("raw_html", "")
            markdown_text = doc.page_content

            existing = check_duplicate(file_hash, tenant_id, org_unit_id) if file_hash else None
            if existing:
                results.append(PageResult(url=page_url, status="duplicate", detail="Already in knowledge base",
                                           data=_page_data_from_record(existing, page_url)))
                continue

            try:
                dest_dir = _web_storage_dir(tenant_id, org_unit_id)
                stem = uuid4()
                (dest_dir / f"{stem}.md").write_text(markdown_text, encoding="utf-8")
                if raw_html:
                    (dest_dir / f"{stem}.html").write_text(raw_html, encoding="utf-8")

                record, _created = register_document(
                    doc_name=page_title, file_hash=file_hash, tenant_id=tenant_id, org_unit_id=org_unit_id,
                    file_type="md", file_size=len(markdown_text.encode("utf-8")), source="web",
                    source_path=page_url, stored_path=str(dest_dir / f"{stem}.md"),
                    category=body.category, effective_from=effective_from, effective_to=effective_to,
                    is_ground_truth=body.is_ground_truth, metadata={},
                )
                background_tasks.add_task(
                    ingest.process_web_document, str(record.id), tenant_id, org_unit_id,
                    markdown_text, raw_html, page_title, file_hash, page_url, body.is_ground_truth, {},
                    _iso(effective_from), _iso(effective_to),
                    body.caption_images,
                )
                results.append(PageResult(url=page_url, status="success", data=_page_data_from_record(record, page_url)))
            except Exception as e:
                logger.exception("Registration failed for %s: %s", page_url, e)
                results.append(PageResult(url=page_url, status="error", detail="Registration failed. Contact support with the X-Request-ID response header if this persists."))

        mode = "full_site"

    else:
        for url_obj in body.urls:
            url = str(url_obj)
            try:
                status, page_data = await _register_and_schedule(
                    url, background_tasks, tenant_id, org_unit_id,
                    body.category, effective_from, effective_to, body.is_ground_truth,
                    body.wait_for_selector, body.extra_wait_ms,
                )
                results.append(PageResult(url=url, status=status, data=page_data))
            except Exception as e:
                results.append(PageResult(url=url, status="error", detail=str(e)))

        mode = "single" if len(body.urls) == 1 else "batch"

    succeeded = sum(1 for r in results if r.status == "success")
    duplicates = sum(1 for r in results if r.status == "duplicate")
    failed = sum(1 for r in results if r.status == "error")

    return WebScrapeResponse(
        mode=mode,
        summary={"total": len(results), "succeeded": succeeded, "duplicates": duplicates, "failed": failed},
        results=results,
    )
