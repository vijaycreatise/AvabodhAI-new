"""
pipeline/scraper.py
───────────────────────────────────────────────────────────────
Playwright-based web scraper for AvabodhAI.

TWO MODES:
  - single : scrape one specific page (fast, focused)
  - crawl  : follow all internal links from a root URL (full site)

Both return a list of LangChain Document objects — same structure
as pipeline/loader.py. Everything after Step 1 is unchanged.

Handles:
  ✅ JS-rendered pages (React, Vue, Angular SPAs)
  ✅ Math / LaTeX  (MathML alttext, KaTeX, MathJax)
  ✅ Code blocks   (fenced ``` with language tag)
  ✅ Images        (absolute URLs)
  ✅ Tables        (clean Markdown tables)
  ✅ Noise removal (nav, footer, cookie banners, ads, sidebar)
  ✅ Main content isolation
  ✅ Dedup-math    (removes doubled formula rendering)
  ✅ Same-domain enforcement for crawl mode
  ✅ Page limit safety cap for crawl mode
  ✅ CONCURRENT crawling — pages fetched in parallel batches,
     not one-by-one (was the main bottleneck: 30 pages took ~3 min
     sequentially; with concurrency it drops to ~15-20s)

Install (add to requirements.txt):
    playwright
    markdownify
    beautifulsoup4
    lxml

After install run once:
    playwright install chromium
"""

import asyncio
import hashlib
import os
import re
import sys
import time
from collections import deque
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment
from langchain_core.documents import Document
from markdownify import markdownify as md

from utils.logger import get_logger
from utils.ssrf_guard import is_safe_scrape_target

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

_NOISE_SELECTORS = [
    "nav", "header", "footer",
    "[role='navigation']", "[role='banner']", "[role='contentinfo']",
    "#mw-navigation", "#mw-head", "#mw-panel", "#mw-page-base",
    "#mw-head-base", "#catlinks", "#footer", ".navbox", ".ambox",
    ".mw-editsection", ".mw-indicators", ".sistersitebox",
    ".refbegin", "#toc", ".mw-empty-elt", "#coordinates",
    "[id*='cookie']", "[class*='cookie']",
    "[id*='consent']", "[class*='consent']",
    "[id*='gdpr']", "[class*='gdpr']",
    "[id*='banner']", "[class*='banner']",
    "[id*='popup']", "[class*='popup']",
    "[id*='ad-']", "[class*='advertisement']", "[class*='sponsored']",
    ".toc-mobile", ".sidebar", "[data-sidebar]",
    ".DocSearch", ".navbar", ".pagination",
    "script", "style", "noscript", "iframe",
    "[aria-hidden='true']",
]

_MAIN_SELECTORS = [
    "main[id]", "article",
    "#mw-content-text",
    "#bodyContent",
    "main",
    "[role='main']",
    ".main-content", ".content", ".post-content", "#content",
]

# File extensions to skip during crawl
_SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".zip", ".tar", ".gz", ".css", ".js", ".xml", ".json",
    ".mp4", ".mp3", ".wav", ".avi", ".mov", ".doc", ".docx", ".xls", ".xlsx",
)

# ── Chromium launch args required for running inside Docker ──
_CHROMIUM_DOCKER_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",   # avoids crashes from Docker's 64MB /dev/shm limit
    "--disable-gpu",
    "--no-first-run",
    "--no-zygote",
]

# ── NEW — concurrency control for crawl mode ───────────────────
# How many pages to fetch simultaneously. 6-8 is a safe default:
# fast enough to matter, gentle enough not to look like a DoS attack
# or get rate-limited/blocked by the target site.
_DEFAULT_CRAWL_CONCURRENCY = 6


# ──────────────────────────────────────────────────────────────
# HTML → Markdown helpers (shared by both modes) — UNCHANGED
# ──────────────────────────────────────────────────────────────

def _replace_math(soup: BeautifulSoup) -> BeautifulSoup:
    """Preserve LaTeX from MathML / KaTeX / MathJax before markdownify strips it."""
    for tag in soup.find_all("math"):
        latex = tag.get("alttext", "").strip()
        if not latex:
            ann = tag.find("annotation", encoding="application/x-tex")
            if ann:
                latex = ann.get_text(strip=True)
        if latex:
            is_block = tag.get("display") == "block"
            repl = f"\n$$\n{latex}\n$$\n" if is_block else f"${latex}$"
            tag.replace_with(soup.new_string(repl))

    for img in soup.find_all("img", class_=re.compile(r"mwe-math")):
        latex = img.get("alt", "").strip()
        if latex:
            classes = " ".join(img.get("class", []))
            is_block = "display" in classes or "block" in classes
            repl = f"\n$$\n{latex}\n$$\n" if is_block else f"${latex}$"
            img.replace_with(soup.new_string(repl))

    for span in soup.find_all("span", class_=re.compile(r"katex|mathjax|math-inline|math-display")):
        latex = span.get("data-latex", "") or span.get("title", "")
        if latex:
            is_block = "display" in " ".join(span.get("class", []))
            repl = f"\n$$\n{latex}\n$$\n" if is_block else f"${latex}$"
            span.replace_with(soup.new_string(repl))

    return soup


def _tag_code_blocks(soup: BeautifulSoup) -> BeautifulSoup:
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        if code:
            classes = " ".join(code.get("class", []))
            m = re.search(r"language-(\w+)|lang-(\w+)|hljs-(\w+)", classes)
            if m:
                lang = m.group(1) or m.group(2) or m.group(3)
                code["class"] = [f"language-{lang}"]
    return soup


def _absolutify_images(soup: BeautifulSoup, base_url: str) -> BeautifulSoup:
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if src and not src.startswith(("http://", "https://", "data:")):
            img["src"] = urljoin(base, src)
        if not img.get("alt"):
            img["alt"] = os.path.basename(img.get("src", "image"))
    return soup


def _remove_noise(soup: BeautifulSoup) -> BeautifulSoup:
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    for sel in _NOISE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    return soup


def _extract_main(soup: BeautifulSoup) -> BeautifulSoup:
    for sel in _MAIN_SELECTORS:
        el = soup.select_one(sel)
        if el and len(el.get_text(strip=True)) > 500:
            return el
    return soup.find("body") or soup


def _clean_markdown(text: str) -> str:
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if re.match(r'^\[edit\]$|^\[hide\]$|^\[show\]$|^\[±\]$', s):
            continue
        if re.match(
            r'^Retrieved from|^This page was last edited|^Privacy policy'
            r'|^Terms of Use|^Cookie', s
        ):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    def _fix(m):
        return m.group(0).replace(r'\_', '_').replace(r'\^', '^')

    text = re.sub(r'\$\$.*?\$\$', _fix, text, flags=re.DOTALL)
    text = re.sub(r'\$[^$\n]+\$', _fix, text)
    text = re.sub(r'(\$\$[^$]+?\$\$)\s*\1', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'(\$[^$\n]+?\$)\1', r'\1', text)

    # Fix \* → {\ast} for KaTeX compatibility
    text = text.replace(r'\*', r'{\ast}')

    return text.strip()


def _html_to_markdown(html: str, url: str) -> str:
    """Full HTML → clean Markdown pipeline."""
    soup = BeautifulSoup(html, "lxml")
    soup = _replace_math(soup)
    soup = _tag_code_blocks(soup)
    soup = _absolutify_images(soup, url)
    soup = _remove_noise(soup)
    main = _extract_main(soup)
    raw = md(
        str(main),
        heading_style="ATX",
        bullets="-",
        strip=["script", "style", "nav", "header", "footer"],
    )
    return _clean_markdown(raw)


def _make_document(html: str, url: str, title: str, elapsed: float) -> Document:
    """Convert raw HTML into a LangChain Document."""
    markdown_text = _html_to_markdown(html, url)
    if not markdown_text.strip():
        raise ValueError(f"Empty content after cleaning: {url}")
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    return Document(
        page_content=markdown_text,
        metadata={
            "source":      url,
            "title":       title,
            "file_hash":   url_hash,
            "source_type": "web",
            "scraped_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_sec": elapsed,
            "page_count":  1,
            "raw_html": html,
        },
    )


def _should_skip_url(url: str) -> bool:
    """Return True if the URL should be ignored during crawl."""
    u = url.lower().split("?")[0].split("#")[0]
    return any(u.endswith(ext) for ext in _SKIP_EXTENSIONS)


def _normalize_url(url: str) -> str:
    """Strip fragment and trailing slash for consistent dedup."""
    return url.split("#")[0].rstrip("/")


# ──────────────────────────────────────────────────────────────
# MODE 1 — Single page — UNCHANGED
# ──────────────────────────────────────────────────────────────

async def _install_ssrf_route_guard(page) -> None:
    """
    Abort any request whose target resolves to a private/internal address.

    utils/ssrf_guard.py checks the URL we were GIVEN, before navigation. A
    browser then follows redirects, so a public host answering
    302 -> http://169.254.169.254/... reaches the metadata endpoint anyway:
    the pre-flight check and the actual fetch are two separate resolutions
    and only the first was guarded. DNS rebinding (a short-TTL record that
    answers public at check time and internal at fetch time) defeats it the
    same way.

    Playwright's routing fires per REQUEST, including each redirect hop, so
    the check happens where the connection is actually about to be made.
    Subresources are covered too - an <img src="http://10.0.0.1/..."> can no
    longer probe the internal network on the page's behalf.

    No-ops when SCRAPER_ALLOW_PRIVATE_NETWORKS is set, matching the guard's
    own escape hatch.
    """
    async def _route(route, request):
        safe, reason = is_safe_scrape_target(request.url)
        if safe:
            await route.continue_()
        else:
            logger.warning("Blocked request to %s during scrape - %s", request.url, reason)
            await route.abort()

    await page.route("**/*", _route)


async def _scrape_url_impl(
    url: str,
    wait_for_selector: str | None = None,
    extra_wait_ms: int = 1500,
    timeout_ms: int = 30_000,
) -> list[Document]:
    """Scrape a single URL with Playwright. Returns [Document]."""
    from playwright.async_api import async_playwright

    logger.info("Single scrape: %s", url)
    start = time.time()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=_CHROMIUM_DOCKER_ARGS,
        )
        try:
            context = await browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                java_script_enabled=True,
                locale="en-US",
            )
            page = await context.new_page()
            await _install_ssrf_route_guard(page)
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            if wait_for_selector:
                try:
                    await page.wait_for_selector(wait_for_selector, timeout=timeout_ms)
                except Exception:
                    logger.warning("Selector '%s' not found — continuing", wait_for_selector)

            await asyncio.sleep(extra_wait_ms / 1000)
            title = await page.title()
            html = await page.content()
        finally:
            await browser.close()

    elapsed = round(time.time() - start, 2)
    logger.info("Fetched %s in %.2fs (%d bytes)", url, elapsed, len(html))
    return [_make_document(html, url, title, elapsed)]


def scrape_url(
    url: str,
    wait_for_selector: str | None = None,
    extra_wait_ms: int = 1500,
    timeout_ms: int = 30_000,
) -> list[Document]:
    """Synchronous wrapper — scrape a single page."""
    return asyncio.run(
        scrape_url_async(url, wait_for_selector, extra_wait_ms, timeout_ms)
    )


# ──────────────────────────────────────────────────────────────
# MODE 2 — Full website crawl — NOW CONCURRENT
# ──────────────────────────────────────────────────────────────

async def _fetch_one_page(
    context,
    url: str,
    wait_for_selector: str | None,
    extra_wait_ms: int,
    timeout_ms: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None, str | None, float, list[str], Exception | None]:
    """
    Fetch a single page under semaphore-controlled concurrency.
    Returns (url, html, title, elapsed, discovered_links, error).
    Never raises — errors are returned, not thrown, so one bad page
    doesn't cancel the whole batch via asyncio.gather.
    """
    async with semaphore:
        page = await context.new_page()
        try:
            await _install_ssrf_route_guard(page)
            start = time.time()
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            if wait_for_selector:
                try:
                    await page.wait_for_selector(wait_for_selector, timeout=5_000)
                except Exception:
                    pass  # selector optional — continue anyway

            await asyncio.sleep(extra_wait_ms / 1000)

            title = await page.title()
            html = await page.content()
            elapsed = round(time.time() - start, 2)

            raw_links: list[str] = await page.eval_on_selector_all(
                "a[href]",
                "els => els.map(el => el.href)"
            )
            return (url, html, title, elapsed, raw_links, None)

        except Exception as e:
            return (url, None, None, 0.0, [], e)
        finally:
            await page.close()


async def _scrape_website_impl(
    url: str,
    max_pages: int = 50,
    same_domain_only: bool = True,
    wait_for_selector: str | None = None,
    extra_wait_ms: int = 1500,
    timeout_ms: int = 30_000,
    concurrency: int = _DEFAULT_CRAWL_CONCURRENCY,
) -> list[Document]:
    """
    Crawl a full website starting from root URL — CONCURRENT batches.

    - Follows all internal links (BFS order, level by level)
    - Within each BFS level, pages are fetched IN PARALLEL
      (bounded by `concurrency`) instead of one at a time
    - Stays within the same domain if same_domain_only=True
    - Stops after max_pages pages (safety cap)
    - Skips PDFs, images, JS/CSS files automatically
    - Each page becomes one Document (same as single mode)
    - Returns list of Documents — one per page scraped

    Why batches instead of one global gather():
    Link discovery is BFS — we don't know level-2 URLs until level-1
    pages are fetched. So we fetch the current frontier of up to
    `concurrency` URLs in parallel, collect their newly found links,
    then fetch the next frontier — repeating until max_pages or the
    queue is empty. This keeps BFS ordering while still parallelising
    the actual network/render work, which is where all the time goes.
    """
    from playwright.async_api import async_playwright

    root_domain = urlparse(url).netloc
    visited: set[str] = set()
    queue: deque[str] = deque([_normalize_url(url)])
    documents: list[Document] = []

    logger.info(
        "Starting website crawl: %s (max_pages=%d, concurrency=%d)",
        url, max_pages, concurrency,
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=_CHROMIUM_DOCKER_ARGS,
        )
        try:
            context = await browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                java_script_enabled=True,
                locale="en-US",
            )

            semaphore = asyncio.Semaphore(concurrency)

            while queue and len(documents) < max_pages:
                # ── Build the next batch of URLs to fetch in parallel ──────
                remaining_slots = max_pages - len(documents)
                batch: list[str] = []

                while queue and len(batch) < concurrency and len(batch) < remaining_slots:
                    candidate = queue.popleft()
                    if candidate in visited:
                        continue
                    if _should_skip_url(candidate):
                        continue
                    visited.add(candidate)
                    batch.append(candidate)

                if not batch:
                    break

                # ── Fetch the whole batch concurrently ─────────────────────
                results = await asyncio.gather(*[
                    _fetch_one_page(
                        context, batch_url, wait_for_selector,
                        extra_wait_ms, timeout_ms, semaphore,
                    )
                    for batch_url in batch
                ])

                # ── Process results, enqueue newly discovered links ────────
                for page_url, html, title, elapsed, raw_links, error in results:
                    if error is not None:
                        logger.warning("Failed to scrape %s: %s", page_url, error)
                        continue

                    try:
                        doc = _make_document(html, page_url, title, elapsed)
                        documents.append(doc)
                        logger.info(
                            "Crawled [%d/%d]: %s (%.2fs)",
                            len(documents), max_pages, page_url, elapsed,
                        )
                    except ValueError as e:
                        logger.warning("Skipping empty page %s: %s", page_url, e)
                        continue

                    for link in raw_links:
                        norm = _normalize_url(link)
                        if not norm.startswith("http"):
                            continue
                        if norm in visited:
                            continue
                        if _should_skip_url(norm):
                            continue
                        if same_domain_only and urlparse(norm).netloc != root_domain:
                            continue
                        # SSRF: only the SEED url was ever checked, by
                        # api/routes/web.py. Every link discovered here is
                        # attacker-influenced (it comes out of a fetched
                        # page), and with same_domain_only=False - a plain
                        # request field - the crawler would follow one to
                        # any host at all, including 169.254.169.254 or an
                        # internal service, and store what it found as a
                        # retrievable document. Checked per-URL here, at the
                        # point the target is actually chosen.
                        safe, reason = is_safe_scrape_target(norm)
                        if not safe:
                            logger.warning("Skipping crawl link %s - %s", norm, reason)
                            continue
                        queue.append(norm)

        finally:
            await browser.close()

    logger.info("Crawl complete — %d pages scraped from %s", len(documents), url)
    return documents



# ── Windows event-loop compatibility ─────────────────────────────────────────

def _loop_can_spawn_subprocesses() -> bool:
    """
    Can the CURRENT event loop start a subprocess?

    Playwright drives the browser through a child process, so it needs one.
    On Windows only ProactorEventLoop supports that - SelectorEventLoop
    raises NotImplementedError from create_subprocess_exec.

    That combination happens in normal development. uvicorn picks its loop
    like this (uvicorn/loops/asyncio.py):

        if sys.platform == "win32" and not use_subprocess:
            return asyncio.ProactorEventLoop
        return asyncio.SelectorEventLoop

    and use_subprocess is True whenever --reload is on. So `uvicorn main:app
    --reload` on Windows silently produces a loop that cannot run
    Playwright, and every /web/scrape call fails with a bare
    NotImplementedError that says nothing about the real cause.

    Always True on Linux/macOS, so this whole mechanism is inert in Docker
    and in production.
    """
    if sys.platform != "win32":
        return True
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return True
    return isinstance(loop, asyncio.ProactorEventLoop)


async def _run_with_subprocess_support(coro_factory):
    """
    Await a coroutine on a loop that can actually spawn subprocesses.

    When the running loop already can (every non-Windows deployment, and
    Windows without --reload), this just awaits it in place - no thread, no
    behaviour change.

    Otherwise the work runs on a dedicated thread with its own
    ProactorEventLoop, and this coroutine waits for that thread without
    blocking the server's loop. coro_factory is a callable rather than a
    coroutine because a coroutine object is bound to the loop that created
    it and cannot be handed to another one.
    """
    if _loop_can_spawn_subprocesses():
        return await coro_factory()

    logger.debug(
        "Current event loop cannot spawn subprocesses (Windows + SelectorEventLoop, "
        "typically 'uvicorn --reload') - running the browser on a dedicated Proactor loop."
    )

    def _runner():
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro_factory())
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    # to_thread propagates the return value AND any exception, so callers
    # see exactly what they would have seen from a direct await.
    return await asyncio.to_thread(_runner)


async def scrape_url_async(
    url: str,
    wait_for_selector: str | None = None,
    extra_wait_ms: int = 1500,
    timeout_ms: int = 30_000,
) -> list[Document]:
    """Scrape one page. See _scrape_url_impl for the real work."""
    return await _run_with_subprocess_support(
        lambda: _scrape_url_impl(url, wait_for_selector, extra_wait_ms, timeout_ms)
    )


async def scrape_website_async(
    url: str,
    max_pages: int = 50,
    same_domain_only: bool = True,
    wait_for_selector: str | None = None,
    extra_wait_ms: int = 1500,
    timeout_ms: int = 30_000,
    concurrency: int = _DEFAULT_CRAWL_CONCURRENCY,
) -> list[Document]:
    """Crawl a site. See _scrape_website_impl for the real work."""
    return await _run_with_subprocess_support(
        lambda: _scrape_website_impl(
            url, max_pages, same_domain_only, wait_for_selector,
            extra_wait_ms, timeout_ms, concurrency,
        )
    )

def scrape_website(
    url: str,
    max_pages: int = 50,
    same_domain_only: bool = True,
    wait_for_selector: str | None = None,
    extra_wait_ms: int = 1500,
    timeout_ms: int = 30_000,
    concurrency: int = _DEFAULT_CRAWL_CONCURRENCY,
) -> list[Document]:
    """Synchronous wrapper — crawl a full website."""
    return asyncio.run(
        scrape_website_async(
            url, max_pages, same_domain_only,
            wait_for_selector, extra_wait_ms, timeout_ms, concurrency,
        )
    )