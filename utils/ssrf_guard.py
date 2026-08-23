"""
utils/ssrf_guard.py
--------------------
Phase H #3 — SSRF guard for /web/scrape. Resolves the target hostname and
rejects private/loopback/link-local IPs before Playwright ever navigates
to it — this specifically closes the cloud-metadata-endpoint attack
(169.254.169.254) and internal-network-scanning-via-scraper class of bug.

Deliberately placed here (called from api/routes/web.py) rather than
inside pipeline/scraper.py itself — scraper.py is on the "reuse, don't
rewrite" list; the guard belongs at the trust boundary where untrusted
URLs first enter the system, not buried inside the fetching library.

Default-on (SCRAPER_ALLOW_PRIVATE_NETWORKS=false) — legitimate scrape
targets are public sites, so this doesn't change behavior for any real
caller. The env flag is an explicit escape hatch for anyone who
deliberately needs to scrape an internal target.
"""

import ipaddress
import socket
from urllib.parse import urlparse

from config.settings import get_settings

settings = get_settings()


def is_safe_scrape_target(url: str) -> tuple[bool, str]:
    """Returns (True, "") if the URL is safe to scrape, else (False, reason)."""
    if settings.SCRAPER_ALLOW_PRIVATE_NETWORKS:
        return True, ""

    try:
        parsed = urlparse(url)
    except Exception:
        return False, "URL could not be parsed"

    if parsed.scheme not in ("http", "https"):
        return False, f"Unsupported scheme '{parsed.scheme}'"

    hostname = parsed.hostname
    if not hostname:
        return False, "URL has no hostname"

    try:
        # getaddrinfo, not gethostbyname — handles IPv6 too, and returns
        # every address a hostname resolves to (a DNS-rebinding attempt
        # could return multiple IPs, some public, some not — reject if
        # ANY of them is private, don't just check the first).
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        return False, f"Could not resolve hostname: {e}"

    for info in addr_infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False, f"Resolved to a private/internal address ({ip_str}) — scraping internal network targets is not allowed"

    return True, ""
