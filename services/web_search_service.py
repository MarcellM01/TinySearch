from __future__ import annotations

import asyncio
import html as _html
import json
import re
import socket
import sys
import urllib.error
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlencode, urlparse
from urllib.request import Request, urlopen


@lru_cache(maxsize=1)
def _async_web_crawler_cls() -> Any:
    from crawl4ai import AsyncWebCrawler

    return AsyncWebCrawler


def _ensure_utf8_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # py311+
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


@dataclass(frozen=True)
class SearchResult:
    result_id: int
    title: str
    url: str
    text: str


class SearchBackendError(Exception):
    """Base error raised when a search backend fails to produce results."""


class SearchBackendUnavailable(SearchBackendError):
    """Network, timeout, non-200, or non-JSON response from a backend."""


class SearchBackendBlocked(SearchBackendError):
    """Backend rejected the request (HTTP 403/429 or CAPTCHA/challenge page)."""


ALLOWED_SEARCH_BACKENDS: frozenset[str] = frozenset({"searxng", "duckduckgo", "auto"})
DEFAULT_SEARXNG_URL = "http://searxng:8080/search"
_DEFAULT_DDG_TIMEOUT = 20.0
_DEFAULT_SEARXNG_TIMEOUT = 8.0

_DDG_CHALLENGE_MARKERS: tuple[str, ...] = (
    "anomaly-modal",
    "anomaly_modal",
    "captcha-container",
    "challenge-form",
    "automated queries",
    "unusual traffic",
)


def normalize_domain(value: str) -> str:
    raw = value.strip().lower()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw or raw.startswith("//") else f"//{raw}")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host or any(char.isspace() for char in host):
        return ""
    return host.removeprefix("www.")


def is_blocked_domain(url: str, blocked_domains: Iterable[str]) -> bool:
    host = normalize_domain(url)
    if not host:
        return False
    for blocked in blocked_domains:
        blocked_host = normalize_domain(blocked)
        if not blocked_host:
            continue
        if host == blocked_host or host.endswith(f".{blocked_host}"):
            return True
    return False


def filter_blocked_search_results(
    search_results: list[SearchResult],
    blocked_domains: Iterable[str],
) -> list[SearchResult]:
    return [
        result
        for result in search_results
        if not is_blocked_domain(result.url, blocked_domains)
    ]


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]*>", "", s)


def _extract_links_from_html(html: str) -> list[str]:
    # Very small/fast extractor; good enough for basic crawling.
    return list(dict.fromkeys(re.findall(r'href="([^"]+)"', html, flags=re.IGNORECASE)))


def _decode_duckduckgo_href(href: str) -> str:
    # DuckDuckGo HTML endpoint often wraps links as /l/?uddg=<urlencoded>
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    if "uddg" in qs and qs["uddg"]:
        return unquote(qs["uddg"][0])
    return href


def _http_get(url: str, *, timeout: float = _DEFAULT_DDG_TIMEOUT) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "TinySearch/0.1 (+https://html.duckduckgo.com/html/)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = getattr(resp.headers, "get_content_charset", lambda default=None: None)("utf-8") or "utf-8"
    return raw.decode(charset, errors="replace")


def _looks_like_ddg_challenge(html: str) -> bool:
    lowered = html.lower()
    return any(marker in lowered for marker in _DDG_CHALLENGE_MARKERS)


async def crawl(url: str) -> dict:
    """
    Crawl a page with crawl4ai and return basic extracted content.

    Returns a dict with: url, markdown, html, links.
    """
    _ensure_utf8_stdio()
    AsyncWebCrawler = _async_web_crawler_cls()
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    html = getattr(result, "html", "") or ""
    markdown = getattr(result, "markdown", "") or ""
    links = _extract_links_from_html(html) if html else []
    return {"url": url, "markdown": markdown, "html": html, "links": links}


def _duckduckgo_search(query: str, limit: int) -> list[SearchResult]:
    """Query DuckDuckGo's HTML endpoint and return the top results."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        html = _http_get(url, timeout=_DEFAULT_DDG_TIMEOUT)
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            raise SearchBackendBlocked(
                f"DuckDuckGo refused the request (HTTP {exc.code})"
            ) from exc
        raise SearchBackendUnavailable(
            f"DuckDuckGo returned HTTP {exc.code}"
        ) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
        raise SearchBackendUnavailable(
            f"DuckDuckGo unreachable: {exc}"
        ) from exc

    if _looks_like_ddg_challenge(html):
        raise SearchBackendBlocked(
            "DuckDuckGo returned a CAPTCHA/challenge page"
        )

    # DDG HTML results use anchors like: <a class="result__a" href="...">Title</a>
    matches = re.findall(
        r'<a[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    snippets = re.findall(
        r'<a[^>]*class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</a>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    out: list[SearchResult] = []
    for idx, (href, title_html) in enumerate(matches):
        result_id = idx + 1
        title = _html.unescape(_strip_tags(title_html)).strip()
        target = _decode_duckduckgo_href(_html.unescape(href)).strip()
        text = ""
        if idx < len(snippets):
            text = _html.unescape(_strip_tags(snippets[idx])).strip()
        if not title or not target:
            continue
        out.append(SearchResult(result_id=result_id, title=title, url=target, text=text))
        if len(out) >= limit:
            break

    return out


def _normalize_engines(engines: Any) -> str:
    if engines is None:
        return ""
    if isinstance(engines, str):
        parts = [part.strip() for part in engines.split(",")]
    elif isinstance(engines, Sequence):
        parts = [str(part).strip() for part in engines]
    else:
        parts = [str(engines).strip()]
    return ",".join(part for part in parts if part)


def _searxng_search(
    query: str,
    limit: int,
    *,
    url: str,
    engines: Any = None,
    region: str | None = None,
    timeout: float = _DEFAULT_SEARXNG_TIMEOUT,
) -> list[SearchResult]:
    """Query a SearXNG-compatible JSON endpoint."""
    if not url or not url.strip():
        raise SearchBackendUnavailable("SearXNG search_backend_url is empty")

    params: list[tuple[str, str]] = [
        ("q", query),
        ("format", "json"),
        ("pageno", "1"),
    ]
    engines_str = _normalize_engines(engines)
    if engines_str:
        params.append(("engines", engines_str))
    if region:
        params.append(("language", str(region)))

    full_url = f"{url}?{urlencode(params)}"
    req = Request(
        full_url,
        headers={
            "User-Agent": "TinySearch/0.1 (+searxng)",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            content_type = resp.headers.get("Content-Type", "") or ""
    except urllib.error.HTTPError as exc:
        raise SearchBackendUnavailable(
            f"SearXNG returned HTTP {exc.code} from {url}"
        ) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
        raise SearchBackendUnavailable(
            f"SearXNG unreachable at {url}: {exc}"
        ) from exc

    text = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SearchBackendUnavailable(
            "SearXNG did not return JSON "
            f"(content-type={content_type!r}). "
            "Enable JSON output by adding 'json' to search.formats in searxng settings.yml."
        ) from exc

    if not isinstance(payload, dict):
        raise SearchBackendUnavailable("SearXNG JSON payload was not an object")

    raw_results = payload.get("results") or []
    if not isinstance(raw_results, list):
        raise SearchBackendUnavailable("SearXNG 'results' field was not a list")

    out: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        target = str(item.get("url") or "").strip()
        text_field = str(item.get("content") or "").strip()
        if not title or not target:
            continue
        out.append(
            SearchResult(
                result_id=len(out) + 1,
                title=title,
                url=target,
                text=text_field,
            )
        )
        if len(out) >= limit:
            break

    return out


def _load_search_config() -> dict[str, Any]:
    # Lazy import to avoid a circular dependency with research_config_service.
    from services.research_config_service import load_research_config

    return load_research_config()


def _dispatch_search(
    query: str,
    limit: int,
    *,
    config: dict[str, Any],
) -> list[SearchResult]:
    backend = str(config.get("search_backend") or "searxng").strip().lower()
    if backend not in ALLOWED_SEARCH_BACKENDS:
        backend = "searxng"
    url = str(config.get("search_backend_url") or DEFAULT_SEARXNG_URL)
    engines = config.get("search_engines")
    region = (
        config.get("search_region")
        or config.get("search_country")
        or ""
    )
    fallback_enabled = bool(config.get("search_backend_fallback", True))

    if backend == "duckduckgo":
        return _duckduckgo_search(query, limit)

    if backend == "auto":
        try:
            return _searxng_search(
                query, limit, url=url, engines=engines, region=str(region) or None
            )
        except SearchBackendError:
            return _duckduckgo_search(query, limit)

    # backend == "searxng"
    try:
        return _searxng_search(
            query, limit, url=url, engines=engines, region=str(region) or None
        )
    except SearchBackendError:
        if fallback_enabled:
            return _duckduckgo_search(query, limit)
        raise


def search(query: str, limit: int = 10) -> list[SearchResult]:
    """
    Run a web search using the configured backend.

    Returns items shaped like:
      Title:
      URL:
      Text:
    """
    config = _load_search_config()
    return _dispatch_search(query, limit, config=config)


def search_to_markdown(search_results: list[SearchResult]) -> str:
    markdown = ""
    for result in search_results:
        markdown += f"## {result.result_id}. {result.title}\n"
        markdown += f"URL: {result.url}\n"
        markdown += f"Text: {result.text}\n\n"
    return markdown

def search_markdown(query: str, limit: int = 10) -> str:
    search_results = search(query, limit)
    return search_to_markdown(search_results)

if __name__ == "__main__":
    _ensure_utf8_stdio()
    print(search_markdown("What is model context protocol?", limit=10))

    crawl_result = asyncio.run(crawl("https://example.com"))
    print(crawl_result.get("markdown", "")[:500])
