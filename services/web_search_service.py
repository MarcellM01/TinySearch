import asyncio
import html as _html
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
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


def _http_get(url: str) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "TinySearch/0.1 (+https://html.duckduckgo.com/html/)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(req, timeout=20) as resp:
        raw = resp.read()
        charset = getattr(resp.headers, "get_content_charset", lambda default=None: None)("utf-8") or "utf-8"
    return raw.decode(charset, errors="replace")


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


def search(query: str, limit: int = 10) -> list[SearchResult]:
    """
    Query DuckDuckGo's HTML endpoint and return the top results.

    Returns items shaped like:
      Title:
      URL:
      Text:
    """
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    html = _http_get(url)

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
