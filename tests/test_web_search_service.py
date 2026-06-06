from __future__ import annotations

import json
import socket
import unittest
import urllib.error
from typing import Any
from unittest.mock import patch

from services import web_search_service
from services.web_search_service import (
    DEFAULT_SEARXNG_URL,
    SearchBackendBlocked,
    SearchBackendError,
    SearchBackendUnavailable,
    SearchResult,
    _dispatch_search,
    _duckduckgo_search,
    _searxng_search,
    filter_blocked_search_results,
    is_blocked_domain,
    normalize_domain,
    search,
)


class _FakeHeaders:
    def __init__(self, content_type: str = "", charset: str | None = "utf-8") -> None:
        self._content_type = content_type
        self._charset = charset

    def get(self, key: str, default: str = "") -> str:
        if key.lower() == "content-type":
            return self._content_type
        return default

    def get_content_charset(self, default: str | None = None) -> str | None:
        return self._charset if self._charset is not None else default


class _FakeUrlopenResponse:
    def __init__(
        self,
        body: bytes | str,
        *,
        content_type: str = "application/json",
        charset: str = "utf-8",
    ) -> None:
        if isinstance(body, str):
            body = body.encode(charset)
        self._body = body
        self._headers = _FakeHeaders(content_type=content_type, charset=charset)

    def __enter__(self) -> "_FakeUrlopenResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def read(self) -> bytes:
        return self._body

    @property
    def headers(self) -> _FakeHeaders:
        return self._headers


def _make_urlopen_returning(
    body: bytes | str,
    *,
    content_type: str = "application/json",
    charset: str = "utf-8",
):
    def fake(req: Any, timeout: Any = None) -> _FakeUrlopenResponse:
        return _FakeUrlopenResponse(body, content_type=content_type, charset=charset)

    return fake


def _make_urlopen_raising(exc: BaseException):
    def fake(req: Any, timeout: Any = None) -> Any:
        raise exc

    return fake


class BlockedDomainTests(unittest.TestCase):
    def test_normalize_domain_accepts_bare_domains_and_urls(self) -> None:
        self.assertEqual(normalize_domain("Example.COM"), "example.com")
        self.assertEqual(normalize_domain("www.example.com"), "example.com")
        self.assertEqual(normalize_domain("https://www.example.com/path"), "example.com")

    def test_blocked_domain_matches_exact_www_and_subdomains(self) -> None:
        blocked = ["example.com"]

        self.assertTrue(is_blocked_domain("https://example.com/page", blocked))
        self.assertTrue(is_blocked_domain("https://www.example.com/page", blocked))
        self.assertTrue(is_blocked_domain("https://news.example.com/page", blocked))

    def test_blocked_domain_does_not_match_sibling_domain(self) -> None:
        self.assertFalse(
            is_blocked_domain("https://badexample.com/page", ["example.com"])
        )

    def test_blocked_domain_accepts_url_style_entries(self) -> None:
        self.assertTrue(
            is_blocked_domain(
                "https://news.example.com/page",
                ["https://www.example.com/anything"],
            )
        )

    def test_filter_blocked_search_results_keeps_allowed_results(self) -> None:
        results = [
            SearchResult(1, "Blocked", "https://blocked.example/page", "blocked"),
            SearchResult(2, "Allowed", "https://allowed.example/page", "allowed"),
        ]

        filtered = filter_blocked_search_results(results, ["blocked.example"])

        self.assertEqual([result.title for result in filtered], ["Allowed"])


class SearXNGBackendTests(unittest.TestCase):
    def test_searxng_json_response_maps_to_search_results(self) -> None:
        payload = {
            "results": [
                {
                    "title": "Async tasks in Python",
                    "url": "https://example.com/python",
                    "content": "Coroutines and tasks.",
                },
                {
                    "title": "Bread baking",
                    "url": "https://example.com/bread",
                    "content": "Flour and yeast.",
                },
            ]
        }
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning(json.dumps(payload)),
        ):
            results = _searxng_search(
                "python async", 5, url="http://searxng:8080/search"
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Async tasks in Python")
        self.assertEqual(results[0].url, "https://example.com/python")
        self.assertEqual(results[0].text, "Coroutines and tasks.")
        self.assertEqual(results[0].result_id, 1)
        self.assertEqual(results[1].result_id, 2)

    def test_searxng_respects_limit(self) -> None:
        payload = {
            "results": [
                {"title": f"r{i}", "url": f"https://example.com/{i}", "content": ""}
                for i in range(10)
            ]
        }
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning(json.dumps(payload)),
        ):
            results = _searxng_search("q", 3, url="http://searxng:8080/search")

        self.assertEqual(len(results), 3)

    def test_searxng_html_response_raises_unavailable_with_actionable_message(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning(
                "<html><body>not json</body></html>", content_type="text/html"
            ),
        ):
            with self.assertRaises(SearchBackendUnavailable) as cm:
                _searxng_search("q", 5, url="http://searxng:8080/search")

        message = str(cm.exception).lower()
        self.assertIn("json", message)
        self.assertIn("formats", message)

    def test_searxng_network_error_raises_unavailable(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_raising(urllib.error.URLError("connection refused")),
        ):
            with self.assertRaises(SearchBackendUnavailable):
                _searxng_search("q", 5, url="http://searxng:8080/search")

    def test_searxng_http_500_raises_unavailable(self) -> None:
        http_error = urllib.error.HTTPError(
            "http://searxng:8080/search", 500, "Server Error", {}, None
        )
        with patch.object(
            web_search_service, "urlopen", new=_make_urlopen_raising(http_error)
        ):
            with self.assertRaises(SearchBackendUnavailable):
                _searxng_search("q", 5, url="http://searxng:8080/search")

    def test_searxng_timeout_raises_unavailable(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_raising(socket.timeout("timed out")),
        ):
            with self.assertRaises(SearchBackendUnavailable):
                _searxng_search("q", 5, url="http://searxng:8080/search")

    def test_searxng_empty_results_list_is_not_an_error(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning(json.dumps({"results": []})),
        ):
            results = _searxng_search("q", 5, url="http://searxng:8080/search")
        self.assertEqual(results, [])


class DuckDuckGoBackendTests(unittest.TestCase):
    _RESULT_HTML = (
        "<html><body>"
        '<a class="result__a" href="https://example.com/page">Example Title</a>'
        '<a class="result__snippet">Example snippet.</a>'
        "</body></html>"
    )

    def test_ddg_returns_results_on_valid_html(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning(self._RESULT_HTML, content_type="text/html"),
        ):
            results = _duckduckgo_search("anything", 5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Example Title")
        self.assertEqual(results[0].url, "https://example.com/page")
        self.assertEqual(results[0].text, "Example snippet.")

    def test_ddg_valid_page_without_matches_returns_empty(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning(
                "<html><body><p>No results for your query.</p></body></html>",
                content_type="text/html",
            ),
        ):
            results = _duckduckgo_search("anything", 5)
        self.assertEqual(results, [])

    def test_ddg_403_raises_blocked(self) -> None:
        http_error = urllib.error.HTTPError(
            "https://html.duckduckgo.com/html/", 403, "Forbidden", {}, None
        )
        with patch.object(
            web_search_service, "urlopen", new=_make_urlopen_raising(http_error)
        ):
            with self.assertRaises(SearchBackendBlocked):
                _duckduckgo_search("anything", 5)

    def test_ddg_429_raises_blocked(self) -> None:
        http_error = urllib.error.HTTPError(
            "https://html.duckduckgo.com/html/", 429, "Too Many", {}, None
        )
        with patch.object(
            web_search_service, "urlopen", new=_make_urlopen_raising(http_error)
        ):
            with self.assertRaises(SearchBackendBlocked):
                _duckduckgo_search("anything", 5)

    def test_ddg_challenge_page_raises_blocked(self) -> None:
        challenge_html = (
            "<html><body><div class='anomaly-modal'>"
            "Please verify you are human</div></body></html>"
        )
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_returning(challenge_html, content_type="text/html"),
        ):
            with self.assertRaises(SearchBackendBlocked):
                _duckduckgo_search("anything", 5)

    def test_ddg_network_error_raises_unavailable(self) -> None:
        with patch.object(
            web_search_service,
            "urlopen",
            new=_make_urlopen_raising(urllib.error.URLError("no network")),
        ):
            with self.assertRaises(SearchBackendUnavailable):
                _duckduckgo_search("anything", 5)


class DispatcherTests(unittest.TestCase):
    _CONFIG_BASE: dict[str, Any] = {
        "search_backend": "searxng",
        "search_backend_url": "http://searxng:8080/search",
        "search_engines": [],
        "search_region": "",
        "search_backend_fallback": True,
    }

    def _config(self, **overrides: Any) -> dict[str, Any]:
        merged = dict(self._CONFIG_BASE)
        merged.update(overrides)
        return merged

    def test_default_backend_calls_searxng(self) -> None:
        searxng_calls: list[tuple[str, int]] = []
        ddg_calls: list[tuple[str, int]] = []

        def fake_searxng(query: str, limit: int, **_: Any) -> list[SearchResult]:
            searxng_calls.append((query, limit))
            return [SearchResult(1, "ok", "https://example.com/", "")]

        def fake_ddg(query: str, limit: int) -> list[SearchResult]:
            ddg_calls.append((query, limit))
            return []

        with patch.object(web_search_service, "_searxng_search", new=fake_searxng), patch.object(
            web_search_service, "_duckduckgo_search", new=fake_ddg
        ):
            results = _dispatch_search("q", 5, config=self._config())

        self.assertEqual(len(results), 1)
        self.assertEqual(len(searxng_calls), 1)
        self.assertEqual(ddg_calls, [])

    def test_searxng_failure_falls_back_to_ddg_when_enabled(self) -> None:
        def failing_searxng(*args: Any, **kwargs: Any) -> list[SearchResult]:
            raise SearchBackendUnavailable("searxng down")

        def fake_ddg(query: str, limit: int) -> list[SearchResult]:
            return [SearchResult(1, "DDG", "https://duck.example/", "snippet")]

        with patch.object(
            web_search_service, "_searxng_search", new=failing_searxng
        ), patch.object(web_search_service, "_duckduckgo_search", new=fake_ddg):
            results = _dispatch_search(
                "q", 5, config=self._config(search_backend_fallback=True)
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "DDG")

    def test_searxng_failure_raises_when_fallback_disabled(self) -> None:
        def failing_searxng(*args: Any, **kwargs: Any) -> list[SearchResult]:
            raise SearchBackendUnavailable("searxng down")

        def fake_ddg(query: str, limit: int) -> list[SearchResult]:
            self.fail("DDG should not be called when fallback is disabled")
            return []

        with patch.object(
            web_search_service, "_searxng_search", new=failing_searxng
        ), patch.object(web_search_service, "_duckduckgo_search", new=fake_ddg):
            with self.assertRaises(SearchBackendError):
                _dispatch_search(
                    "q", 5, config=self._config(search_backend_fallback=False)
                )

    def test_duckduckgo_backend_skips_searxng_entirely(self) -> None:
        def fake_searxng(*args: Any, **kwargs: Any) -> list[SearchResult]:
            self.fail("SearXNG should not be called for the duckduckgo backend")
            return []

        def fake_ddg(query: str, limit: int) -> list[SearchResult]:
            return [SearchResult(1, "DDG", "https://duck.example/", "snippet")]

        with patch.object(
            web_search_service, "_searxng_search", new=fake_searxng
        ), patch.object(web_search_service, "_duckduckgo_search", new=fake_ddg):
            results = _dispatch_search(
                "q", 5, config=self._config(search_backend="duckduckgo")
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "DDG")

    def test_auto_backend_falls_back_regardless_of_fallback_flag(self) -> None:
        def failing_searxng(*args: Any, **kwargs: Any) -> list[SearchResult]:
            raise SearchBackendUnavailable("nope")

        def fake_ddg(query: str, limit: int) -> list[SearchResult]:
            return [SearchResult(1, "DDG", "https://duck.example/", "")]

        with patch.object(
            web_search_service, "_searxng_search", new=failing_searxng
        ), patch.object(web_search_service, "_duckduckgo_search", new=fake_ddg):
            results = _dispatch_search(
                "q",
                5,
                config=self._config(
                    search_backend="auto", search_backend_fallback=False
                ),
            )

        self.assertEqual(len(results), 1)

    def test_invalid_backend_falls_back_to_default(self) -> None:
        def fake_searxng(*args: Any, **kwargs: Any) -> list[SearchResult]:
            return [SearchResult(1, "ok", "https://example.com/", "")]

        with patch.object(web_search_service, "_searxng_search", new=fake_searxng):
            results = _dispatch_search(
                "q", 5, config=self._config(search_backend="bogus")
            )

        self.assertEqual(len(results), 1)

    def test_search_default_uses_searxng(self) -> None:
        captured: dict[str, Any] = {}

        def fake_searxng(query: str, limit: int, *, url: str, **kwargs: Any) -> list[SearchResult]:
            captured["url"] = url
            captured["query"] = query
            return [SearchResult(1, "ok", "https://example.com/", "")]

        with patch.object(
            web_search_service, "_load_search_config", return_value=self._config()
        ), patch.object(web_search_service, "_searxng_search", new=fake_searxng):
            results = search("hello")

        self.assertEqual(len(results), 1)
        self.assertEqual(captured["url"], "http://searxng:8080/search")
        self.assertEqual(captured["query"], "hello")

    def test_default_searxng_url_constant_matches_issue(self) -> None:
        self.assertEqual(DEFAULT_SEARXNG_URL, "http://searxng:8080/search")


class ConfigCoercionTests(unittest.TestCase):
    def test_default_search_backend_is_searxng(self) -> None:
        from services.research_config_service import DEFAULT_RESEARCH_CONFIG

        self.assertEqual(DEFAULT_RESEARCH_CONFIG["search_backend"], "searxng")
        self.assertEqual(
            DEFAULT_RESEARCH_CONFIG["search_backend_url"],
            "http://searxng:8080/search",
        )
        self.assertTrue(DEFAULT_RESEARCH_CONFIG["search_backend_fallback"])

    def test_invalid_backend_raises(self) -> None:
        from services.research_config_service import _coerce_config

        with self.assertRaises(ValueError):
            _coerce_config({"search_backend": "yandex"})

    def test_country_aliases_to_region(self) -> None:
        from services.research_config_service import _coerce_config

        config = _coerce_config({"search_country": "us-en"})
        self.assertEqual(config["search_region"], "us-en")
        self.assertNotIn("search_country", config)

    def test_engines_comma_string_is_normalized_to_list(self) -> None:
        from services.research_config_service import _coerce_config

        config = _coerce_config({"search_engines": "google, bing , duckduckgo"})
        self.assertEqual(config["search_engines"], ["google", "bing", "duckduckgo"])

    def test_searxng_url_env_overrides_config(self) -> None:
        from services.research_config_service import _coerce_config

        with patch.dict("os.environ", {"SEARXNG_URL": "http://example.test/search"}):
            config = _coerce_config({"search_backend_url": "http://other:8080/"})
        self.assertEqual(config["search_backend_url"], "http://example.test/search")

    def test_duckduckgo_backend_preserves_existing_behavior(self) -> None:
        from services.research_config_service import _coerce_config

        config = _coerce_config({"search_backend": "duckduckgo"})
        self.assertEqual(config["search_backend"], "duckduckgo")


if __name__ == "__main__":
    unittest.main()
