from __future__ import annotations

import unittest

from services.web_search_service import (
    SearchResult,
    filter_blocked_search_results,
    is_blocked_domain,
    normalize_domain,
)


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


if __name__ == "__main__":
    unittest.main()
