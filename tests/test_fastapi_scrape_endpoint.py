from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from servers.fastapi_server import ScrapeRequest, scrape_endpoint
from services.scrape_service import (
    EmptyContentError,
    FetchFailedError,
    FetchTimeoutError,
    ScrapeResult,
    UnsupportedDocumentError,
)
from services.url_safety_service import BlockedUrlError, InvalidUrlError


def _result(metadata=None) -> ScrapeResult:
    return ScrapeResult(
        answer="URL-GROUNDED ANSWER PROMPT...",
        url="https://example.com/x",
        title="Title",
        query="q",
        content_tokens=42,
        answer_tokens=123,
        truncated=False,
        retrieved_at="2026-06-12T10:30:00Z",
        metadata=metadata,
    )


class ScrapeRequestValidationTests(unittest.TestCase):
    def test_rejects_empty_query(self) -> None:
        with self.assertRaises(ValidationError):
            ScrapeRequest(url="https://example.com/x", query="")

    def test_rejects_zero_max_tokens(self) -> None:
        with self.assertRaises(ValidationError):
            ScrapeRequest(url="https://example.com/x", query="q", max_tokens=0)

    def test_rejects_non_http_scheme(self) -> None:
        with self.assertRaises(ValidationError):
            ScrapeRequest(url="ftp://example.com/x", query="q")

    def test_defaults(self) -> None:
        req = ScrapeRequest(url="https://example.com/x", query="q")
        self.assertEqual(req.max_tokens, 4000)
        self.assertTrue(req.include_metadata)


class ScrapeEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_payload_with_metadata(self) -> None:
        scrape_mock = AsyncMock(
            return_value=_result(
                metadata={
                    "description": "d",
                    "author": "a",
                    "published_date": "2026-01-01",
                }
            )
        )
        with patch("servers.fastapi_server.scrape_url", scrape_mock), patch(
            "servers.fastapi_server._ensure_local_bundle_for_config",
            new_callable=AsyncMock,
        ):
            payload = await scrape_endpoint(
                ScrapeRequest(url="https://example.com/x", query="q")
            )

        self.assertEqual(payload["answer"], "URL-GROUNDED ANSWER PROMPT...")
        self.assertEqual(payload["url"], "https://example.com/x")
        self.assertEqual(payload["title"], "Title")
        self.assertEqual(payload["query"], "q")
        self.assertEqual(payload["content_tokens"], 42)
        self.assertEqual(payload["answer_tokens"], 123)
        self.assertFalse(payload["truncated"])
        self.assertEqual(payload["retrieved_at"], "2026-06-12T10:30:00Z")
        self.assertEqual(payload["metadata"]["author"], "a")

    async def test_omits_metadata_when_include_metadata_false(self) -> None:
        scrape_mock = AsyncMock(return_value=_result(metadata=None))
        with patch("servers.fastapi_server.scrape_url", scrape_mock), patch(
            "servers.fastapi_server._ensure_local_bundle_for_config",
            new_callable=AsyncMock,
        ):
            payload = await scrape_endpoint(
                ScrapeRequest(
                    url="https://example.com/x",
                    query="q",
                    include_metadata=False,
                )
            )

        self.assertNotIn("metadata", payload)


class ScrapeEndpointErrorMappingTests(unittest.IsolatedAsyncioTestCase):
    async def _run_with_exc(self, exc: Exception) -> HTTPException:
        scrape_mock = AsyncMock(side_effect=exc)
        with patch("servers.fastapi_server.scrape_url", scrape_mock), patch(
            "servers.fastapi_server._ensure_local_bundle_for_config",
            new_callable=AsyncMock,
        ):
            try:
                await scrape_endpoint(
                    ScrapeRequest(url="https://example.com/x", query="q")
                )
            except HTTPException as raised:
                return raised
        self.fail("expected HTTPException")

    async def test_invalid_url_maps_to_400(self) -> None:
        raised = await self._run_with_exc(InvalidUrlError("bad"))
        self.assertEqual(raised.status_code, 400)
        self.assertEqual(raised.detail["code"], "invalid_url")

    async def test_blocked_url_maps_to_403(self) -> None:
        raised = await self._run_with_exc(BlockedUrlError("nope"))
        self.assertEqual(raised.status_code, 403)
        self.assertEqual(raised.detail["code"], "blocked_url")

    async def test_fetch_timeout_maps_to_504(self) -> None:
        raised = await self._run_with_exc(FetchTimeoutError("slow"))
        self.assertEqual(raised.status_code, 504)
        self.assertEqual(raised.detail["code"], "fetch_timeout")

    async def test_fetch_failed_maps_to_502(self) -> None:
        raised = await self._run_with_exc(FetchFailedError("dead"))
        self.assertEqual(raised.status_code, 502)
        self.assertEqual(raised.detail["code"], "fetch_failed")

    async def test_unsupported_document_maps_to_415(self) -> None:
        raised = await self._run_with_exc(UnsupportedDocumentError(".doc"))
        self.assertEqual(raised.status_code, 415)
        self.assertEqual(raised.detail["code"], "unsupported_document")

    async def test_empty_content_maps_to_422(self) -> None:
        raised = await self._run_with_exc(EmptyContentError("nothing"))
        self.assertEqual(raised.status_code, 422)
        self.assertEqual(raised.detail["code"], "empty_content")


if __name__ == "__main__":
    unittest.main()
