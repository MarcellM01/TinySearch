from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from servers.fastapi_server import ScrapeRequest, scrape_endpoint
from servers.mcp_server import scrape_url_tool
from services.scrape_service import ScrapeResult


def _fn(coro):
    return getattr(coro, "fn", coro)


def _shared_result() -> ScrapeResult:
    return ScrapeResult(
        answer="URL-GROUNDED ANSWER PROMPT SHARED",
        url="https://example.com/x",
        title="Title",
        query="q",
        content_tokens=10,
        answer_tokens=20,
        truncated=False,
        retrieved_at="2026-06-12T10:30:00Z",
        metadata={"description": "d", "author": None, "published_date": None},
    )


class ScrapeFastApiMcpParityTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_adapters_return_identical_answer(self) -> None:
        fastapi_mock = AsyncMock(return_value=_shared_result())
        mcp_mock = AsyncMock(return_value=_shared_result())

        with patch("servers.fastapi_server.scrape_url", fastapi_mock), patch(
            "servers.fastapi_server._ensure_local_bundle_for_config",
            new_callable=AsyncMock,
        ):
            fastapi_payload = await scrape_endpoint(
                ScrapeRequest(url="https://example.com/x", query="q")
            )

        with patch("servers.mcp_server.scrape_url", mcp_mock), patch(
            "servers.mcp_server._ensure_local_bundle_for_config"
        ):
            mcp_payload = await _fn(scrape_url_tool)("https://example.com/x", "q")

        self.assertEqual(fastapi_payload["answer"], mcp_payload["answer"])
        self.assertEqual(fastapi_payload["url"], mcp_payload["url"])
        self.assertEqual(fastapi_payload["title"], mcp_payload["title"])
        self.assertEqual(
            fastapi_payload["content_tokens"], mcp_payload["content_tokens"]
        )
        self.assertEqual(
            fastapi_payload["answer_tokens"], mcp_payload["answer_tokens"]
        )
        self.assertEqual(fastapi_payload["truncated"], mcp_payload["truncated"])
        self.assertEqual(
            fastapi_payload["retrieved_at"], mcp_payload["retrieved_at"]
        )

    async def test_both_adapters_pass_default_mcp_args_to_service(self) -> None:
        fastapi_mock = AsyncMock(return_value=_shared_result())
        mcp_mock = AsyncMock(return_value=_shared_result())

        with patch("servers.fastapi_server.scrape_url", fastapi_mock), patch(
            "servers.fastapi_server._ensure_local_bundle_for_config",
            new_callable=AsyncMock,
        ):
            await scrape_endpoint(
                ScrapeRequest(url="https://example.com/x", query="q")
            )

        with patch("servers.mcp_server.scrape_url", mcp_mock), patch(
            "servers.mcp_server._ensure_local_bundle_for_config"
        ):
            await _fn(scrape_url_tool)("https://example.com/x", "q")

        fastapi_kwargs = fastapi_mock.await_args.kwargs
        mcp_kwargs = mcp_mock.await_args.kwargs
        self.assertEqual(fastapi_kwargs["max_tokens"], mcp_kwargs["max_tokens"])
        self.assertEqual(
            fastapi_kwargs["include_metadata"], mcp_kwargs["include_metadata"]
        )
        self.assertEqual(
            fastapi_kwargs["tokenizer_name"], mcp_kwargs["tokenizer_name"]
        )


if __name__ == "__main__":
    unittest.main()
