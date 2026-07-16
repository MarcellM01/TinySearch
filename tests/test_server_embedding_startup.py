from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from servers.fastapi_server import (
    ResearchRequest,
    _ensure_local_bundle_for_config as ensure_fastapi_bundle,
    _tinysearch_version,
    research_endpoint,
)
from servers.mcp_server import _ensure_local_bundle_for_config as ensure_mcp_bundle
from servers.mcp_server import _mcp_cors_origins
from services.research_config_service import (
    DEFAULT_RESEARCH_CONFIG,
    config_trace_path,
    normalize_research_query,
    research_run_kwargs,
)


class ServerEmbeddingStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_fastapi_startup_ensures_selected_local_embedding_model(self) -> None:
        cfg = {"embedding_backend": "onnx", "embedding_model": "balanced"}

        with patch("services.onnx_bundle_service.ensure_onnx_bundle_sync") as ensure:
            await ensure_fastapi_bundle(cfg)

        ensure.assert_called_once_with("balanced")

    async def test_fastapi_startup_skips_openai_compatible_backend(self) -> None:
        cfg = {"embedding_backend": "openai_compatible", "embedding_model": "balanced"}

        with patch("services.onnx_bundle_service.ensure_onnx_bundle_sync") as ensure:
            await ensure_fastapi_bundle(cfg)

        ensure.assert_not_called()


class FastApiResearchParityTests(unittest.IsolatedAsyncioTestCase):
    async def test_research_uses_same_config_defaults_as_mcp(self) -> None:
        config = dict(DEFAULT_RESEARCH_CONFIG)
        config["embedding_backend"] = "openai_compatible"
        run = AsyncMock(return_value=SimpleNamespace(answer="grounded prompt"))

        with patch(
            "servers.fastapi_server.load_research_config", return_value=config
        ), patch("servers.fastapi_server.agentic_run", new=run):
            response = await research_endpoint(ResearchRequest(query="  test query  "))

        self.assertEqual(response, {"answer": "grounded prompt"})
        run.assert_awaited_once_with(
            "test query",
            **research_run_kwargs(config),
            trace_path=config_trace_path(config),
        )

    async def test_research_rejects_whitespace_only_query(self) -> None:
        with patch(
            "servers.fastapi_server.load_research_config",
            return_value=dict(DEFAULT_RESEARCH_CONFIG),
        ):
            with self.assertRaisesRegex(ValueError, "query must not be empty"):
                await research_endpoint(ResearchRequest(query="   "))


class ServerRuntimeMetadataTests(unittest.TestCase):
    def test_version_comes_from_environment(self) -> None:
        with patch.dict("os.environ", {"TINYSEARCH_VERSION": "v0.2.0"}):
            self.assertEqual(_tinysearch_version(), "v0.2.0")

    def test_version_defaults_to_dev(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_tinysearch_version(), "dev")

    def test_query_normalization_is_shared(self) -> None:
        self.assertEqual(normalize_research_query("  hello  "), "hello")
        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            normalize_research_query("  ")


class McpEmbeddingStartupTests(unittest.TestCase):
    def test_mcp_startup_ensures_selected_local_embedding_model(self) -> None:
        cfg = {"embedding_backend": "onnx", "embedding_model": "quality"}

        with patch("services.onnx_bundle_service.ensure_onnx_bundle_sync") as ensure:
            ensure_mcp_bundle(cfg)

        ensure.assert_called_once_with("quality")

    def test_mcp_startup_skips_openai_compatible_backend(self) -> None:
        cfg = {"embedding_backend": "openai_compatible", "embedding_model": "quality"}

        with patch("services.onnx_bundle_service.ensure_onnx_bundle_sync") as ensure:
            ensure_mcp_bundle(cfg)

        ensure.assert_not_called()


class McpCorsConfigTests(unittest.TestCase):
    def test_cors_origins_default_to_wildcard(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_mcp_cors_origins(), ["*"])

    def test_cors_origins_parse_comma_separated_list(self) -> None:
        with patch.dict(
            "os.environ",
            {"MCP_CORS_ORIGINS": "http://localhost:8080, http://172.20.210.53:8080"},
        ):
            self.assertEqual(
                _mcp_cors_origins(),
                ["http://localhost:8080", "http://172.20.210.53:8080"],
            )


if __name__ == "__main__":
    unittest.main()
