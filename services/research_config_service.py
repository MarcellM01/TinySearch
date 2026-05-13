from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.embedding_service import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_OPENAI_ENV_FILE,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESEARCH_CONFIG_PATH = PROJECT_ROOT / "configs" / "research_config.json"

DEFAULT_RESEARCH_CONFIG: dict[str, Any] = {
    "search_top_k": 10,
    "search_rrf_cutoff": 0.0,
    "search_dense_weight": 0.5,
    "search_max_results_to_keep": 5,
    "chunk_rrf_cutoff": 0.0,
    "chunk_dense_weight": 0.5,
    "chunk_max_results_to_keep": 2,
    "chunk_rank_oversample": 3,
    "chunk_dedupe_jaccard_threshold": 0.92,
    "chunk_max_per_source_url": 4,
    "max_concurrent_crawls": 5,
    "max_concurrent_embedding_calls": 3,
    "embedding_timeout_seconds": 60.0,
    "embedding_timeout_retries": 2,
    "crawl_fit_markdown_mode": "bm25",
    "crawl_fit_min_chars": 200,
    "crawl_bm25_threshold": 1.5,
    "crawl_bm25_language": "english",
    "crawl_pruning_threshold": 0.48,
    "crawl_max_chunk_tokens": 300,
    "crawl_overlap_tokens": 80,
    "crawl_max_page_tokens": 0,
    "encoding_name": "o200k_base",
    "embedding_backend": DEFAULT_EMBEDDING_BACKEND,
    "embedding_openai_env_file": DEFAULT_EMBEDDING_OPENAI_ENV_FILE,
    "dense_query_prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
    "dense_document_prefix": "",
    "dense_document_embed_batch_size": 32,
    "trace_path": "trace_logs/agentic_trace.json",
}

_INT_FIELDS = {
    "search_top_k",
    "search_max_results_to_keep",
    "chunk_max_results_to_keep",
    "chunk_rank_oversample",
    "chunk_max_per_source_url",
    "max_concurrent_crawls",
    "max_concurrent_embedding_calls",
    "embedding_timeout_retries",
    "crawl_fit_min_chars",
    "crawl_max_chunk_tokens",
    "crawl_overlap_tokens",
    "crawl_max_page_tokens",
    "dense_document_embed_batch_size",
}
_FLOAT_FIELDS = {
    "search_rrf_cutoff",
    "search_dense_weight",
    "chunk_rrf_cutoff",
    "chunk_dense_weight",
    "chunk_dedupe_jaccard_threshold",
    "crawl_bm25_threshold",
    "crawl_pruning_threshold",
    "embedding_timeout_seconds",
}


def _coerce_config(raw: dict[str, Any]) -> dict[str, Any]:
    config = dict(DEFAULT_RESEARCH_CONFIG)
    config.update(raw)
    for legacy in ("embedding_model", "embedding_gguf_file", "mcp_transport"):
        config.pop(legacy, None)
    for key in _INT_FIELDS:
        config[key] = int(config[key])
    for key in _FLOAT_FIELDS:
        config[key] = float(config[key])
    for key in (
        "encoding_name",
        "embedding_backend",
        "embedding_openai_env_file",
        "dense_query_prefix",
        "dense_document_prefix",
        "crawl_fit_markdown_mode",
        "crawl_bm25_language",
        "trace_path",
    ):
        if config.get(key) is not None:
            config[key] = str(config[key])
    return config


def load_research_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_RESEARCH_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        return dict(DEFAULT_RESEARCH_CONFIG)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"research config must be a JSON object: {config_path}")
    return _coerce_config(raw)


def research_run_kwargs(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_research_config() if config is None else config
    keys = (
        "search_top_k",
        "search_rrf_cutoff",
        "search_dense_weight",
        "search_max_results_to_keep",
        "chunk_rrf_cutoff",
        "chunk_dense_weight",
        "chunk_max_results_to_keep",
        "chunk_rank_oversample",
        "chunk_dedupe_jaccard_threshold",
        "chunk_max_per_source_url",
        "max_concurrent_crawls",
        "max_concurrent_embedding_calls",
        "embedding_timeout_seconds",
        "embedding_timeout_retries",
        "crawl_fit_markdown_mode",
        "crawl_fit_min_chars",
        "crawl_bm25_threshold",
        "crawl_bm25_language",
        "crawl_pruning_threshold",
        "crawl_max_chunk_tokens",
        "crawl_overlap_tokens",
        "crawl_max_page_tokens",
        "encoding_name",
        "embedding_backend",
        "embedding_openai_env_file",
        "dense_query_prefix",
        "dense_document_prefix",
        "dense_document_embed_batch_size",
    )
    return {key: config[key] for key in keys}


def config_trace_path(config: dict[str, Any] | None = None) -> Path | None:
    config = load_research_config() if config is None else config
    value = str(config.get("trace_path") or "").strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
