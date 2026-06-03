from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from pydantic import BaseModel, Field, HttpUrl

from pipelines.agentic_research import agentic_run
from services.embedding_service import normalize_embedding_backend
from services.research_config_service import (
    config_trace_path,
    load_research_config,
    research_tokenizer_name,
)
from services.site_crawl_service import crawl_search
from services.web_search_service import (
    filter_blocked_search_results,
    search,
    search_to_markdown,
)


async def _ensure_local_bundle_for_config(config: dict[str, Any]) -> None:
    if normalize_embedding_backend(str(config["embedding_backend"])) != "onnx":
        return
    from services.onnx_bundle_service import ensure_onnx_bundle_sync

    await asyncio.to_thread(ensure_onnx_bundle_sync, str(config["embedding_model"]))


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    cfg = load_research_config()
    await _ensure_local_bundle_for_config(cfg)
    yield


app = FastAPI(
    title="TinySearch API",
    description="Web search, site crawl, and hybrid research endpoints.",
    version="0.1.4",
    lifespan=_lifespan,
)


class WebSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=50)
    include_markdown: bool = True


class SiteCrawlRequest(BaseModel):
    url: HttpUrl
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    max_chunk_tokens: int = Field(default=500, ge=100, le=4000)
    overlap_tokens: int = Field(default=80, ge=0, le=1000)
    max_return_tokens: int | None = Field(default=None, ge=1)
    crawl4ai_bm25_threshold: float = Field(default=1.5, ge=0)
    crawl4ai_language: str = "english"
    encoding_name: str | None = None


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    search_top_k: int | None = Field(default=None, ge=1, le=50)
    search_rrf_cutoff: float | None = Field(default=None, ge=0.0, le=1.0)
    search_dense_weight: float | None = Field(default=None, gt=0.0, le=1.0)
    search_max_results_to_keep: int | None = Field(default=None, ge=1, le=50)
    chunk_rrf_cutoff: float | None = Field(default=None, ge=0.0, le=1.0)
    chunk_dense_weight: float | None = Field(default=None, gt=0.0, le=1.0)
    chunk_max_results_to_keep: int | None = Field(default=None, ge=1, le=50)
    chunk_rank_oversample: int | None = Field(default=None, ge=1, le=50)
    chunk_dedupe_jaccard_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    chunk_max_per_source_url: int | None = Field(default=None, ge=0, le=500)
    max_concurrent_crawls: int | None = Field(default=None, ge=1, le=20)
    max_concurrent_embedding_calls: int | None = Field(default=None, ge=1, le=20)
    embedding_timeout_seconds: float | None = Field(default=None, gt=0)
    embedding_timeout_retries: int | None = Field(default=None, ge=0, le=10)
    crawl_fit_markdown_mode: str | None = None
    crawl_fit_min_chars: int | None = Field(default=None, ge=0, le=500_000)
    crawl_bm25_threshold: float | None = Field(default=None, ge=0.0)
    crawl_bm25_language: str | None = None
    crawl_pruning_threshold: float | None = Field(default=None, ge=0.0)
    crawl_max_chunk_tokens: int | None = Field(default=None, ge=100, le=4000)
    crawl_overlap_tokens: int | None = Field(default=None, ge=0, le=1000)
    crawl_max_page_tokens: int | None = Field(default=None, ge=0, le=500_000)
    dense_query_prefix: str | None = None
    dense_document_prefix: str | None = None
    dense_document_embed_batch_size: int | None = Field(default=None, ge=1, le=512)
    encoding_name: str | None = None
    embedding_model: str | None = None
    trace_path: str | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/web_search")
async def web_search_endpoint(request: WebSearchRequest) -> dict[str, Any]:
    config = load_research_config()
    results = filter_blocked_search_results(
        search(request.query, limit=request.limit),
        config["blocked_domains"],
    )
    payload: dict[str, Any] = {
        "query": request.query,
        "results": [result.__dict__ for result in results],
    }
    if request.include_markdown:
        payload["markdown"] = search_to_markdown(results)
    return payload


@app.get("/web_search")
async def web_search_get(
    query: str,
    limit: int = 10,
    include_markdown: bool = True,
) -> dict[str, Any]:
    return await web_search_endpoint(
        WebSearchRequest(
            query=query,
            limit=limit,
            include_markdown=include_markdown,
        )
    )


@app.post("/site_crawl")
async def site_crawl_endpoint(request: SiteCrawlRequest) -> dict[str, Any]:
    config = load_research_config()
    return await crawl_search(
        url=str(request.url),
        user_query=request.query,
        top_k=request.top_k,
        max_chunk_tokens=request.max_chunk_tokens,
        overlap_tokens=request.overlap_tokens,
        max_return_tokens=request.max_return_tokens,
        encoding_name=request.encoding_name or research_tokenizer_name(config),
        crawl4ai_bm25_threshold=request.crawl4ai_bm25_threshold,
        crawl4ai_language=request.crawl4ai_language,
    )


@app.get("/site_crawl")
async def site_crawl_get(
    url: HttpUrl,
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    return await site_crawl_endpoint(
        SiteCrawlRequest(
            url=url,
            query=query,
            top_k=top_k,
        )
    )


@app.post("/research")
async def research_endpoint(request: ResearchRequest) -> dict[str, Any]:
    config = load_research_config()
    embedding_model = request.embedding_model or str(config["embedding_model"])
    if normalize_embedding_backend(str(config["embedding_backend"])) == "onnx":
        from services.onnx_bundle_service import ensure_onnx_bundle_sync

        await asyncio.to_thread(ensure_onnx_bundle_sync, embedding_model)
    result = await agentic_run(
        request.query,
        search_top_k=request.search_top_k or int(config["search_top_k"]),
        search_rrf_cutoff=request.search_rrf_cutoff
        if request.search_rrf_cutoff is not None
        else float(config["search_rrf_cutoff"]),
        search_dense_weight=request.search_dense_weight
        if request.search_dense_weight is not None
        else float(config["search_dense_weight"]),
        search_max_results_to_keep=request.search_max_results_to_keep
        or int(config["search_max_results_to_keep"]),
        chunk_rrf_cutoff=request.chunk_rrf_cutoff
        if request.chunk_rrf_cutoff is not None
        else float(config["chunk_rrf_cutoff"]),
        chunk_dense_weight=request.chunk_dense_weight
        if request.chunk_dense_weight is not None
        else float(config["chunk_dense_weight"]),
        chunk_max_results_to_keep=request.chunk_max_results_to_keep
        or int(config["chunk_max_results_to_keep"]),
        chunk_rank_oversample=request.chunk_rank_oversample
        or int(config["chunk_rank_oversample"]),
        chunk_dedupe_jaccard_threshold=request.chunk_dedupe_jaccard_threshold
        if request.chunk_dedupe_jaccard_threshold is not None
        else float(config["chunk_dedupe_jaccard_threshold"]),
        chunk_max_per_source_url=request.chunk_max_per_source_url
        if request.chunk_max_per_source_url is not None
        else int(config["chunk_max_per_source_url"]),
        max_concurrent_crawls=request.max_concurrent_crawls
        or int(config["max_concurrent_crawls"]),
        max_concurrent_embedding_calls=request.max_concurrent_embedding_calls
        or int(config["max_concurrent_embedding_calls"]),
        embedding_timeout_seconds=request.embedding_timeout_seconds
        if request.embedding_timeout_seconds is not None
        else float(config["embedding_timeout_seconds"]),
        embedding_timeout_retries=request.embedding_timeout_retries
        if request.embedding_timeout_retries is not None
        else int(config["embedding_timeout_retries"]),
        crawl_max_chunk_tokens=request.crawl_max_chunk_tokens
        or int(config["crawl_max_chunk_tokens"]),
        crawl_overlap_tokens=request.crawl_overlap_tokens
        if request.crawl_overlap_tokens is not None
        else int(config["crawl_overlap_tokens"]),
        crawl_max_page_tokens=request.crawl_max_page_tokens
        if request.crawl_max_page_tokens is not None
        else int(config["crawl_max_page_tokens"]),
        crawl_fit_markdown_mode=(
            request.crawl_fit_markdown_mode
            if request.crawl_fit_markdown_mode is not None
            else str(config["crawl_fit_markdown_mode"])
        ),
        crawl_fit_min_chars=(
            request.crawl_fit_min_chars
            if request.crawl_fit_min_chars is not None
            else int(config["crawl_fit_min_chars"])
        ),
        crawl_bm25_threshold=request.crawl_bm25_threshold
        if request.crawl_bm25_threshold is not None
        else float(config["crawl_bm25_threshold"]),
        crawl_bm25_language=(
            request.crawl_bm25_language
            if request.crawl_bm25_language is not None
            else str(config["crawl_bm25_language"])
        ),
        crawl_pruning_threshold=(
            request.crawl_pruning_threshold
            if request.crawl_pruning_threshold is not None
            else float(config["crawl_pruning_threshold"])
        ),
        embedding_backend=str(config["embedding_backend"]),
        embedding_model=embedding_model,
        embedding_openai_env_file=str(config["embedding_openai_env_file"]),
        dense_query_prefix=request.dense_query_prefix
        if request.dense_query_prefix is not None
        else str(config["dense_query_prefix"]),
        dense_document_prefix=request.dense_document_prefix
        if request.dense_document_prefix is not None
        else str(config["dense_document_prefix"]),
        dense_document_embed_batch_size=request.dense_document_embed_batch_size
        if request.dense_document_embed_batch_size is not None
        else int(config["dense_document_embed_batch_size"]),
        blocked_domains=config["blocked_domains"],
        encoding_name=request.encoding_name or str(config["encoding_name"]),
        trace_path=Path(request.trace_path) if request.trace_path else config_trace_path(config),
    )
    return {"answer": result.answer}


@app.get("/research")
async def research_get(query: str) -> dict[str, Any]:
    return await research_endpoint(ResearchRequest(query=query))
