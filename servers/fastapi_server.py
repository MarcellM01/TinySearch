from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from pipelines.agentic_research import agentic_run
from services.embedding_service import normalize_embedding_backend
from services.research_config_service import (
    config_trace_path,
    load_research_config,
    normalize_research_query,
    research_run_kwargs,
    research_tokenizer_name,
)
from services.scrape_service import (
    DEFAULT_SCRAPE_MAX_TOKENS,
    SCRAPE_ERROR_MAP,
    ScrapeError,
    scrape_url,
)
from services.site_crawl_service import crawl_search
from services.url_safety_service import BlockedUrlError, InvalidUrlError
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


def _tinysearch_version() -> str:
    return os.environ.get("TINYSEARCH_VERSION", "dev").strip() or "dev"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    cfg = load_research_config()
    await _ensure_local_bundle_for_config(cfg)
    yield


app = FastAPI(
    title="TinySearch API",
    description="Web search, site crawl, and hybrid research endpoints.",
    version=_tinysearch_version(),
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


class ScrapeRequest(BaseModel):
    url: HttpUrl
    query: str = Field(..., min_length=1)
    max_tokens: int = Field(default=DEFAULT_SCRAPE_MAX_TOKENS, ge=1, le=200_000)
    include_metadata: bool = True


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
    pipeline_timeout_seconds: float | None = Field(default=None, gt=0)
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


def _raise_scrape_http_error(exc: Exception) -> None:
    mapping = SCRAPE_ERROR_MAP.get(type(exc))
    if mapping is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "internal_error", "message": "internal error"},
        ) from exc
    code, status_code = mapping
    raise HTTPException(
        status_code=status_code,
        detail={"code": code, "message": str(exc)},
    ) from exc


@app.post("/scrape")
async def scrape_endpoint(request: ScrapeRequest) -> dict[str, Any]:
    config = load_research_config()
    await _ensure_local_bundle_for_config(config)
    tokenizer = research_tokenizer_name(config)
    try:
        result = await scrape_url(
            str(request.url),
            request.query,
            max_tokens=request.max_tokens,
            include_metadata=request.include_metadata,
            config=config,
            tokenizer_name=tokenizer,
        )
    except (InvalidUrlError, BlockedUrlError, ScrapeError) as exc:
        _raise_scrape_http_error(exc)
    return result.to_response(include_metadata=request.include_metadata)


@app.get("/scrape")
async def scrape_get(
    url: HttpUrl,
    query: str,
    max_tokens: int = DEFAULT_SCRAPE_MAX_TOKENS,
    include_metadata: bool = True,
) -> dict[str, Any]:
    return await scrape_endpoint(
        ScrapeRequest(
            url=url,
            query=query,
            max_tokens=max_tokens,
            include_metadata=include_metadata,
        )
    )


@app.post("/research")
async def research_endpoint(request: ResearchRequest) -> dict[str, Any]:
    config = load_research_config()
    query = normalize_research_query(request.query)
    overrides = request.model_dump(exclude_none=True)
    overrides.pop("query")
    trace_path = overrides.pop("trace_path", None)

    run_kwargs = research_run_kwargs(config)
    run_kwargs.update(overrides)
    embedding_model = str(run_kwargs["embedding_model"])
    if normalize_embedding_backend(str(config["embedding_backend"])) == "onnx":
        from services.onnx_bundle_service import ensure_onnx_bundle_sync

        await asyncio.to_thread(ensure_onnx_bundle_sync, embedding_model)
    result = await agentic_run(
        query,
        **run_kwargs,
        trace_path=Path(trace_path) if trace_path else config_trace_path(config),
    )
    return {"answer": result.answer}


@app.get("/research")
async def research_get(query: str) -> dict[str, Any]:
    return await research_endpoint(ResearchRequest(query=query))
