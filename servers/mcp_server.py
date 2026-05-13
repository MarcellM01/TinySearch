from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server.fastmcp import Context, FastMCP

from pipelines.agentic_research import agentic_run
from services.research_config_service import config_trace_path, load_research_config, research_run_kwargs
from services.token_counter_service import token_count


MCP_INSTRUCTIONS = """
This MCP server exposes one high-level web research tool:

1. research(query)

Pass the user's question as-is in query. Do not rewrite, correct spelling,
expand abbreviations, add dates, add missing context, simplify, translate, or
otherwise improve the user's wording before calling the tool.

The tool searches DuckDuckGo, ranks search results with dense embeddings and
BM25 using reciprocal rank fusion, crawls kept pages, ranks page chunks, and
returns a prompt in the answer field. The caller's LLM should answer from that
prompt and cite source URLs from the result blocks.
""".strip()


def _research_settings() -> dict[str, Any]:
    return research_run_kwargs(load_research_config())


def _answer_tokens(answer: str) -> int:
    encoding_name = str(load_research_config()["encoding_name"])
    return token_count(answer, encoding_name=encoding_name)


def _validate_query(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    return query


def _log(message: str) -> None:
    print(f"[tinysearch] {message}", file=sys.stderr, flush=True)


async def _emit_mcp_progress(ctx: Context, event: str, payload: dict[str, Any]) -> None:
    messages = {
        "start": "Starting research",
        "search_start": "Searching the web",
        "search_results": f"Found {payload.get('results_count', 0)} search result(s)",
        "search_ranked": f"Kept {payload.get('kept_results', 0)} page(s) to crawl",
        "crawl_start": f"Reading {payload.get('url')}",
        "crawl_done": f"Crawled page ({payload.get('chunks', 0)} text chunk(s))",
        "crawl_error": (
            f"Could not read {payload.get('url')}"
            + (
                f" ({str(payload.get('error') or '')[:120]})"
                if payload.get("error")
                else ""
            )
        ),
        "pages_indexed": (
            f"Indexed {payload.get('chunks_extracted', 0)} chunk(s) from "
            f"{payload.get('urls_read', 0)} URL(s); "
            f"{payload.get('chunks_in_prompt', 0)} in prompt"
            + (
                f"; {payload.get('crawl_errors_count', 0)} crawl error(s)"
                if payload.get("crawl_errors_count")
                else ""
            )
        ),
        "done": "Research prompt ready",
    }
    progress_by_event = {
        "start": 1,
        "search_start": 2,
        "search_results": 3,
        "search_ranked": 4,
        "pages_indexed": 8,
        "done": 10,
    }
    notify_events = {"search_ranked", "pages_indexed", "crawl_error", "done"}
    message = messages.get(event, event.replace("_", " "))
    if event in notify_events:
        await ctx.info(message)
    if event in progress_by_event:
        await ctx.report_progress(progress_by_event[event], total=10, message=message)


mcp = FastMCP(
    "tinysearch",
    instructions=MCP_INSTRUCTIONS,
)


@mcp.tool(
    name="research",
    title="Research",
    description=(
        "Search the web, crawl ranked pages, and return a grounded answer prompt. "
        "Input schema has exactly one field: query. Pass the user's question as-is."
    ),
)
async def research(query: str, ctx: Context) -> dict[str, Any]:
    query = _validate_query(query)
    started = time.monotonic()
    _log(f"research called query={query!r}")
    await ctx.report_progress(0, total=10, message="Research request received")
    try:
        result = await agentic_run(
            query,
            trace_path=config_trace_path(),
            progress_callback=lambda event, payload: _emit_mcp_progress(ctx, event, payload),
            **_research_settings(),
        )
        elapsed = time.monotonic() - started
        _log(
            "research returning "
            f"answer_tokens={_answer_tokens(result.answer)} "
            f"elapsed={elapsed:.2f}s"
        )
        await ctx.report_progress(10, total=10, message="Research prompt ready")
        _log(f"research returning answer_tokens={_answer_tokens(result.answer)}")
        return {"answer": result.answer}
    except Exception as exc:
        elapsed = time.monotonic() - started
        _log(f"research failed elapsed={elapsed:.2f}s error={exc!r}")
        await ctx.error(f"research failed: {exc!r}")
        raise


if __name__ == "__main__":
    transport = str(load_research_config()["mcp_transport"])
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise ValueError("mcp_transport must be one of: stdio, sse, streamable-http")
    mcp.run(transport=transport)
