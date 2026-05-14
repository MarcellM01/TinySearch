# TinySearch

TinySearch is a tiny, local-first research engine for agents. It searches the
web, reranks results with dense embeddings plus BM25, crawls the best pages,
pulls out the most relevant chunks, and hands your LLM a source-grounded prompt
instead of a mystery-meat answer.

It is built for MCP, simple enough to run locally, and boring in the best way:
no hosted dashboard, no account system, no analytics, no scraped data cache.

## Why It Exists

- Give agents a compact web research tool they can call over MCP.
- Keep source URLs attached to every factual claim your LLM should make.
- Avoid dumping entire pages into context when a few ranked chunks will do.
- Let you swap between the default local embedding model and an OpenAI-compatible embedding API.

## What It Does

- Searches DuckDuckGo's HTML endpoint.
- Ranks search results with weighted reciprocal rank fusion over dense vectors and BM25.
- Crawls kept pages with Crawl4AI.
- Chunks and reranks page text globally, not one page at a time.
- Returns a `SEARCH-GROUNDED ANSWER PROMPT` for the caller's model to answer from.

## Entrypoints

- `pipelines.agentic_research.agentic_run`: single-turn search, crawl, ranking, and prompt assembly.
- `servers.mcp_server`: MCP server for agent clients.
- `servers.fastapi_server`: optional HTTP API.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The default embedding backend downloads `all-MiniLM-L6-v2` through
`sentence-transformers` on first use and caches it outside the repo. Set
`TINYSEARCH_HF_CACHE` if you want to choose the cache location.

For faster cold starts (especially in MCP subprocesses), ship the ONNX bundle under
`models/all-minilm-l6-v2-onnx/` (see `models/all-minilm-l6-v2-onnx/README.md` and
`scripts/export_embedding_onnx.py`). When `model.onnx` and tokenizer files are
present, TinySearch uses `onnxruntime` instead of loading the PyTorch stack. Override
the directory with `TINYSEARCH_ONNX_MODEL_DIR` if needed.

## MCP Setup

Add TinySearch to your MCP client config. Use absolute paths.

macOS / Linux:

```json
{
  "mcpServers": {
    "tinysearch": {
      "command": "/absolute/path/to/TinySearch/.venv/bin/python",
      "args": [
        "/absolute/path/to/TinySearch/servers/mcp_server.py"
      ]
    }
  }
}
```

Windows:

```json
{
  "mcpServers": {
    "tinysearch": {
      "command": "C:/absolute/path/to/TinySearch/.venv/Scripts/python.exe",
      "args": [
        "C:/absolute/path/to/TinySearch/servers/mcp_server.py"
      ]
    }
  }
}
```

The MCP server exposes one tool:

```text
research(query)
```

Pass the user's question as-is. TinySearch does the search and returns a prompt
in `answer`; your client model should use that prompt to produce the final,
cited response.

Template config files live in `mcp_templates/`.

The repo also includes [`agentic_coding_templates/global-rules-recommended.md`](agentic_coding_templates/global-rules-recommended.md), a global-rules template we **strongly recommend** if you wire TinySearch into any agentic coding tool (Cline, Roo Code, and similar). With those rules in place, it works like a charm.

The server uses **stdio** by default (what Cursor and similar clients expect when
they spawn `python .../mcp_server.py`). To run with `sse` or `streamable-http`
instead, set environment variable `MCP_TRANSPORT` when starting the process; do
not put transport in `configs/research_config.json`.

## Optional HTTP Server

```bash
uvicorn servers.fastapi_server:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /web_search?query=...`
- `POST /site_crawl`
- `POST /research`

## Research Flow

```mermaid
flowchart TD
    A[User query] --> B[DuckDuckGo search]
    B --> C[Build result docs<br/>title URL domain snippet]
    C --> D[Dense + BM25 weighted RRF]
    D --> E[Kept search results]
    E --> F[Parallel crawl4ai markdown crawl]
    F --> G[Chunk all kept pages]
    G --> H[Global dense + BM25 weighted RRF<br/>shared BM25 corpus]
    H --> I[Final RESULTS prompt]
```

The returned `answer` is not a synthesized answer. It is a prompt containing
ranked source blocks:

```text
========================================================================================
SEARCH-GROUNDED ANSWER PROMPT
========================================================================================
QUESTION
========================================================================================
...
========================================================================================

TODAY
========================================================================================
2026-05-12
========================================================================================

CRITICAL INSTRUCTIONS
========================================================================================
You are answering a question using search results.
Use only the text under RESULTS.
If the answer is not directly supported, say the results are not enough.
Use TODAY to understand relative dates like today, yesterday, this year, or last month.
If the RESULTS text contains dates, use those dates when they matter.
Cite the source URL after each factual claim.
========================================================================================

========================================================================================
RESULTS
========================================================================================
========================================================================================
RESULT 1
========================================================================================
TITLE 1
======
...
======
URL 1
======
...
======
SEARCH PREVIEW 1
======
...
======
RELEVANT TEXT 1
======
----- RELEVANT CHUNK 1 -----
...
======

========================================================================================
QUESTION
========================================================================================
...
========================================================================================

TODAY
========================================================================================
2026-05-12
========================================================================================

SEARCH-GROUNDED ANSWER PROMPT
========================================================================================
```

## Configuration

Tune research defaults in `configs/research_config.json`:

- Search: `search_top_k`, `search_rrf_cutoff`, `search_dense_weight`, `search_max_results_to_keep`
- Chunks: `chunk_rrf_cutoff`, `chunk_dense_weight`, `chunk_max_results_to_keep` (default `2`, global across the chunk pool)
- Crawl: `crawl_max_chunk_tokens` (default `300`), `crawl_overlap_tokens`, `max_concurrent_crawls`
- Embeddings: `embedding_backend` (`default` = ONNX bundle in `models/all-minilm-l6-v2-onnx/` when present, otherwise fixed local `all-MiniLM-L6-v2` via `sentence-transformers` cached outside the repo, or `openai_compatible`), `embedding_openai_env_file` (path to `.env` for API URL, key, and model when using `openai_compatible`), `max_concurrent_embedding_calls`; optional `TINYSEARCH_ONNX_MODEL_DIR` for a custom ONNX bundle path
- Dense input prefixes: `dense_query_prefix`, `dense_document_prefix`
- Trace: `trace_path`

For `embedding_backend` `openai_compatible`, add a `.env` file at the project root (or set `embedding_openai_env_file`) with `OPENAI_BASE_URL` (optional for api.openai.com), `OPENAI_API_KEY`, and `OPENAI_EMBEDDING_MODEL` (aliases: `EMBEDDING_MODEL`, `MODEL_NAME`).

The research pipeline requires dense embeddings. It raises if
`search_dense_weight` or `chunk_dense_weight` is set to `0`.

Edit `dense_query_prefix` and `dense_document_prefix` if a different embedding
model expects raw text or a different instruction format.

## Tests

Run the unittest suite:

```bash
python -m unittest discover tests
```

## Privacy Notes

TinySearch reads the pages it crawls and returns ranked excerpts to the calling
client. It does not include credentials in the repo, and `.env` / trace output
should stay local. If you enable `openai_compatible` embeddings, your embedding
provider receives the text snippets sent for vectorization.
