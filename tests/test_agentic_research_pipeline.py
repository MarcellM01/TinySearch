from __future__ import annotations

import unittest

from pipelines.agentic_research import agentic_run
from services.web_search_service import SearchResult


def _fake_search(query: str, limit: int) -> list[SearchResult]:
    return [
        SearchResult(
            result_id=1,
            title="Python Async Search",
            url="https://example.com/python",
            text="Python asyncio search guide.",
        ),
        SearchResult(
            result_id=2,
            title="Bread Recipes",
            url="https://example.com/bread",
            text="Bread recipes use flour and yeast.",
        ),
    ][:limit]


async def _fake_crawl(**kwargs):
    url = kwargs["url"]
    pages = {
        "https://example.com/python": (
            "# Intro\n\n"
            "Python asyncio search uses async tasks.\n\n"
            "Bread recipes use flour and yeast."
        ),
        "https://example.com/bread": (
            "# Cooking\n\n"
            "Bread recipes use flour and yeast."
        ),
    }
    return {"url": url, "markdown": pages[url], "markdown_raw": pages[url], "html": "", "tokens_raw": 10}


async def _fake_embedder(inputs: list[str]) -> list[list[float]]:
    vectors = []
    for text in inputs:
        lowered = text.removeprefix("task: search result | query: ").removeprefix(
            "title: none | text: "
        ).lower()
        if "python" in lowered or "async" in lowered:
            vectors.append([1.0, 0.0, 0.0])
        elif "bread" in lowered:
            vectors.append([0.0, 1.0, 0.0])
        else:
            vectors.append([0.0, 0.0, 1.0])
    return vectors


class AgenticResearchPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_returns_prompt_from_ranked_search_and_chunks(self) -> None:
        result = await agentic_run(
            "python async search",
            search_top_k=2,
            search_max_results_to_keep=1,
            chunk_max_results_to_keep=1,
            crawl_max_chunk_tokens=40,
            crawl_overlap_tokens=0,
            embedder=_fake_embedder,
            search_fn=_fake_search,
            crawl_fn=_fake_crawl,
        )

        self.assertIn("SEARCH-GROUNDED ANSWER PROMPT", result.answer)
        self.assertIn("CRITICAL INSTRUCTIONS", result.answer)
        self.assertIn("You are answering a question using search results.", result.answer)
        self.assertIn("TODAY", result.answer)
        self.assertIn("Use TODAY to understand relative dates", result.answer)
        self.assertIn("If the RESULTS text contains dates", result.answer)
        self.assertIn("RESULTS", result.answer)
        self.assertIn("RESULT 1", result.answer)
        self.assertIn("TITLE 1\n======\nPython Async Search", result.answer)
        self.assertIn("URL 1\n======\nhttps://example.com/python", result.answer)
        self.assertIn(
            "SEARCH PREVIEW 1\n======\nPython asyncio search guide.",
            result.answer,
        )
        self.assertIn("RELEVANT TEXT 1\n======", result.answer)
        self.assertIn(
            "----- RELEVANT CHUNK 1 -----\nPython asyncio search uses async tasks.",
            result.answer,
        )
        self.assertIn("python async search", result.answer)
        self.assertEqual(result.answer.count("QUESTION"), 2)
        self.assertEqual(result.answer.count("\nTODAY\n"), 2)
        self.assertNotIn("START", result.answer)
        self.assertNotIn("END", result.answer)
        self.assertNotIn("Bread Recipes", result.answer)

    async def test_pipeline_formats_empty_results_prompt(self) -> None:
        result = await agentic_run(
            "no results",
            search_top_k=2,
            search_fn=lambda query, limit: [],
            crawl_fn=_fake_crawl,
        )

        self.assertIn("RESULTS", result.answer)
        self.assertIn("QUESTION", result.answer)
        self.assertIn("no results", result.answer)
        self.assertNotIn("RESULT 1", result.answer)
        self.assertNotIn("RELEVANT TEXT 1", result.answer)

    async def test_pipeline_ranks_chunks_in_one_global_pool(self) -> None:
        result = await agentic_run(
            "python async search",
            search_top_k=2,
            search_max_results_to_keep=2,
            chunk_max_results_to_keep=1,
            crawl_max_chunk_tokens=40,
            crawl_overlap_tokens=0,
            embedder=_fake_embedder,
            search_fn=_fake_search,
            crawl_fn=_fake_crawl,
        )

        self.assertIn("RESULT 1", result.answer)
        self.assertIn("RESULT 2", result.answer)
        self.assertIn(
            "----- RELEVANT CHUNK 1 -----\nPython asyncio search uses async tasks.",
            result.answer,
        )
        self.assertEqual(result.answer.count("----- RELEVANT CHUNK 1 -----"), 1)

    async def test_pipeline_rejects_bm25_only_configuration(self) -> None:
        with self.assertRaises(ValueError):
            await agentic_run(
                "python async search",
                search_dense_weight=0.0,
                embedder=_fake_embedder,
                search_fn=_fake_search,
                crawl_fn=_fake_crawl,
            )

    async def test_pipeline_propagates_embedding_failures(self) -> None:
        call_count = 0

        async def failing_embedder(inputs: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise RuntimeError("embedding model died")
            return await _fake_embedder(inputs)

        with self.assertRaises(RuntimeError):
            await agentic_run(
                "python async search",
                search_top_k=1,
                search_max_results_to_keep=1,
                embedder=failing_embedder,
                search_fn=_fake_search,
                crawl_fn=_fake_crawl,
            )


if __name__ == "__main__":
    unittest.main()
