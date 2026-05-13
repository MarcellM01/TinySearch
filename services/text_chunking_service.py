from __future__ import annotations

from services.token_counter_service import resolve_tokenizer


def truncate_text_to_max_tokens(
    text: str,
    max_tokens: int | None,
    encoding_name: str,
) -> str:
    """Keep the start of ``text`` up to ``max_tokens`` (inclusive of tokenizer).

    ``max_tokens`` <= 0 or ``None`` means do not truncate.
    """

    if not text or max_tokens is None or max_tokens <= 0:
        return text
    encoding = resolve_tokenizer(encoding_name)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def chunk_text(
    text: str,
    max_chunk_tokens: int = 500,
    overlap_tokens: int = 80,
    encoding_name: str = "o200k_base",
) -> list[dict]:
    """
    Structure-aware text chunking.

    Markdown headings are preserved as chunk metadata when present, but callers
    can pass any plain text.
    """
    encoding = resolve_tokenizer(encoding_name)

    import re

    blocks = re.split(r"(?=\n#{1,6}\s)|\n{2,}", text.strip())

    chunks: list[dict] = []
    current_blocks: list[str] = []
    current_heading = ""

    def flush() -> None:
        nonlocal current_blocks, current_heading

        chunk_body = "\n\n".join(block.strip() for block in current_blocks if block.strip()).strip()
        if not chunk_body:
            current_blocks = []
            return

        chunks.append(
            {
                "chunk_id": len(chunks),
                "heading": current_heading,
                "text": chunk_body,
                "tokens": len(encoding.encode(chunk_body)),
            }
        )
        current_blocks = []

    def split_large_block(block: str) -> None:
        tokens = encoding.encode(block)
        step = max_chunk_tokens - overlap_tokens
        if step <= 0:
            step = max_chunk_tokens

        for start in range(0, len(tokens), step):
            piece = encoding.decode(tokens[start : start + max_chunk_tokens]).strip()
            if piece:
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "heading": current_heading,
                        "text": piece,
                        "tokens": len(encoding.encode(piece)),
                    }
                )

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", block)
        if heading_match:
            current_heading = heading_match.group(2).strip()

        block_tokens = len(encoding.encode(block))

        if block_tokens > max_chunk_tokens:
            flush()
            split_large_block(block)
            continue

        candidate = "\n\n".join(current_blocks + [block]).strip()
        candidate_tokens = len(encoding.encode(candidate))

        if candidate_tokens <= max_chunk_tokens:
            current_blocks.append(block)
        else:
            flush()
            current_blocks.append(block)

    flush()
    return chunks

chunk_markdown = chunk_text
