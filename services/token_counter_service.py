from __future__ import annotations

from functools import lru_cache
from typing import Any


DEFAULT_ENCODING_NAME = "o200k_base"


class HuggingFaceTokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


class CharacterTokenizerAdapter:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class TiktokenAdapter:
    def __init__(self, encoding: Any) -> None:
        self._encoding = encoding

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)


def _tiktoken_module():
    try:
        import tiktoken
    except Exception:
        return None
    return tiktoken


def _get_tiktoken_encoding(name: str):
    tiktoken = _tiktoken_module()
    if tiktoken is None:
        return None
    try:
        return TiktokenAdapter(tiktoken.get_encoding(name))
    except Exception:
        return None


def _get_tiktoken_encoding_for_model(name: str):
    tiktoken = _tiktoken_module()
    if tiktoken is None:
        return None
    try:
        return TiktokenAdapter(tiktoken.encoding_for_model(name))
    except Exception:
        return None


def _resolve_huggingface_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        return None
    return HuggingFaceTokenizerAdapter(tokenizer)


@lru_cache(maxsize=64)
def resolve_tokenizer(
    tokenizer_name: str | None = None,
    *,
    model: str | None = None,
):
    """
    Resolve the best available tokenizer for token counting.

    Prefer a locally available Hugging Face tokenizer for model-like names,
    then tiktoken's explicit encoding/model mapping, then fall back to the
    default OpenAI tokenizer.
    """
    if tokenizer_name:
        encoding = _get_tiktoken_encoding(tokenizer_name)
        if encoding is not None:
            return encoding

    candidates = [tokenizer_name, model]
    for candidate in candidates:
        if not candidate:
            continue
        hf_tokenizer = _resolve_huggingface_tokenizer(candidate)
        if hf_tokenizer is not None:
            return hf_tokenizer
        encoding = _get_tiktoken_encoding_for_model(candidate)
        if encoding is not None:
            return encoding
        encoding = _get_tiktoken_encoding(candidate)
        if encoding is not None:
            return encoding
    fallback = _get_tiktoken_encoding(DEFAULT_ENCODING_NAME)
    if fallback is not None:
        return fallback
    return CharacterTokenizerAdapter()


def token_count(
    text: str,
    encoding_name: str | None = DEFAULT_ENCODING_NAME,
    *,
    model: str | None = None,
) -> int:
    encoding = resolve_tokenizer(encoding_name, model=model)
    return len(encoding.encode(text))


def encode_tokens(
    text: str,
    encoding_name: str | None = DEFAULT_ENCODING_NAME,
    *,
    model: str | None = None,
) -> list[int]:
    return resolve_tokenizer(encoding_name, model=model).encode(text)


def decode_tokens(
    tokens: list[int],
    encoding_name: str | None = DEFAULT_ENCODING_NAME,
    *,
    model: str | None = None,
) -> str:
    return resolve_tokenizer(encoding_name, model=model).decode(tokens)
