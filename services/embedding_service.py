from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


# Fixed local model (not configurable); weights cached outside the repo by default
# (avoids huge trees under the workspace, Defender/IDE churn, and duplicate downloads).
FIXED_LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_EMBEDDING_BACKEND = "default"
DEFAULT_EMBEDDING_OPENAI_ENV_FILE = ".env"

SUPPORTED_EMBEDDING_BACKENDS = (
    "default",
    "openai_compatible",
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EMBED_LOCK = threading.Lock()


def _sentence_transformers_cache_folder() -> str:
    """HF / sentence-transformers download cache. Override with TINYSEARCH_HF_CACHE."""
    raw = os.environ.get("TINYSEARCH_HF_CACHE", "").strip()
    if raw:
        path = Path(raw).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return str(path.resolve())
    if os.name == "nt":
        base = Path(
            os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        )
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    path = base / "tinysearch" / "huggingface"
    path.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())


def _onnx_bundle_dir() -> Path:
    raw = os.environ.get("TINYSEARCH_ONNX_MODEL_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (_PROJECT_ROOT / "models" / "all-minilm-l6-v2-onnx").resolve()


def _onnx_bundle_ready() -> bool:
    d = _onnx_bundle_dir()
    if not (d / "model.onnx").is_file():
        return False
    if (d / "tokenizer.json").is_file():
        return True
    if (d / "tokenizer_config.json").is_file() and (d / "vocab.txt").is_file():
        return True
    return (d / "tokenizer.model").is_file()


def default_local_will_use_onnx_bundle() -> bool:
    """True when ``embedding_backend`` ``default`` will embed via the shipped ONNX bundle."""
    return _onnx_bundle_ready()


@lru_cache(maxsize=1)
def _load_onnx_runtime_bundle() -> tuple[Any, Any]:
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "ONNX embedding bundle is present but `onnxruntime` (and `transformers`) "
            "are required. Install with: pip install onnxruntime"
        ) from exc

    d = _onnx_bundle_dir()
    session = ort.InferenceSession(
        str(d / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    tokenizer = AutoTokenizer.from_pretrained(str(d), local_files_only=True)
    return session, tokenizer


def _embed_onnx_sync(inputs: list[str]) -> list[list[float]]:
    import numpy as np

    if not inputs:
        return []

    t0 = time.perf_counter()
    session, tokenizer = _load_onnx_runtime_bundle()
    t_after_load = time.perf_counter()
    n_chars = sum(len(s) for s in inputs)
    batch_size = 32
    all_rows: list[list[float]] = []
    with _EMBED_LOCK:
        t_embed0 = time.perf_counter()
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="np",
            )
            ort_inputs = {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }
            out = session.run(("sentence_embedding",), ort_inputs)[0]
            all_rows.extend(out.tolist())
        embed_s = time.perf_counter() - t_embed0
    total_s = time.perf_counter() - t0
    if _embed_timing_log_enabled():
        prep_s = t_embed0 - t0
        lock_wait_s = t_embed0 - t_after_load
        print(
            f"[embedding] backend=onnx_cpu n_inputs={len(inputs)} chars={n_chars} "
            f"embed_s={embed_s:.3f} prep_s={prep_s:.3f} lock_wait_s={lock_wait_s:.3f} "
            f"total_s={total_s:.3f} bundle={_onnx_bundle_dir()}",
            file=sys.stderr,
            flush=True,
        )
    return all_rows


def _embed_timing_log_enabled() -> bool:
    v = os.environ.get("TINYSEARCH_LOG_EMBED_TIMING", "0").strip().lower()
    return v not in ("0", "false", "no", "off")


def _as_vectors(value: Any) -> list[list[float]]:
    if hasattr(value, "reshape") and hasattr(value, "ndim") and value.ndim == 1:
        value = value.reshape(1, -1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not value:
        return []
    if isinstance(value[0], (int, float)):
        return [[float(x) for x in value]]
    vectors = [list(vector) for vector in value]
    return [[float(item) for item in vector] for vector in vectors]


def normalize_embedding_backend(backend: str) -> str:
    key = (backend or DEFAULT_EMBEDDING_BACKEND).strip().lower()
    if key in ("default", "local", "sentence_transformers"):
        return "default"
    if key in ("openai_compatible", "openai"):
        return "openai_compatible"
    if key == "llama_cpp":
        raise ValueError(
            "embedding_backend 'llama_cpp' is no longer supported; "
            "use 'default' (fixed local MiniLM) or 'openai_compatible' (credentials in .env)"
        )
    return key


def _resolve_openai_env_path(openai_env_file: str | Path | None) -> Path:
    raw = (openai_env_file or DEFAULT_EMBEDDING_OPENAI_ENV_FILE).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


def _parse_openai_env_file(path: Path) -> tuple[str | None, str, str]:
    """Read base URL, API key, and embedding model name from a .env-style file."""
    if not path.is_file():
        raise RuntimeError(
            f"openai_compatible backend requires {path} with OPENAI_BASE_URL (optional), "
            "OPENAI_API_KEY, and OPENAI_EMBEDDING_MODEL (or EMBEDDING_MODEL)"
        )
    text = path.read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        name, _, rest = stripped.partition("=")
        key = name.strip().upper()
        val = rest.strip().strip('"').strip("'")
        if key:
            values[key] = val

    def pick(*keys: str) -> str | None:
        for k in keys:
            v = values.get(k.upper())
            if v:
                return v
        return None

    api_key = pick("OPENAI_API_KEY", "API_KEY")
    if not api_key:
        raise RuntimeError(
            f"{path} must set OPENAI_API_KEY (or API_KEY) for openai_compatible embeddings"
        )
    base_raw = pick("OPENAI_BASE_URL", "BASE_URL", "API_URL")
    base_url = base_raw.strip() if base_raw else None
    if base_url == "":
        base_url = None
    model = pick(
        "OPENAI_EMBEDDING_MODEL",
        "EMBEDDING_MODEL",
        "MODEL_NAME",
        "MODEL",
    )
    if not model:
        raise RuntimeError(
            f"{path} must set OPENAI_EMBEDDING_MODEL (or EMBEDDING_MODEL / MODEL_NAME) "
            "for openai_compatible embeddings"
        )
    return base_url, api_key, model


# ---------------------------------------------------------------------------
# default backend: fixed sentence-transformers model (cache via _sentence_transformers_cache_folder).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_fixed_sentence_transformer() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "The default embedding backend requires `sentence-transformers`. "
            "Install with: pip install sentence-transformers"
        ) from exc

    try:
        return SentenceTransformer(
            FIXED_LOCAL_EMBEDDING_MODEL,
            cache_folder=_sentence_transformers_cache_folder(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to load fixed local embedding model {FIXED_LOCAL_EMBEDDING_MODEL!r}"
        ) from exc


def _embed_default_local_sync(inputs: list[str]) -> list[list[float]]:
    if not inputs:
        return []

    if _onnx_bundle_ready():
        return _embed_onnx_sync(inputs)

    t0 = time.perf_counter()
    model = _load_fixed_sentence_transformer()
    t_after_load = time.perf_counter()
    n_chars = sum(len(s) for s in inputs)
    with _EMBED_LOCK:
        t_embed0 = time.perf_counter()
        try:
            raw = model.encode(
                inputs,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            embed_s = time.perf_counter() - t_embed0
        except Exception as exc:
            raise RuntimeError("failed to generate default local embeddings") from exc

    total_s = time.perf_counter() - t0
    if _embed_timing_log_enabled():
        prep_s = t_embed0 - t0
        lock_wait_s = t_embed0 - t_after_load
        device = getattr(getattr(model, "device", None), "type", "?")
        print(
            f"[embedding] backend=default n_inputs={len(inputs)} chars={n_chars} "
            f"embed_s={embed_s:.3f} prep_s={prep_s:.3f} lock_wait_s={lock_wait_s:.3f} "
            f"total_s={total_s:.3f} model={FIXED_LOCAL_EMBEDDING_MODEL!r} device={device}",
            file=sys.stderr,
            flush=True,
        )

    return _as_vectors(raw)


def _create_default_local_embedder() -> Callable[[list[str]], Any]:
    async def embedder(inputs: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(_embed_default_local_sync, list(inputs))

    return embedder


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP backend (reads URL / key / model from .env at repo root).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _load_openai_client(base_url: str | None, api_key: str) -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "The openai_compatible backend requires the `openai` package. "
            "Install with: pip install openai"
        ) from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return OpenAI(**kwargs)
    except Exception as exc:
        raise RuntimeError("failed to construct OpenAI-compatible client") from exc


def _embed_openai_compatible_sync(
    client: Any,
    model_name: str,
    inputs: list[str],
    *,
    base_url: str | None,
) -> list[list[float]]:
    if not inputs:
        return []

    t0 = time.perf_counter()
    n_chars = sum(len(s) for s in inputs)
    try:
        response = client.embeddings.create(model=model_name, input=inputs)
    except Exception as exc:
        raise RuntimeError(
            f"failed to generate embeddings with openai_compatible model {model_name!r} "
            f"(base_url={base_url!r})"
        ) from exc
    embed_s = time.perf_counter() - t0
    if _embed_timing_log_enabled():
        print(
            f"[embedding] backend=openai_compatible n_inputs={len(inputs)} chars={n_chars} "
            f"embed_s={embed_s:.3f} model={model_name!r} base_url={base_url!r}",
            file=sys.stderr,
            flush=True,
        )

    return [list(item.embedding) for item in response.data]


def _create_openai_compatible_embedder(env_path: Path) -> Callable[[list[str]], Any]:
    base_url, api_key, model_name = _parse_openai_env_file(env_path)
    client = _load_openai_client(base_url, api_key)

    async def embedder(inputs: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(
            _embed_openai_compatible_sync,
            client,
            model_name,
            list(inputs),
            base_url=base_url,
        )

    return embedder


# ---------------------------------------------------------------------------
# Public factory.
# ---------------------------------------------------------------------------


def create_embedder(
    *,
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    openai_env_file: str | Path | None = None,
) -> Callable[[list[str]], Any]:
    backend_key = normalize_embedding_backend(backend)
    if backend_key == "default":
        return _create_default_local_embedder()
    if backend_key == "openai_compatible":
        return _create_openai_compatible_embedder(_resolve_openai_env_path(openai_env_file))
    raise ValueError(
        f"unknown embedding_backend {backend!r}; expected one of {SUPPORTED_EMBEDDING_BACKENDS} "
        "(aliases: sentence_transformers, local -> default; openai -> openai_compatible)"
    )
