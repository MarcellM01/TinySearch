"""
Export sentence-transformers/all-MiniLM-L6-v2 to ONNX + tokenizer files for TinySearch.

Writes to models/all-minilm-l6-v2-onnx/ (repo root). Requires PyTorch, transformers,
sentence-transformers, and the ``onnx`` package (``pip install onnx``). Runtime
inference uses ``onnxruntime`` only.

Edit the variables under ``if __name__ == "__main__"`` then run:

  python scripts/export_embedding_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _export(hf_model_id: str, out_dir: Path) -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "The ``onnx`` package is required to export (``pip install onnx``). "
            "Runtime uses ``onnxruntime`` only."
        ) from exc

    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    class MeanPooledBert(torch.nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(name)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / summed_mask

    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(hf_model_id)
    tok.save_pretrained(out_dir)

    model = MeanPooledBert(hf_model_id)
    model.eval()
    dummy_ids = torch.ones(1, 32, dtype=torch.long)
    dummy_mask = torch.ones(1, 32, dtype=torch.long)
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        dynamo=False,
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "sentence_embedding": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"Wrote tokenizer + {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    _HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    _OUT_DIR = _PROJECT_ROOT / "models" / "all-minilm-l6-v2-onnx"
    _export(_HF_MODEL_ID, _OUT_DIR)
