# ONNX embedding bundle (`all-MiniLM-L6-v2`)

TinySearch prefers this folder when `model.onnx` and tokenizer files are present: same
mean-pooled BERT vectors as `sentence-transformers/all-MiniLM-L6-v2`, with much faster
cold start than loading PyTorch inside `SentenceTransformer`.

## Generate (once)

From the repo root, with `onnx` installed for export (`pip install onnx`):

```bash
python scripts/export_embedding_onnx.py
```

This downloads the HF model once, then writes `model.onnx` (~90 MB) plus tokenizer
artifacts into this directory. Commit the result if you want embeddings to work
offline without the PyTorch load path.

## Override path

Set `TINYSEARCH_ONNX_MODEL_DIR` to an absolute directory that contains the same layout
(`model.onnx` + tokenizer files from `save_pretrained`).
