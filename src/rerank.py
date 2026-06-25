"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3, baked into the image).

The model is loaded lazily on the first call (the first /recall after boot pays a
one-time load cost), then cached. CPU by default; uses CUDA if present.
"""
import math
import os

_model = None
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


def _load():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = CrossEncoder(RERANKER_MODEL, device=device, max_length=512)
    return _model


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def rerank(query, candidates, top_n=8):
    """Rerank candidate memories (dicts with a 'value') against the query. Returns the
    top_n, best first, each with `score` set to a 0..1 relevance. Falls back to input
    order on failure (e.g. model unavailable)."""
    if not candidates:
        return []
    try:
        model = _load()
        logits = model.predict([(query, c["value"]) for c in candidates])
        scored = []
        for c, logit in zip(candidates, logits):
            c = dict(c)
            c["score"] = _sigmoid(float(logit))
            scored.append(c)
        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored[:top_n]
    except Exception as e:
        print(f"[rerank error: {e}]")
        return candidates[:top_n]
