"""Hybrid retrieval for the 'relevant' section of /recall.

Vector search (pgvector cosine) and keyword search (Postgres full-text) are fused with
Reciprocal Rank Fusion, then the fused candidates are cross-encoder reranked. RRF needs no
score normalization between the two retrievers — it fuses on rank — which is why pure
cosine top-k misses keyword-heavy queries that this recovers.
"""
import os

import db
import rerank

RRF_K = 60  # standard RRF dampening constant
CANDIDATE_K = int(os.getenv("RETRIEVE_CANDIDATE_K", "20"))


def _rrf(*ranked_lists):
    """Fuse ranked lists by reciprocal rank. Returns (id->score, id->row)."""
    scores, rows = {}, {}
    for lst in ranked_lists:
        for rank, m in enumerate(lst):
            mid = m["id"]
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (RRF_K + rank)
            rows.setdefault(mid, m)
    return scores, rows


def relevant(user_id, query, q_emb, top_n=8):
    """Return up to top_n query-relevant memories, reranked. Each carries `score`
    (0..1 rerank relevance) for gating and citations."""
    vec = db.search_memories(user_id, q_emb, limit=CANDIDATE_K) if q_emb is not None else []
    kw = db.keyword_search_memories(user_id, query, limit=CANDIDATE_K)
    if not vec and not kw:
        return []
    scores, rows = _rrf(vec, kw)
    fused = sorted(rows.values(), key=lambda m: scores[m["id"]], reverse=True)
    return rerank.rerank(query, fused, top_n=top_n)
