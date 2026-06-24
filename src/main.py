import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import db
import extract
import supersede


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(lifespan=lifespan)

# Noise gate: if the best memory is less similar than this to the query, recall
# returns empty instead of forcing weakly-related facts. Tune against real scores.
RECALL_MIN_SCORE = float(os.getenv("RECALL_MIN_SCORE", "0.3"))


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    messages: List[Message]
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class RecallRequest(BaseModel):
    query: str
    session_id: str
    user_id: Optional[str] = None
    max_tokens: int = 1024


class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    limit: int = 10


@app.get("/health")
def health_check():
    try:
        db.ping()
    except Exception:
        raise HTTPException(status_code=503, detail="db unavailable")
    return {"status": "healthy"}


@app.post("/turns", status_code=201)
def create_turn(request: ChatRequest):
    msgs = [m.model_dump() for m in request.messages]
    turn_id = db.insert_turn(request.session_id, request.user_id, request.timestamp, msgs, request.metadata)
    if request.user_id:                         # memorylar user'ga bog'liq
        for mem in extract.extract_memories(msgs):
            candidates = db.get_active_by_key(request.user_id, mem["key"])
            stale = supersede.resolve_supersession(mem, candidates)
            emb = extract.embed(mem["value"])
            db.insert_memory(
                request.user_id, request.session_id, turn_id,
                mem["type"], mem["key"], mem["value"], mem.get("confidence", 1.0), emb,
                supersedes=(stale[0] if stale else None),
            )
            for old_id in stale:
                db.deactivate_memory(old_id)
    return {"id": turn_id}


@app.post("/recall")
def recall(request: RecallRequest):
    if not request.user_id:
        return {"context": "", "citations": []}

    # A. Stable profile — active facts/preferences are user-level, always included
    #    (this is why multi-hop works: the answer need not appear in the query).
    profile = [m for m in db.get_user_memories(request.user_id, active_only=True)
               if m["type"] in ("fact", "preference")]

    # B. Query-relevant memories (any type) above the noise gate.
    relevant = []
    q_emb = extract.embed(request.query)
    if q_emb is not None:
        relevant = [h for h in db.search_memories(request.user_id, q_emb, limit=10)
                    if h["score"] >= RECALL_MIN_SCORE]

    if not profile and not relevant:
        return {"context": "", "citations": []}

    parts, citations, seen = [], [], set()
    if profile:
        parts.append("## Known facts about this user")
        for m in profile:
            parts.append(f"- {m['value']}")
            seen.add(m["id"])
            citations.append({"turn_id": m["source_turn"] or "", "score": 1.0, "snippet": m["value"]})
    rel_new = [h for h in relevant if h["id"] not in seen]
    if rel_new:
        parts.append("\n## Relevant to this query")
        for h in rel_new:
            parts.append(f"- {h['value']}")
            citations.append({"turn_id": h["source_turn"] or "", "score": float(h["score"]), "snippet": h["value"]})

    return {"context": "\n".join(parts), "citations": citations}


@app.post("/search")
def search(request: SearchRequest):
    return {"results": []}


@app.get("/users/{user_id}/memories")
def get_memories(user_id: str):
    return {"memories": db.get_user_memories(user_id)}


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    db.delete_session(session_id)


@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: str):
    db.delete_user(user_id)
