from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import db
import extract


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(lifespan=lifespan)


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
            emb = extract.embed(mem["value"])
            db.insert_memory(
                request.user_id, request.session_id, turn_id,
                mem["type"], mem["key"], mem["value"], mem.get("confidence", 1.0), emb,
            )
    return {"id": turn_id}


@app.post("/recall")
def recall(request: RecallRequest):
    if not request.user_id:
        return {"context": "", "citations": []}
    q_emb = extract.embed(request.query)
    if q_emb is None:
        return {"context": "", "citations": []}
    hits = db.search_memories(request.user_id, q_emb, limit=10)
    if not hits:
        return {"context": "", "citations": []}
    context = "## Known facts about this user\n" + "\n".join(f"- {h['value']}" for h in hits)
    citations = [
        {"turn_id": h["source_turn"] or "", "score": float(h["score"]), "snippet": h["value"]}
        for h in hits
    ]
    return {"context": context, "citations": citations}


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
