from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import db


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
    turn_id = db.insert_turn(
        request.session_id,
        request.user_id,
        request.timestamp,
        [m.model_dump() for m in request.messages],
        request.metadata,
    )
    return {"id": turn_id}


@app.post("/recall")
def recall(request: RecallRequest):
    # ponytail: phase 0 — empty. Real ranking (hybrid + rerank) lands in phase 4.
    return {"context": "", "citations": []}


@app.post("/search")
def search(request: SearchRequest):
    return {"results": []}


@app.get("/users/{user_id}/memories")
def get_memories(user_id: str):
    return {"memories": []}


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    db.delete_session(session_id)


@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: str):
    db.delete_user(user_id)
