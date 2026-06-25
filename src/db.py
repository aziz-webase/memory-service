import os
import uuid

import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from psycopg.types.json import Json
from pgvector.psycopg import register_vector

# DB_URL is SQLAlchemy-style ("postgresql+psycopg://..."); psycopg wants plain.
DB_URL = os.getenv("DB_URL", "postgresql+psycopg://ai:ai@localhost:5432/ai")
CONNINFO = DB_URL.replace("postgresql+psycopg://", "postgresql://", 1)

# Embedding dimension must match OPENAI_EMBED_MODEL (text-embedding-3-small = 1536).
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

# Small pool — a few concurrent sessions. register_vector runs per connection so
# the `vector` type round-trips as Python lists. Opened in init_db() (open=False)
# so the extension/type exist before the pool configures any connection.
pool = ConnectionPool(CONNINFO, min_size=1, max_size=10, open=False, configure=register_vector)


def init_db() -> None:
    """Create extension + tables, then open the pool. Call once at startup."""
    # Throwaway connection (no vector registration yet) to ensure the type exists
    # BEFORE the pool's register_vector runs. Robust even if init.sql didn't run.
    with psycopg.connect(CONNINFO) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id          text PRIMARY KEY,
                session_id  text NOT NULL,
                user_id     text,
                ts          timestamptz,
                messages    jsonb NOT NULL,
                metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
                created_at  timestamptz NOT NULL DEFAULT now()
            );
            CREATE INDEX IF NOT EXISTS turns_session_idx ON turns (session_id);
            CREATE INDEX IF NOT EXISTS turns_user_idx    ON turns (user_id);
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS memories (
                id          text PRIMARY KEY,
                user_id     text NOT NULL,
                session_id  text,
                source_turn text,
                type        text NOT NULL,
                key         text NOT NULL,
                value       text NOT NULL,
                confidence  real NOT NULL DEFAULT 1.0,
                embedding   vector({EMBED_DIM}),
                active      boolean NOT NULL DEFAULT true,
                supersedes  text,
                created_at  timestamptz NOT NULL DEFAULT now(),
                updated_at  timestamptz NOT NULL DEFAULT now()
            );
            CREATE INDEX IF NOT EXISTS memories_user_idx   ON memories (user_id);
            CREATE INDEX IF NOT EXISTS memories_active_idx ON memories (user_id, active);
            """
        )
        # ponytail: no ivfflat/hnsw index — exact scan is fine at eval scale
        # (hundreds of rows). Add one if memory count grows large.
        conn.commit()

    pool.open()
    pool.wait(timeout=30)


def ping() -> None:
    """Liveness/readiness check for /health. Raises if the DB is unreachable."""
    with pool.connection() as conn:
        conn.execute("SELECT 1")


# -------------------------------------------------------
# Turns (raw log)
# -------------------------------------------------------

def insert_turn(session_id, user_id, ts, messages, metadata=None) -> str:
    """Persist one turn, return its generated id. `messages` is a list of dicts
    (e.g. [m.model_dump() for m in request.messages]); `metadata` a dict."""
    turn_id = str(uuid.uuid4())
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO turns (id, session_id, user_id, ts, messages, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (turn_id, session_id, user_id, ts, Json(messages), Json(metadata or {})),
        )
    return turn_id


def recent_turns(session_id, limit=10):
    """Newest-first turns for a session, as dicts."""
    with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute(
            """
            SELECT id, session_id, user_id, ts, messages, metadata, created_at
            FROM turns
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (session_id, limit),
        )
        return cur.fetchall()


# -------------------------------------------------------
# Memories (extracted, structured)
# -------------------------------------------------------

def insert_memory(user_id, session_id, source_turn, mtype, key, value,
                  confidence, embedding, supersedes=None) -> str:
    """Insert one extracted memory. `embedding` is a list[float] of length EMBED_DIM
    (or None). Returns the generated id."""
    mem_id = str(uuid.uuid4())
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO memories
              (id, user_id, session_id, source_turn, type, key, value, confidence, embedding, supersedes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (mem_id, user_id, session_id, source_turn, mtype, key, value, confidence, embedding, supersedes),
        )
    return mem_id


def get_user_memories(user_id, active_only=False):
    """All memories for a user, as dicts. Used by /users/{id}/memories and for
    contradiction checks. session_id is aliased to source_session per the contract."""
    extra = "AND active = true" if active_only else ""
    with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute(
            f"""
            SELECT id, type, key, value, confidence,
                   session_id AS source_session, source_turn,
                   supersedes, active, created_at, updated_at
            FROM memories
            WHERE user_id = %s {extra}
            ORDER BY created_at
            """,
            (user_id,),
        )
        return cur.fetchall()


def get_active_by_key(user_id, key):
    """Active memories for a user with the same key — supersession candidates (phase 3)."""
    with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute(
            """
            SELECT id, type, key, value, confidence, embedding
            FROM memories
            WHERE user_id = %s AND key = %s AND active = true
            """,
            (user_id, key),
        )
        return cur.fetchall()


def search_memories(user_id, query_embedding, limit=10, active_only=True):
    """Vector recall: user's memories ranked by cosine similarity to the query.
    Returns dicts with a `score` (1 - cosine distance), highest first."""
    extra = "AND active = true" if active_only else ""
    with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute(
            f"""
            SELECT id, type, key, value, confidence, session_id, source_turn,
                   1 - (embedding <=> %s::vector) AS score
            FROM memories
            WHERE user_id = %s {extra} AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, user_id, query_embedding, limit),
        )
        return cur.fetchall()


def keyword_search_memories(user_id, query_text, limit=20, active_only=True):
    """Full-text (BM25-ish) recall over memory values. Pairs with vector search for
    hybrid retrieval. Computed on the fly — fine at eval scale; add a stored tsvector
    + GIN index if memory counts grow."""
    extra = "AND active = true" if active_only else ""
    with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute(
            f"""
            SELECT id, type, key, value, confidence, session_id, source_turn,
                   ts_rank_cd(to_tsvector('english', value),
                              plainto_tsquery('english', %s)) AS score
            FROM memories
            WHERE user_id = %s {extra}
              AND to_tsvector('english', value) @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query_text, user_id, query_text, limit),
        )
        return cur.fetchall()


def deactivate_memory(mem_id) -> None:
    """Mark a memory superseded (kept for history, not deleted). For phase 3."""
    with pool.connection() as conn:
        conn.execute(
            "UPDATE memories SET active = false, updated_at = now() WHERE id = %s",
            (mem_id,),
        )


# -------------------------------------------------------
# Cleanup (used by the eval between scenarios)
# -------------------------------------------------------

def delete_session(session_id) -> None:
    with pool.connection() as conn:
        conn.execute("DELETE FROM memories WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM turns    WHERE session_id = %s", (session_id,))


def delete_user(user_id) -> None:
    with pool.connection() as conn:
        conn.execute("DELETE FROM memories WHERE user_id = %s", (user_id,))
        conn.execute("DELETE FROM turns    WHERE user_id = %s", (user_id,))


# Runnable smoke check: `python db.py` after `docker compose up` (needs a live DB).
if __name__ == "__main__":
    from datetime import datetime, timezone

    init_db()
    ping()
    delete_user("smoke-user")  # clean any residue from a previous failed run

    tid = insert_turn(
        "smoke-sess", "smoke-user", datetime.now(timezone.utc),
        [{"role": "user", "content": "hello"}], {"k": "v"},
    )
    rows = recent_turns("smoke-sess")
    assert rows and rows[0]["id"] == tid, "turn insert/read failed"
    assert rows[0]["messages"][0]["content"] == "hello", "jsonb round-trip failed"

    emb = [0.1] * EMBED_DIM
    mid = insert_memory("smoke-user", "smoke-sess", tid, "fact", "greeting", "said hello", 0.9, emb)
    assert any(m["id"] == mid for m in get_user_memories("smoke-user")), "memory insert/read failed"
    hits = search_memories("smoke-user", emb, limit=5)
    assert any(h["id"] == mid for h in hits), "vector search failed"

    delete_user("smoke-user")
    assert not recent_turns("smoke-sess"), "delete (turns) failed"
    assert not get_user_memories("smoke-user"), "delete (memories) failed"
    print("db.py: round-trip OK (turns + memories + vector search)")
