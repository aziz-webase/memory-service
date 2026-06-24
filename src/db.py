import os
import uuid

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from psycopg.types.json import Json

# DB_URL is SQLAlchemy-style ("postgresql+psycopg://..."); psycopg wants plain.
DB_URL = os.getenv("DB_URL", "postgresql+psycopg://ai:ai@localhost:5432/ai")
CONNINFO = DB_URL.replace("postgresql+psycopg://", "postgresql://", 1)

# Small pool — the challenge expects only a few concurrent sessions. Opened
# explicitly in init_db() so import never blocks on the DB.
pool = ConnectionPool(CONNINFO, min_size=1, max_size=10, open=False)


def init_db() -> None:
    """Open the pool and create tables if missing. Call once at startup.
    The `vector` extension is enabled by init.sql; here we own the raw turns log.
    """
    pool.open()
    pool.wait(timeout=30)  # block until the DB is actually reachable
    with pool.connection() as conn:
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


def ping() -> None:
    """Liveness/readiness check for /health. Raises if the DB is unreachable."""
    with pool.connection() as conn:
        conn.execute("SELECT 1")


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
    """Newest-first turns for a session, as dicts. Naive recall/search for phase 0."""
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


def delete_session(session_id) -> None:
    with pool.connection() as conn:
        conn.execute("DELETE FROM turns WHERE session_id = %s", (session_id,))


def delete_user(user_id) -> None:
    with pool.connection() as conn:
        conn.execute("DELETE FROM turns WHERE user_id = %s", (user_id,))


# Runnable smoke check: `python db.py` after `docker compose up` (needs a live DB).
if __name__ == "__main__":
    from datetime import datetime, timezone

    init_db()
    ping()
    tid = insert_turn(
        "smoke-sess", "smoke-user", datetime.now(timezone.utc),
        [{"role": "user", "content": "hello"}], {"k": "v"},
    )
    rows = recent_turns("smoke-sess")
    assert rows and rows[0]["id"] == tid, "insert/read round-trip failed"
    assert rows[0]["messages"][0]["content"] == "hello", "jsonb round-trip failed"
    delete_session("smoke-sess")
    assert not recent_turns("smoke-sess"), "delete failed"
    print("db.py: round-trip OK")
