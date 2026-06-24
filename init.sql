-- Runs once on first db boot (empty volume). pgvector binary ships in the
-- pgvector/pgvector image; this just enables the extension.
CREATE EXTENSION IF NOT EXISTS vector;

-- ponytail: your own turns/memories tables go here (or let the app create them
-- on startup) in phase 2. Phase-0 skeleton only needs the extension; langchain
-- PGVector creates langchain_pg_collection / langchain_pg_embedding itself.
