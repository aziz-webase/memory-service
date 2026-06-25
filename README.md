# Memory Service

A memory service for an AI agent. It ingests conversation turns, extracts structured
knowledge, tracks how facts change over time, and answers recall queries that decide what
context the agent sees next.

Single deployable: a FastAPI app + Postgres (with pgvector) behind `docker compose up`.

## Architecture

```
                         POST /turns                 POST /recall  /search
                              |                            |
                              v                            v
        +-------------------------------------------------------------+
        |                      FastAPI app (main.py)                  |
        |                                                             |
        |   /turns: persist raw turn ─► extract ─► supersede ─► store |
        |   /recall: profile (always) + relevant (gated) + budget     |
        +------------------+-------------------------+----------------+
                           |                         |
                  extract.py / supersede.py          | db.py (psycopg pool)
                  (OpenAI: chat + embeddings)         |
                           |                         v
                           |             +-------------------------+
                           +────────────►|  Postgres + pgvector    |
                                         |  turns (raw log)        |
                                         |  memories (typed +      |
                                         |   embedding + active)   |
                                         +-------------------------+
                                              named volume (persist)
```

A turn arrives at `/turns`. We persist the raw turn (audit/replay), then **synchronously**
extract structured memories from it, resolve any contradictions against existing memories,
embed each memory, and store it. Everything is committed before `/turns` returns, so a
memory is immediately queryable — no eventual consistency. `/recall` assembles context from
the stored memories. Extraction and the supersession judge use OpenAI; everything else is
plain SQL over one Postgres instance.

## Backing store: Postgres + pgvector

One store for everything: raw turns, structured memories, **and** their embedding vectors
(`pgvector`). Why a single relational store rather than a dedicated vector DB:

- **Synchronous correctness for free.** The contract requires that after `/turns` returns,
  the data is readable. One ACID transaction per write gives that with no sync dance.
- **Persistence for free.** A named Docker volume on the Postgres data dir survives
  `docker compose down && up`. No separate vector-store persistence to manage.
- **Cross-signal queries in one place.** Filtering by `user_id` / `active` and ranking by
  cosine distance happen in the same `SELECT` — no app-side join between a SQL store and a
  vector store.
- **Few moving parts.** A memory service for "a few concurrent sessions" (per the brief's
  scope) does not need a distributed vector engine. pgvector exact search is plenty at this
  scale; an HNSW/IVFFlat index is a one-line add if memory counts grow.

Two tables: `turns` (raw conversation log) and `memories` (extracted, typed, embedded,
with `active` / `supersedes` for fact evolution). See `src/db.py`.

## Extraction pipeline

`src/extract.py`. Each turn is sent to `gpt-4o-mini` with a **strict JSON schema**
(structured outputs), which returns a list of memories:

```
{ "type": "fact|preference|opinion|event", "key": "...", "value": "...", "confidence": 0..1 }
```

- **`type`** separates durable identity (`fact`), tastes (`preference`), stances
  (`opinion`), and time-bound happenings (`event`).
- **`key`** is a *canonical* snake_case topic id (`employer`, `location`, `pet_name`, ...).
  This is the load-bearing design choice: two statements about the same topic collide on
  the same `key`, which is what makes supersession possible.
- **`value`** is a self-contained sentence; **`confidence`** is higher for explicit
  statements, lower for inferred ones.

We extract personal facts, preferences/opinions, corrections, and **implicit** facts:
"walking Biscuit this morning" → `pet_name = Biscuit`; "the best part of moving to Berlin
was the parks" → `location = Berlin` (plus the separate opinion). Each `value` is embedded
with `text-embedding-3-small` (1536-d) and stored on the row.

**What we miss / limits.** Extraction quality is bounded by the prompt and the model — long
multi-topic turns can drop a secondary fact; the canonical-key vocabulary is a fixed list,
so a novel topic gets a free-form key and won't collide as cleanly for supersession.
Extraction runs inline in `/turns` (the brief allows up to 60s), so a turn costs one chat
call plus one embedding per extracted memory.

## Recall strategy

`/recall` builds context in two prioritized sections, then trims to the token budget:

1. **Known facts (always included).** All *active* `fact` / `preference` memories for the
   user. These are user-level identity and are surfaced regardless of the query. This is
   why **multi-hop** works: "what city does the user with the dog Biscuit live in?" is
   answered because `location` is in the profile even though "Berlin" never appears in the
   query — a pure top-k vector search can't retrieve an answer whose tokens aren't in the
   query.
2. **Relevant to this query.** Hybrid retrieval — pgvector cosine **and** Postgres
   full-text search, fused with **Reciprocal Rank Fusion**, then **cross-encoder reranked**
   (`bge-reranker-v2-m3`, `src/retrieve.py` + `src/rerank.py`). The reranker's relevance
   (0..1) gates out noise via `RECALL_MIN_SCORE`. RRF fuses on rank (no score
   normalization), so keyword-heavy queries a pure vector search misses are recovered.

**Token budget.** We fill in strict priority order — profile first, then relevant — adding
lines while they fit under `max_tokens` (estimated at ~4 chars/token; the brief says
approximate is fine), and stop at the first line that doesn't fit. A lower-priority memory
never displaces a higher-priority one. Empty result on a cold/unknown user is `{"context":
"", "citations": []}` — never an error.

**Noise resistance.** We only ever return *stored* memories, never invented ones, so we
can't hallucinate a fact for an unmentioned topic. A cold user returns empty context.

## Fact evolution

`src/supersede.py`, **hybrid**:

- For **single-valued keys** (`employer`, `location`, `job_title`, ...) a rule applies: a
  new value supersedes the old — no LLM call.
- For ambiguous keys, an **LLM judge** decides whether the new memory replaces existing
  ones or is independent (e.g. a second pet is additive, not a replacement).
- **Events never supersede** — they accumulate.

Superseded memories are **soft-deleted**: `active = false`, kept in the table, with the new
memory carrying a `supersedes` pointer. `/recall` reads only `active` rows (returns the
current fact); `/users/{id}/memories` shows the full chain (history preserved). Example:
"I work at Stripe" → later "I joined Notion" leaves Stripe `active=false` and Notion
`active=true, supersedes=<stripe id>`.

**Opinion arcs** ("I love TypeScript" → "TS generics are annoying" → "TS is fine for big
projects") are handled *partially*: opinions are typed `opinion` and accumulate rather than
overwrite, so the arc is preserved as history and is recallable. We do not yet synthesize
the arc into a single "current stance" summary — that's noted as future work.

## Tradeoffs

- **One Postgres vs. specialized stores** — chose operational simplicity and synchronous
  correctness over best-in-class vector latency. Fine at the brief's scale.
- **Recall = profile-always + hybrid-relevant + budget.** The relevant section uses vector
  + keyword fused with RRF and a cross-encoder reranker; the profile section is a straight
  active-facts dump. Reranking adds CPU latency to `/recall` (the model loads lazily on the
  first call), acceptable at this scale. Char-based token estimate avoids a `tiktoken`
  dependency at the cost of exactness.
- **Inline extraction** — simpler and meets the sync-correctness requirement, at the cost of
  `/turns` latency (one LLM + N embedding calls). The brief explicitly allows this.
- **Fixed canonical-key vocabulary** — strong supersession on known topics, weaker on novel
  ones.

## Failure modes

- **No data / cold session** — `/recall` and `/search` return empty, `200`, never error.
- **Missing API key** — OpenAI calls are caught and degrade: `/turns` still persists the raw
  turn and returns `201`, but extracts no memories; `/recall` returns empty. The service
  stays up.
- **DB unreachable at startup** — the pool waits up to 30s; `/health` returns `503` while the
  DB is down, `200` when ready (used as the compose/Docker healthcheck).
- **Malformed / oversized / unicode input** — FastAPI validation returns `4xx` (422), the
  service never crashes (proven by `tests/test_contract.py`). Unicode round-trips through
  `jsonb`.
- **Restart mid-flight** — writes are single committed transactions; a named volume makes
  restart invisible to clients (proven by the restart-persistence test).
- **First recall after boot** — the cross-encoder reranker loads lazily (a few seconds on
  CPU); later recalls are fast. If it can't load, reranking falls back to fusion order.

## Cross-session scoping

Memories are scoped to `user_id`. Knowledge **is intentionally shared across sessions for
the same user** — that's the point of long-term memory (a fact stated in session 1 is
recalled in session 9). Sessions belonging to **different users never bleed** (proven by the
isolation test). `session_id` is recorded for provenance and is the unit for
`DELETE /sessions/{id}`.

## Running

```bash
cp .env.example .env          # set OPENAI_API_KEY (needed for extraction/recall)
docker compose up -d          # app on :8080 (override host port via HOST_PORT in .env)
until curl -sf http://localhost:8080/health; do sleep 1; done
```

Models (set in `.env`): `OPENAI_MODEL=gpt-4o-mini` (extraction + supersession judge),
`OPENAI_EMBED_MODEL=text-embedding-3-small` (embeddings). `RECALL_MIN_SCORE` tunes the noise
gate.

> Note: on a clean machine `docker compose up` works as-is. If your local Docker Compose v5
> hits a `bake` panic, build with `docker build -t memory-service-app .` then
> `docker compose up -d` — the compose file itself is standard.

## Tests

```bash
# 1. Recall quality (5 scenarios: location, employment change, implicit pet, multi-hop, noise)
MEMORY_URL=http://localhost:8080 python tests/eval.py

# 2. Contract & robustness (roundtrip, cross-user isolation, malformed input, restart persistence)
MEMORY_URL=http://localhost:8080 python tests/test_contract.py

# 3. DB plumbing smoke (turns + memories + vector search), from inside the container
docker compose exec app python db.py
```

`tests/eval.py` is the iteration loop — it ingests `fixtures/eval.json` and reports how many
expected facts surfaced in `/recall`. See `CHANGELOG.md` for the design history and the
metric at each step.
