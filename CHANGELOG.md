# Changelog

Iteration history for the memory service. Newest first. Each entry: what changed,
why, what it produced, and what's next.

## v3 — Hybrid supersession (rule + LLM judge)
**What changed:** `supersede.py`. On each new fact, fetch active memories with the
same `key`. A RULE supersedes for single-valued keys (employer, location, job_title,
hometown, dietary, marital/relationship status); an LLM JUDGE decides for ambiguous
keys (e.g. is a second pet independent, or a replacement?); events never supersede.
The superseded memory is set `active=false` (kept, not deleted); the new memory stores
a `supersedes` pointer to it.

**Why:** the fact-evolution category. A pure append-only log returns stale facts
("I work at Stripe" long after the user joined Notion). Soft-supersede keeps the
history inspectable, which the contract requires via `/users/{id}/memories`.

**Result:** verified on the `employment_change` scenario — `employer="Works at Stripe"`
is now `active=false`; `employer="...Notion"` is `active=true` with `supersedes` → the
Stripe id; same for `job_title`. `/recall` returns only Notion (search filters
`active=true`); both remain visible in `/users/{id}/memories`.

**Next:** real recall ranking. Current `/recall` is vanilla cosine top-k, which the
brief says won't score — add hybrid (vector + keyword), reranking, a token-budget
priority policy, and multi-hop.

## v2 — LLM extraction (structured, typed memories)
**What changed:** `extract.py`. `gpt-4o-mini` with a strict JSON schema turns each turn
into typed memories `{type, key, value, confidence}` with canonical snake_case keys
(employer, location, pet_name, ...). Each value is embedded (`text-embedding-3-small`,
1536-d) into a pgvector `memories` table. `/recall` now does cosine top-k over the
user's active memories; `/users/{id}/memories` returns the structured store.

**Why:** a message log isn't a memory service — extraction is the core. Canonical keys
are a deliberate choice: they make same-topic statements collide on `key`, which is what
makes supersession (v3) possible.

**Why these models:** `gpt-4o-mini` is cheap and reliably honors structured outputs for
extraction; `text-embedding-3-small` (1536-d) is a low-cost embedding that's plenty for
short memory values.

**Result:** `/users/{id}/memories` returns clean typed memories — e.g. `employer`,
`job_title` — including implicit facts ("walking Biscuit" → `pet_name`). Recall surfaces
location / pet / employer probes.
_Self-eval after this change: __/4 facts (fill in from `python tests/eval.py`)._

**Observed gap:** two `employer` memories were both active (Stripe + Notion) — no
supersession yet. Drove v3.

**Next:** supersession / contradiction handling.

## v1 — Recall self-eval harness
**What changed:** `fixtures/eval.json` (5 scenarios: location, employment_change,
implicit_pet, multi_hop, noise) + `tests/eval.py` (ingest each scenario via `/turns`,
probe `/recall`, count expected facts present in the returned context; `noise` must
return empty).

**Why:** the iteration loop the brief asks for. Without a scoreboard every later change
is a guess.

**Result:** baseline **0/4 facts, 1/5 probes** — only `noise` passes (recall was still
empty). Confirms the harness measures correctly.

**Next:** extraction, so recall has structured memories to surface.

## v0 — Dockerized skeleton
**What changed:** FastAPI service with all 7 contract endpoints, Postgres + pgvector,
named volume, `docker compose up`. `/turns` persists raw turns to a `turns` table;
`/recall`, `/search`, `/memories` return valid-but-empty shapes; deletes return 204.

**Why:** get the contract green and a deployable shell before any quality work, so there
is always a submittable artifact and a target for the eval harness.

**Result:** smoke test passes — `/health` 200, `/turns` 201 `{id}`, deletes 204, shapes
match the contract. (Local note: Docker Compose v5's `bake` builder panics on this
machine; build the image with `docker build` and run `docker compose up` — the compose
file itself is standard and works on a normal compose.)

**Next:** a self-eval harness to make quality measurable.
