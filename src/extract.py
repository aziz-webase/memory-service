"""Extraction: turn a raw conversation turn into structured, typed memories.

This is the core of the memory service — NOT raw-message storage. An LLM reads
the turn and emits a list of {type, key, value, confidence}. The `key` is a
canonical topic id so later statements about the same topic collide (that's what
makes supersession possible in phase 3).

Design levers you tune as the eval guides you:
  - SYSTEM prompt (what counts as a memory, how to phrase values)
  - KEY vocabulary (how aggressively topics are normalized → supersession quality)
"""
import json
import os

from openai import OpenAI

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Empty key tolerated at construction; calls fail gracefully (caught below).
_client = OpenAI(base_url=OPENAI_BASE_URL, api_key=os.getenv("OPENAI_API_KEY", ""))

SYSTEM = """You extract durable, user-specific memories from one conversation turn \
for a long-term memory system. Extract ONLY information about the USER — never the \
assistant's words, never general world facts, never greetings or small talk.

For each memory output:
- type: one of
  - "fact"       stable personal facts: employer, job title, location, hometown, \
family, pets, dietary, health, languages
  - "preference" what the user likes/wants (in answers or in general)
  - "opinion"    the user's subjective stance on some topic
  - "event"      something the user did or that happened, tied to a time
- key: a SHORT canonical snake_case topic id, STABLE across phrasings, so that two \
statements about the same topic produce the SAME key. Reuse a small vocabulary, e.g.:
    employer, job_title, location, hometown, pet_name, dietary, family, \
communication_style
  For opinions/preferences use "opinion:<topic>" / "preference:<topic>" \
(e.g. "opinion:typescript"). For events use "event:<short_slug>".
- value: a concise, self-contained statement of the memory (readable on its own).
- confidence: 0.0–1.0 — explicit statement = high (~0.95); inferred/implicit = lower (~0.7).

Rules:
- Resolve IMPLICIT facts. "walking Biscuit this morning" -> \
{type:"fact", key:"pet_name", value:"Has a pet named Biscuit", confidence:0.8}.
- A mention of moving to / relocating to / living in / settling in a place is a \
LOCATION fact, even when the sentence is phrased as an aside or opinion: "the best \
part of moving to Berlin was the parks" -> \
{type:"fact", key:"location", value:"Lives in Berlin", confidence:0.8} (you may also \
extract the separate opinion about the parks).
- Handle CORRECTIONS. "actually I meant X, not Y" -> extract the corrected value X.
- One memory per distinct fact; do not duplicate.
- If there is nothing memorable about the user, return an empty list."""

# Strict JSON schema — gpt-4o-mini honors response_format json_schema. If you ever
# point OPENAI_BASE_URL at an endpoint without json_schema support, switch to
# response_format={"type": "json_object"} and keep the schema described in SYSTEM.
_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": ["fact", "preference", "opinion", "event"]},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["type", "key", "value", "confidence"],
            },
        }
    },
    "required": ["memories"],
}


def _format(messages) -> str:
    """Render the turn as plain text for the extractor."""
    lines = []
    for m in messages:
        role = m.get("role", "?")
        name = m.get("name")
        prefix = role + (f"/{name}" if name else "")
        lines.append(f"{prefix}: {m.get('content', '')}")
    return "\n".join(lines)


def extract_memories(messages):
    """messages: list of {role, content, name?}. Returns a list of
    {type, key, value, confidence} dicts. Never raises — returns [] on failure."""
    convo = _format(messages)
    try:
        resp = _client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "memories", "strict": True, "schema": _SCHEMA},
            },
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": convo},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return data.get("memories", [])
    except Exception as e:
        print(f"[extraction error: {e}]")
        return []


def embed(text):
    """Embed one string. Returns list[float] (len 1536 for text-embedding-3-small)
    or None on failure."""
    try:
        r = _client.embeddings.create(model=EMBED_MODEL, input=text)
        return r.data[0].embedding
    except Exception as e:
        print(f"[embed error: {e}]")
        return None


# Quick manual check: `OPENAI_API_KEY=... python extract.py` (hits the API).
if __name__ == "__main__":
    sample = [
        {"role": "user", "content": "Sorry I'm late, was out walking Biscuit. Btw I just started at Notion as a PM."},
        {"role": "assistant", "content": "Congrats on Notion!"},
    ]
    for mem in extract_memories(sample):
        print(mem)
