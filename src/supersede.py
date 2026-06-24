"""Supersession: when a new fact contradicts an existing one on the same topic
(`key`), mark the old one inactive (kept for history, NOT deleted) instead of
leaving both active. /recall then returns only the current fact.

Hybrid decision:
  - a RULE for clearly single-valued keys (employer, location, ...) — cheap, no LLM
  - an LLM JUDGE for everything else — general, handles "are these independent?"
"""
import json
import os

from openai import OpenAI

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_client = OpenAI(base_url=OPENAI_BASE_URL, api_key=os.getenv("OPENAI_API_KEY", ""))

# Keys that hold exactly ONE current value — a new value replaces the old.
# Tune this set as the eval reveals gaps; it's a core design lever.
SINGLE_VALUED = {
    "employer", "job_title", "location", "hometown",
    "dietary", "marital_status", "relationship_status",
}

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"superseded": {"type": "array", "items": {"type": "integer"}}},
    "required": ["superseded"],
}


def resolve_supersession(new_mem, candidates):
    """Return ids of existing active memories that `new_mem` makes stale.
    candidates: active memories sharing the key (dicts with id, type, key, value)."""
    if not candidates:
        return []
    if new_mem.get("type") == "event":          # events accumulate, never supersede
        return []
    if new_mem.get("key", "") in SINGLE_VALUED:  # rule: latest value wins
        return [c["id"] for c in candidates]
    return _llm_judge(new_mem, candidates)       # ambiguous key -> ask the model


def _llm_judge(new_mem, candidates):
    listing = "\n".join(f"{i}: {c['value']}" for i, c in enumerate(candidates))
    prompt = (
        f"A new memory about a user was recorded:\n  NEW: {new_mem['value']}\n\n"
        f"Existing memories with the same topic key '{new_mem.get('key', '')}':\n{listing}\n\n"
        "Return the indices of existing memories that the NEW one makes outdated or "
        "incorrect (it replaces them). If the new memory is independent and the existing "
        "ones are still true, return an empty list."
    )
    try:
        resp = _client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "supersede", "strict": True, "schema": _SCHEMA},
            },
            messages=[{"role": "user", "content": prompt}],
        )
        idxs = json.loads(resp.choices[0].message.content or "{}").get("superseded", [])
        return [candidates[i]["id"] for i in idxs if 0 <= i < len(candidates)]
    except Exception as e:
        print(f"[supersede error: {e}]")
        return []  # on failure keep both — safe, no data loss
