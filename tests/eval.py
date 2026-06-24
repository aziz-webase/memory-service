"""Recall-quality self-eval. Ingests fixtures/eval.json, probes /recall, reports
how many expected facts surfaced. This is the iteration loop — run it after every
change.

Usage:
    docker compose up -d
    MEMORY_URL=http://localhost:8888 python tests/eval.py   # default: :8888
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = os.getenv("MEMORY_URL", "http://localhost:8888").rstrip("/")
TOKEN = os.getenv("MEMORY_AUTH_TOKEN", "")
FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "eval.json"


def _headers():
    h = {"Content-Type": "application/json"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    return h


def _request(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if raw else None)
    except urllib.error.URLError as e:
        print(f"ERROR: cannot reach {URL} ({e}). Is the service up?")
        sys.exit(1)


def main():
    data = json.loads(FIXTURE.read_text())
    found_facts = total_facts = 0
    probes_passed = probes_total = 0

    for sc in data["scenarios"]:
        uid = sc["user_id"]
        _request("DELETE", f"/users/{uid}")  # clean slate per scenario

        for turn in sc["turns"]:
            status, _ = _request("POST", "/turns", {
                "session_id": turn["session_id"],
                "user_id": uid,
                "messages": turn["messages"],
                "timestamp": turn["timestamp"],
                "metadata": {},
            })
            if status != 201:
                print(f"  WARN: /turns returned {status} in scenario '{sc['name']}'")

        print(f"=== {sc['name']} ===")
        for p in sc["probes"]:
            status, resp = _request("POST", "/recall", {
                "query": p["query"],
                "session_id": p["session_id"],
                "user_id": uid,
                "max_tokens": 512,
            })
            context = (resp or {}).get("context", "") if status == 200 else ""
            ctx = context.lower()
            probes_total += 1

            if p.get("expect_empty"):
                ok = not context.strip()
                probes_passed += int(ok)
                detail = "empty as expected" if ok else "expected empty, got text"
            else:
                expect = p.get("expect", [])
                hits = [f for f in expect if f.lower() in ctx]
                total_facts += len(expect)
                found_facts += len(hits)
                ok = len(hits) == len(expect)
                probes_passed += int(ok)
                detail = f"found {hits}" if ok else f"missing {[f for f in expect if f.lower() not in ctx]}"

            print(f"  [{'PASS' if ok else 'FAIL'}] {p['query']}  ({detail})")

    print()
    print(f"Facts recalled: {found_facts}/{total_facts}")
    print(f"Probes passed:  {probes_passed}/{probes_total}")
    # Non-zero exit if nothing recalled, so CI / scripts can gate on it.
    sys.exit(0 if probes_passed == probes_total else 1)


if __name__ == "__main__":
    main()
