"""Contract & robustness tests against a live service.

    docker compose up -d
    MEMORY_URL=http://localhost:8888 python tests/test_contract.py

Covers the four required cases: contract roundtrip, concurrent-session isolation,
malformed input (4xx, no crash), and restart persistence. The restart test shells out
to `docker restart` (set RESTART_CONTAINER; it's skipped if docker isn't reachable).
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

URL = os.getenv("MEMORY_URL", "http://localhost:8888").rstrip("/")
TOKEN = os.getenv("MEMORY_AUTH_TOKEN", "")
RESTART_CONTAINER = os.getenv("RESTART_CONTAINER", "memory-service-app-1")


def _headers():
    h = {"Content-Type": "application/json"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    return h


def req(method, path, body=None, raw_body=None):
    """Return (status, parsed_body). status 0 = unreachable. Never raises on 4xx/5xx."""
    data = raw_body if raw_body is not None else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request(URL + path, data=data, method=method, headers=_headers())
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            b = resp.read()
            return resp.status, (json.loads(b) if b else None)
    except urllib.error.HTTPError as e:
        b = e.read()
        try:
            return e.code, (json.loads(b) if b else None)
        except Exception:
            return e.code, None
    except OSError:
        # connection refused/reset/timeout — e.g. while the service restarts
        return 0, None


def turn_body(session, user, content, ts="2025-03-15T10:30:00Z"):
    return {"session_id": session, "user_id": user,
            "messages": [{"role": "user", "content": content},
                         {"role": "assistant", "content": "ok"}],
            "timestamp": ts, "metadata": {}}


def recall_body(query, user, session="probe"):
    return {"query": query, "session_id": session, "user_id": user, "max_tokens": 512}


results = []


def check(name, cond):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def test_roundtrip():
    print("=== contract roundtrip ===")
    u = "ct-roundtrip"
    req("DELETE", f"/users/{u}")
    s, b = req("POST", "/turns", turn_body("ct-s1", u, "I live in Lisbon and love sailing."))
    check("POST /turns -> 201", s == 201)
    check("POST /turns returns {id}", isinstance(b, dict) and "id" in b)
    s, b = req("POST", "/recall", recall_body("Where does the user live?", u))
    b = b or {}
    check("POST /recall -> 200", s == 200)
    check("recall shape {context, citations[]}", "context" in b and isinstance(b.get("citations"), list))
    check("recall surfaces Lisbon", "lisbon" in b.get("context", "").lower())
    s, b = req("GET", f"/users/{u}/memories")
    b = b or {}
    check("GET memories -> 200", s == 200)
    check("memories is a list", isinstance(b.get("memories"), list))
    if b.get("memories"):
        m = b["memories"][0]
        check("memory has type/key/value", all(k in m for k in ("type", "key", "value")))
    req("DELETE", f"/users/{u}")


def test_isolation():
    print("=== concurrent sessions (no bleed) ===")
    a, bu = "ct-userA", "ct-userB"
    req("DELETE", f"/users/{a}")
    req("DELETE", f"/users/{bu}")
    req("POST", "/turns", turn_body("a-s1", a, "I live in Paris."))
    req("POST", "/turns", turn_body("b-s1", bu, "I live in Tokyo."))
    _, ra = req("POST", "/recall", recall_body("Where does the user live?", a))
    _, rb = req("POST", "/recall", recall_body("Where does the user live?", bu))
    ca, cb = (ra or {}).get("context", "").lower(), (rb or {}).get("context", "").lower()
    check("user A sees Paris, not Tokyo", "paris" in ca and "tokyo" not in ca)
    check("user B sees Tokyo, not Paris", "tokyo" in cb and "paris" not in cb)
    req("DELETE", f"/users/{a}")
    req("DELETE", f"/users/{bu}")


def test_malformed():
    print("=== malformed input (4xx, no crash) ===")
    s, _ = req("POST", "/turns", raw_body=b"{not valid json")
    check("bad JSON -> 4xx", 400 <= s < 500)
    s, _ = req("POST", "/turns", body={"session_id": "x"})  # missing messages/timestamp
    check("missing fields -> 422", s == 422)
    u = "ct-unicode"
    req("DELETE", f"/users/{u}")
    s, _ = req("POST", "/turns", turn_body("u-s1", u, "Я живу в Москве 🐉 — 日本語 — Ⅷ ﷽"))
    check("unicode turn -> 201", s == 201)
    req("DELETE", f"/users/{u}")
    s, _ = req("GET", "/health")
    check("service alive after malformed input", s == 200)


def test_restart():
    print("=== restart persistence ===")
    u = "ct-persist"
    req("DELETE", f"/users/{u}")
    req("POST", "/turns", turn_body("p-s1", u, "I work at Acme Corp."))
    try:
        subprocess.run(["docker", "restart", RESTART_CONTAINER],
                       check=True, capture_output=True, timeout=60)
    except Exception as e:
        print(f"  [SKIP] could not restart '{RESTART_CONTAINER}' ({e}); set RESTART_CONTAINER")
        req("DELETE", f"/users/{u}")
        return
    for _ in range(60):  # wait for health to come back
        if req("GET", "/health")[0] == 200:
            break
        time.sleep(1)
    _, b = req("POST", "/recall", recall_body("Where does the user work?", u))
    check("fact survives restart", "acme" in (b or {}).get("context", "").lower())
    req("DELETE", f"/users/{u}")


if __name__ == "__main__":
    if req("GET", "/health")[0] == 0:
        print(f"Service not reachable at {URL}. Is it up?")
        sys.exit(1)
    test_roundtrip()
    test_isolation()
    test_malformed()
    test_restart()
    passed = sum(results)
    print(f"\nContract: {passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)
