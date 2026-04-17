"""End-to-end proxy test suite using httpx as the HTTP client.

Tests every feature + the new dead-subprocess recovery path.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid

import httpx

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 4108
BASE = f"http://127.0.0.1:{PORT}"
MODEL = "claude-haiku-4-5"
TOOL_MODEL = "claude-sonnet-4-6"  # more reliable tool-calling for E2E assertions

# UUID5 namespace — must match middleware._SESSION_NAMESPACE
NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# Fresh session prefix per test run so we don't resume state from prior runs.
RUN = str(int(time.time()))


def H(name: str) -> dict:
    return {"x-session-id": f"{name}-{RUN}"}


def sid5(header_value: str) -> str:
    return str(uuid.uuid5(NS, header_value))


def kill_session_subproc(header_value: str) -> bool:
    """Kill the CLI subprocess whose --session-id/--resume matches this header."""
    internal = sid5(header_value)
    pids: list[int] = []
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", f"claude -p --input-format.*{internal}"],
            text=True,
        )
        pids = [int(p) for p in out.split()]
    except subprocess.CalledProcessError:
        return False
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # Wait long enough for the proxy's asyncio loop to notice the SIGCHLD.
    time.sleep(3)
    return bool(pids)


passed = 0
failed = 0


def check(cond: bool, name: str, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")


def post(path: str, payload: dict, headers: dict | None = None, timeout: float = 45) -> dict:
    h = {"content-type": "application/json"}
    if headers:
        h.update(headers)
    r = httpx.post(BASE + path, json=payload, headers=h, timeout=timeout)
    try:
        return r.json()
    except Exception:
        return {"_raw": r.text, "_status": r.status_code}


def stream(path: str, payload: dict, headers: dict, timeout: float = 45) -> str:
    with httpx.stream("POST", BASE + path, json=payload, headers=headers, timeout=timeout) as r:
        return "".join(r.iter_text())


def content(resp: dict) -> str:
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return json.dumps(resp)[:200]


def finish(resp: dict) -> str:
    try:
        return resp["choices"][0]["finish_reason"]
    except (KeyError, IndexError, TypeError):
        return "err"


def tool_calls(resp: dict) -> list[dict]:
    try:
        return resp["choices"][0]["message"].get("tool_calls") or []
    except (KeyError, IndexError, TypeError):
        return []


# --- T1: basic chat ---
print("=== T1 basic chat ===")
r = post("/v1/chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Reply OK only"}],
}, H("T1"))
check("OK" in content(r), "T1 basic", f"got {content(r)[:100]}")

# --- T2: session memory ---
print("=== T2 session memory ===")
post("/v1/chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Remember MANGO. Reply OK."}],
}, H("T2"))
r = post("/v1/chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What word did I say? One word."}],
}, H("T2"))
check("MANGO" in content(r), "T2 memory", f"got {content(r)[:100]}")

# --- T3: tool roundtrip ---
print("=== T3 tool roundtrip ===")
tool_def = {"type": "function", "function": {"name": "add", "description": "Add two integers.",
    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}
T3_SYS = "You MUST call the add tool. Never compute yourself."
r1 = post("/v1/chat/completions", {
    "model": TOOL_MODEL,
    "messages": [
        {"role": "system", "content": T3_SYS},
        {"role": "user", "content": "Please use the add tool with a=2 b=3"},
    ],
    "tools": [tool_def],
}, H("T3"))
check(finish(r1) == "tool_calls", "T3a tool_calls emitted", f"got finish={finish(r1)}")
if finish(r1) == "tool_calls":
    tc = tool_calls(r1)[0]
    tcid = tc["id"]
    r2 = post("/v1/chat/completions", {
        "model": TOOL_MODEL,
        "messages": [
            {"role": "system", "content": T3_SYS},
            {"role": "user", "content": "Please use the add tool with a=2 b=3"},
            {"role": "assistant", "tool_calls": [{"id": tcid, "type": "function",
                "function": {"name": "add", "arguments": '{"a":2,"b":3}'}}]},
            {"role": "tool", "tool_call_id": tcid, "content": "5"},
        ],
        "tools": [tool_def],
    }, H("T3"))
    check(finish(r2) == "stop", "T3b continuation finishes", f"got {finish(r2)}")
    check("5" in content(r2), "T3c answer has 5", f"got {content(r2)[:100]}")

# --- T4: streaming ---
print("=== T4 streaming ===")
s = stream("/v1/chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Say STREAMX only"}],
    "stream": True,
}, H("T4"))
check("STREAMX" in s, "T4 stream content")
check("[DONE]" in s, "T4 stream [DONE]")

# --- T5: stateless multi-turn via history ---
print("=== T5 stateless multi-turn ===")
r = post("/v1/chat/completions", {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Remember KIWI."},
        {"role": "assistant", "content": "OK, remembered KIWI."},
        {"role": "user", "content": "Respond with just the word I asked you to remember."},
    ],
})
check("KIWI" in content(r), "T5 stateless history", f"got {content(r)[:100]}")

# --- T6: CORE FIX — subprocess killed mid-session, --resume recovers ---
print("=== T6 dead subprocess + --resume ===")
post("/v1/chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Remember DOLPHIN. Reply OK."}],
}, H("T6"))
killed = kill_session_subproc(f"T6-{RUN}")
check(killed, "T6 subprocess was killed")
r = post("/v1/chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What word did I tell you to remember? One word."}],
}, H("T6"))
check("DOLPHIN" in content(r), "T6 respawn resumes history", f"got {content(r)[:100]}")

# --- T7: streaming after respawn ---
print("=== T7 streaming after respawn ===")
post("/v1/chat/completions", {
    "model": MODEL, "messages": [{"role": "user", "content": "say OK"}],
}, H("T7"))
kill_session_subproc(f"T7-{RUN}")
s = stream("/v1/chat/completions", {
    "model": MODEL, "messages": [{"role": "user", "content": "Say STREAMRESUMED only"}],
    "stream": True,
}, H("T7"))
check("STREAMRESUMED" in s, "T7 streaming after respawn")

# --- T8: tool use after respawn ---
print("=== T8 tool use after respawn ===")
post("/v1/chat/completions", {
    "model": MODEL, "messages": [{"role": "user", "content": "say hi"}],
}, H("T8"))
kill_session_subproc(f"T8-{RUN}")
tool_def_strong = {"type": "function", "function": {"name": "add", "description": "Add.",
    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}
r = post("/v1/chat/completions", {
    "model": TOOL_MODEL,
    "messages": [
        {"role": "system", "content": "You MUST call the add tool. Do not compute yourself."},
        {"role": "user", "content": "Use the add tool with a=3 b=4"},
    ],
    "tools": [tool_def_strong],
}, H("T8"))
check(finish(r) == "tool_calls", "T8 tool_calls after respawn", f"got {finish(r)}")

# --- T9: orphan tool_result — respawn mid-tool-cycle errors cleanly ---
print("=== T9 orphan tool_result ===")
r1 = post("/v1/chat/completions", {
    "model": TOOL_MODEL,
    "messages": [
        {"role": "system", "content": "You MUST call the add tool."},
        {"role": "user", "content": "Use the add tool with a=1 b=1"},
    ],
    "tools": [tool_def_strong],
}, H("T9"))
if finish(r1) == "tool_calls":
    tcid = tool_calls(r1)[0]["id"]
    kill_session_subproc(f"T9-{RUN}")
    r2 = post("/v1/chat/completions", {
        "model": TOOL_MODEL,
        "messages": [
            {"role": "user", "content": "Use the add tool with a=1 b=1"},
            {"role": "assistant", "tool_calls": [{"id": tcid, "type": "function",
                "function": {"name": "add", "arguments": '{}'}}]},
            {"role": "tool", "tool_call_id": tcid, "content": "2"},
        ],
        "tools": [tool_def_strong],
    }, H("T9"), timeout=20)
    raw = json.dumps(r2)
    check("respawned mid-cycle" in raw, "T9 clean error on orphan tool_result",
          f"got {raw[:200]}")
else:
    print(f"  SKIP T9 — first turn didn't emit tool_calls (finish={finish(r1)})")

# --- T10: stateless rejects tools ---
print("=== T10 stateless+tools rejected ===")
r = post("/v1/chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "x"}],
    "tools": [{"type": "function", "function": {"name": "x",
        "parameters": {"type": "object"}}}],
}, timeout=15)
raw = json.dumps(r)
check("requires a session header" in raw, "T10 stateless+tools rejected",
      f"got {raw[:150]}")

# --- T13: streaming tool_calls → tool result continuation on same session ---
# This is the path OpenCode exercises: streaming request, receive tool_calls
# chunk, send tool result in a follow-up request. Uses x-session-affinity
# (OpenCode's header) to exactly match the real client. Regression guard for
# the "subprocess was respawned mid-cycle" bug where GeneratorExit during
# stream teardown wrongly marked the session dead.
print("=== T13 streaming tool_calls → continuation (OpenCode-shape) ===")
T13_SYS = "You MUST call the add tool. Never compute yourself."
T13_HEADER = {"x-session-affinity": f"T13-oc-{RUN}"}
s = stream("/v1/chat/completions", {
    "model": TOOL_MODEL,
    "messages": [
        {"role": "system", "content": T13_SYS},
        {"role": "user", "content": "Use the add tool with a=4 b=6"},
    ],
    "tools": [tool_def],
    "stream": True,
}, {**T13_HEADER, "content-type": "application/json"})
check("tool_calls" in s, "T13a streaming emits tool_calls")
# Extract the call_id from the streamed SSE
m = re.search(r'"id":"(call_[a-f0-9]+)"', s)
if m:
    tcid13 = m.group(1)
    r = post("/v1/chat/completions", {
        "model": TOOL_MODEL,
        "messages": [
            {"role": "system", "content": T13_SYS},
            {"role": "user", "content": "Use the add tool with a=4 b=6"},
            {"role": "assistant", "tool_calls": [{"id": tcid13, "type": "function",
                "function": {"name": "add", "arguments": '{"a":4,"b":6}'}}]},
            {"role": "tool", "tool_call_id": tcid13, "content": "10"},
        ],
        "tools": [tool_def],
    }, T13_HEADER)
    check(finish(r) == "stop", "T13b continuation after streamed tool_calls",
          f"got finish={finish(r)}")
    check("10" in content(r), "T13c answer has 10", f"got {content(r)[:100]}")
else:
    check(False, "T13 extract call_id from stream", "no call_ id found")


# --- T11: client disconnect then reconnect — session must be reusable ---
print("=== T11 client disconnect then reconnect ===")
post("/v1/chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "reply OK"}],
}, H("T11"))  # session warm
try:
    with httpx.stream("POST", f"{BASE}/v1/chat/completions",
                       headers={**H("T11"), "content-type": "application/json"},
                       timeout=1.5, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count slowly 1 to 100"}],
        "stream": True,
    }) as r:
        for _ in r.iter_bytes():
            pass  # will ReadTimeout mid-stream
except (httpx.ReadTimeout, httpx.RemoteProtocolError):
    pass
time.sleep(2)
r = post("/v1/chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "say RECOVERED only"}],
}, H("T11"), timeout=30)
check("RECOVERED" in content(r), "T11 session reusable after disconnect",
      f"got {content(r)[:100]}")

# --- T12: OpenCode-shape request (x-session-affinity + many tools + big system) ---
print("=== T12 OpenCode-shape hello ===")
opencode_tools = [
    {"type": "function", "function": {
        "name": nm,
        "description": f"{nm} tool. " * 8,
        "parameters": {"type": "object",
            "properties": {"arg": {"type": "string"}}, "required": ["arg"]},
    }}
    for nm in ["read", "write", "edit", "bash", "grep", "glob",
               "list_directory", "multi_edit", "web_fetch", "task", "todo_write"]
]
opencode_sys = "You are a coding assistant.\n" + ("Detailed instructions.\n" * 100)
r = post("/v1/chat/completions", {
    "model": "claude-sonnet-4-6",
    "messages": [
        {"role": "system", "content": opencode_sys},
        {"role": "user", "content": "hello"},
    ],
    "tools": opencode_tools,
}, {"x-session-affinity": f"opencode-test-{RUN}"}, timeout=60)
check(finish(r) in {"stop", "tool_calls"}, "T12 OpenCode hello succeeded",
      f"got {finish(r)}")

print()
print(f"=== {passed} passed, {failed} failed ===")
sys.exit(0 if failed == 0 else 1)
