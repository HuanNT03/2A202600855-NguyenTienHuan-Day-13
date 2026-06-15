"""YOUR mitigation + observability layer. The simulator calls mitigate() around the
opaque agent (a REAL LLM) for every request. This is the ONLY place observability can
live -- the agent is silent. Legal moves: retry / cache / route / guardrail / sanitize
/ fallback / session-reset / PROMPT ROUTING, plus your own logging/tracing/metrics.
Illegal: hardcoding answers, importing the agent internals, reading instructor files,
network exfiltration.

  call_next(question, config) -> result   # the only way to reach the black box
  context = {"session_id","turn_index","qid","cache": <shared dict>, "cache_lock": <Lock>}
  result  = {"answer","status","steps","trace","meta":{latency_ms,usage,...}}

PROMPT ROUTING: you can override the agent's system prompt PER REQUEST by setting it in
the config you pass to call_next, e.g.:
    conf = dict(config); conf["system_prompt"] = my_better_prompt
    result = call_next(question, conf)
(Or just edit solution/prompt.txt for a single static prompt used on every request.)
"""
from __future__ import annotations

import sys
import os
import re

# Ensure that the simulator's embedded Python interpreter can import packages from the local virtualenv
_wrapper_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_wrapper_dir)

# 1. Add local virtual environment site-packages
_venv_lib_dir = os.path.join(_project_dir, ".venv", "lib")
if os.path.isdir(_venv_lib_dir):
    for _subdir in os.listdir(_venv_lib_dir):
        if _subdir.startswith("python"):
            _site_packages = os.path.join(_venv_lib_dir, _subdir, "site-packages")
            if os.path.isdir(_site_packages) and _site_packages not in sys.path:
                sys.path.insert(0, _site_packages)

# 2. Add Python 3.12 standard library paths (PURE PYTHON ONLY, NO lib-dynload to prevent segfaults)
_std_libs = [
    "/usr/lib/python3.12",
]
_home = os.path.expanduser("~")
_pyenv_version_file = os.path.join(_project_dir, ".python-version")
if os.path.isfile(_pyenv_version_file):
    try:
        with open(_pyenv_version_file, "r") as f:
            _ver = f.read().strip()
        _std_libs.append(os.path.join(_home, ".pyenv", "versions", _ver, "lib", "python3.12"))
    except Exception:
        pass

for _lib in _std_libs:
    if os.path.isdir(_lib) and _lib not in sys.path:
        sys.path.append(_lib)

# You may reuse the Day 13 toolkit, e.g.:
# from telemetry.logger import logger
# from telemetry.cost import cost_from_usage
# from telemetry.redact import redact


def mitigate(call_next, question, config, context):
    # TODO: add observability here (log latency, tokens, cost, errors, PII, tool counts).
    # TODO: add mitigations (retry on error, cache repeats, route cheap, reset drifting
    #       sessions, validate arithmetic, sanitize order notes, redact PII...).
    # TODO: optionally route a better system prompt:
    #       conf = dict(config); conf["system_prompt"] = "..."; return call_next(question, conf)

    import time
    from telemetry.logger import logger, set_correlation_id, new_correlation_id
    from telemetry.cost import cost_from_usage
    from telemetry.redact import redact

    # Sanitize prompt injection by removing lines starting with GHI CHU KHACH / GHI CHU / NOTE
    cleaned_question = re.sub(r"(?im)^(ghi\s*chú\s*khách|ghi\s*chu\s*khách|ghi\s*chú|ghi\s*chu|note|notes)\s*:.*$", "", question)

    # 1. Setup Correlation ID for trace linking
    qid = context.get("qid", "unknown")
    session_id = context.get("session_id", "unknown")
    turn_index = context.get("turn_index", 0)
    
    cid = f"{session_id}-{turn_index}-{qid}"
    set_correlation_id(cid)

    # Print incoming request info for real-time terminal monitoring
    print(f"\n>>> [Wrapper] QID: {qid} | Session: {session_id} | Turn: {turn_index}", file=sys.stderr)
    print(f"    Question: {cleaned_question.strip()}", file=sys.stderr)

    # 2. Cache Lookup (Thread-safe)
    cache = context.get("cache")
    cache_lock = context.get("cache_lock")
    
    if cache is not None:
        cache_key = cleaned_question.strip()
        if cache_key in cache:
            print(f"    [Cache HIT] Returning cached response", file=sys.stderr)
            return cache[cache_key]

    t0 = time.time()

    # 3. Execute request via call_next with exception handling
    try:
        result = call_next(cleaned_question, config)
        status = result.get("status", "unknown")
    except Exception as e:
        print(f"    [Wrapper ERROR] Exception during call_next: {e}", file=sys.stderr)
        if logger:
            logger.log_event("AGENT_EXCEPTION", {
                "qid": qid,
                "session_id": session_id,
                "turn_index": turn_index,
                "error": str(e)
            })
        raise e

    wall_ms = int((time.time() - t0) * 1000)

    # 4. Extract response details
    meta = result.get("meta", {}) or {}
    usage = meta.get("usage", {}) or {}
    model = meta.get("model", "")
    tools_used = meta.get("tools_used", [])
    latency_ms = meta.get("latency_ms", 0)
    steps = result.get("steps", 0)
    answer = result.get("answer", "")
    
    # Calculate token cost
    cost_usd = cost_from_usage(model, usage)

    # 5. Check and redact PII in output
    redacted_answer, num_redactions = redact(answer)
    if num_redactions > 0:
        print(f"    [PII REDACTED] Masked {num_redactions} PII items in answer.", file=sys.stderr)
        result["answer"] = redacted_answer

    # Print debug metrics to console
    print(f"    [Wrapper Response] Status: {status} | Latency: {latency_ms}ms (Wall: {wall_ms}ms)", file=sys.stderr)
    print(f"    Cost: ${cost_usd:.6f} | Steps: {steps} | Tools: {tools_used}", file=sys.stderr)
    if status != "ok":
        print(f"    [WARNING] Status is NOT 'ok'. Result status: {status}", file=sys.stderr)

    # 6. Log event to logs/YYYY-MM-DD.log
    if logger:
        logger.log_event("AGENT_CALL_METRICS", {
            "qid": qid,
            "session_id": session_id,
            "turn_index": turn_index,
            "status": status,
            "latency_ms": latency_ms,
            "wall_ms": wall_ms,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cost_usd": cost_usd,
            "tools_used": tools_used,
            "steps": steps,
            "pii_redacted_count": num_redactions
        })

    # 7. Write to cache (Thread-safe)
    if cache is not None and cache_lock is not None:
        with cache_lock:
            cache[cleaned_question.strip()] = result

    return result
