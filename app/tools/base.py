"""Shared tool plumbing: a uniform ``ToolResult`` and an instrumented ``call_tool``.

Every tool returns a ToolResult and NEVER raises; failures are values
(``ok=False`` + ``error_code``) so the orchestrator can branch on them.
Every call is logged into the case trace, which powers the UI's "Agent Trace"
tab and the evaluation metrics (tool-call success rate, latency).
"""
from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.database.db import DatabaseError

log = logging.getLogger(__name__)
_REDACT = {"contact_email", "recipient", "body", "email", "case_summary"}


@dataclass
class ToolResult:
    ok: bool
    data: dict | None = None
    error_code: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "data": self.data, "error_code": self.error_code, "message": self.message}


def safe_tool(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
    """Convert unexpected exceptions into ToolResult failures."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> ToolResult:
        try:
            return fn(*args, **kwargs)
        except DatabaseError as exc:
            log.error("database error in %s: %s", fn.__name__, exc)
            return ToolResult(False, error_code="DB_ERROR",
                              message="The baggage database is temporarily unavailable.")
        except Exception as exc:  # noqa: BLE001 - last line of defence
            log.exception("unexpected error in tool %s", fn.__name__)
            return ToolResult(False, error_code="TOOL_EXCEPTION", message=f"{type(exc).__name__}: {exc}")
    return wrapper


def call_tool(case, agent: str, fn: Callable[..., ToolResult], *args, **kwargs) -> ToolResult:
    """Invoke a tool and record name, agent, outcome and latency on the case."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    if not isinstance(result, ToolResult):  # defensive: a tool broke the contract
        result = ToolResult(False, error_code="BAD_TOOL_RESULT", message="tool returned no result")
    elapsed = (time.perf_counter() - start) * 1000
    case.tool_calls.append({
        "agent": agent,
        "tool": getattr(fn, "__name__", str(fn)),
        "args": {k: ("***" if k in _REDACT else v) for k, v in kwargs.items()},
        "ok": result.ok,
        "error_code": result.error_code,
        "ms": round(elapsed, 2),
    })
    return result
