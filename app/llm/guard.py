"""Grounding guard - the anti-hallucination layer.

Facts (status, location, claim ID) come from tools. When an LLM writes prose
around those facts, its output is checked before use:

  1. every REQUIRED fact must appear in the text, and
  2. any identifier-looking token (CLM-xxxx, BGxxxxx, BKxxxx, FLxxx) must be one
     of the ALLOWED ids taken from tool results.

If the check fails, or the LLM is unavailable, the agent's deterministic
template is used instead - so a bad model can never put a wrong fact in front
of a passenger.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

from app.llm.client import LLMClient, LLMError

_ID_TOKEN = re.compile(r"\b(?:CLM-\d+|BG\d{3,}|BK\d{3,}|FL\d{2,4})\b", re.I)


def check_grounding(text: str, required: Iterable[str], allowed_ids: Iterable[str]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not text or not text.strip():
        return False, ["empty output"]
    if len(text) > 3000:
        problems.append("output too long")
    low = text.lower()
    for fact in required:
        if fact and str(fact).lower() not in low:
            problems.append(f"missing fact: {fact}")
    allowed = {str(a).upper() for a in allowed_ids if a}
    for token in _ID_TOKEN.findall(text):
        if token.upper() not in allowed:
            problems.append(f"unexpected identifier: {token}")
    return not problems, problems


def grounded_generate(llm: LLMClient | None, system: str, prompt: str, *, required: Iterable[str],
                      allowed_ids: Iterable[str], fallback: Callable[[], str], case=None,
                      agent: str = "", label: str = "text") -> tuple[str, str]:
    """Return ``(text, source)`` where source is llm | rules | fallback:llm_error | fallback:guard."""
    if llm is None or not llm.enabled:
        return fallback(), "rules"
    try:
        text = llm.complete(system, prompt).strip()
    except LLMError as exc:
        _note(case, agent, f"{label}: LLM unavailable ({exc}) - deterministic fallback used")
        return fallback(), "fallback:llm_error"
    ok, problems = check_grounding(text, required, allowed_ids)
    if not ok:
        _note(case, agent, f"{label}: guard rejected LLM output ({'; '.join(problems[:3])}) - fallback used")
        return fallback(), "fallback:guard"
    return text, "llm"


def _note(case, agent: str, message: str) -> None:
    if case is not None:
        case.log(agent, "llm_fallback", "warn", message)
        case.llm_events.append({"agent": agent, "message": message})
