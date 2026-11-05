"""Structured shared state passed between agents by the orchestrator."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Any

AGENTS = ("intake", "tracking", "claim", "communication")
AGENT_LABELS = {
    "intake": "Baggage Intake Agent",
    "tracking": "Baggage Tracking Agent",
    "claim": "Claim Agent",
    "communication": "Communication Agent",
}
TERMINAL_STAGES = {"completed", "escalated"}
REQUIRED_FIELDS = ("passenger_name", "booking_reference", "flight_number", "flight_date",
                   "baggage_tag", "email")
FIELD_LABELS = {
    "passenger_name": "passenger name",
    "booking_reference": "booking reference",
    "flight_number": "flight number",
    "flight_date": "flight date",
    "baggage_tag": "baggage tag number",
    "email": "email address",
}


@dataclass
class CaseState:
    """Everything the agents know about one baggage case. One instance per conversation case."""

    # --- conversation -----------------------------------------------------
    user_message: str = ""
    messages: list[dict] = field(default_factory=list)      # [{"role","content"}]
    reply: str = ""                                          # assistant answer for the current turn
    intent: str = ""                                         # baggage_issue | policy_question | other
    stage: str = "idle"    # idle | collecting | verifying | processing | completed | escalated | error
    language: str = "en"                                      # "ar" | "en" - detected from the passenger's own text
    # --- intake -----------------------------------------------------------
    intake: dict = field(default_factory=dict)               # validated passenger/flight/baggage inputs
    missing_fields: list[str] = field(default_factory=list)
    intake_complete: bool = False
    last_asked: list[str] = field(default_factory=list)
    suggested_tag: str | None = None
    tag_failures: int = 0
    # --- retrieved data (ONLY ever filled from tool results) ---------------
    booking: dict | None = None
    flight: dict | None = None
    baggage: dict | None = None
    tracking_outcome: str | None = None                      # ok | verify | escalate | error
    existing_claim: dict | None = None
    # --- decisions / actions ----------------------------------------------
    decision: dict | None = None
    analysis: str = ""
    claim: dict | None = None
    claim_created: bool = False
    follow_up: dict | None = None
    email: dict | None = None
    summary: dict | None = None
    summary_text: str = ""
    report_path: str | None = None
    policy_answer: dict | None = None
    # --- observability ----------------------------------------------------
    agent_status: dict = field(default_factory=lambda: {a: "pending" for a in AGENTS})
    trace: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    llm_events: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    halt: bool = False
    last_run_ms: float = 0.0

    # ----------------------------------------------------------------------
    def log(self, agent: str, event: str, status: str = "ok", detail: str = "") -> None:
        self.trace.append({"t": time.strftime("%H:%M:%S"), "agent": agent, "event": event,
                           "status": status, "detail": detail})

    def set_agent(self, agent: str, status: str) -> None:
        self.agent_status[agent] = status

    def reset_case(self) -> None:
        """Start a fresh case but keep the chat history."""
        fresh = CaseState(messages=self.messages)
        for f in fields(self):
            setattr(self, f.name, getattr(fresh, f.name))

    def begin_turn(self, message: str) -> None:
        self.user_message = message
        self.reply = ""
        self.intent = ""
        self.halt = False
        self.errors = []
        if self.stage in ("collecting", "verifying", "error"):
            for a in ("tracking", "claim", "communication"):
                self.agent_status[a] = "pending"

    @property
    def facts_ids(self) -> set[str]:
        """Identifiers that came from tools/user data - the only ones an LLM may mention."""
        ids: set[str] = set()
        for src in (self.intake, self.baggage or {}, self.booking or {}, self.claim or {},
                    self.existing_claim or {}):
            for key in ("baggage_tag", "booking_reference", "flight_number", "claim_id"):
                if src.get(key):
                    ids.add(str(src[key]))
        return ids

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
