"""Unit tests for individual agents (Requirement 4: Multi-Agent Orchestration).

Each agent is exercised directly against a ``CaseState``, without going through
the full LangGraph orchestrator, to isolate its behaviour.
"""
from __future__ import annotations

from app.agents.claim_agent import ClaimAgent
from app.agents.communication_agent import CommunicationAgent
from app.agents.intake_agent import IntakeAgent
from app.agents.tracking_agent import TrackingAgent
from app.workflow.state import CaseState


# ----------------------------------------------------------------- Agent 1
def test_intake_agent_extracts_all_fields(services):
    agent = IntakeAgent(services.llm)
    case = CaseState()
    case.user_message = ("My baggage did not arrive. I am Ghada Mohamed, booking BK1023, "
                          "flight FL123 on 2026-09-20, baggage tag BG77821, email ghada@example.com")
    agent.run(case)
    assert case.intent == "baggage_issue"
    assert case.intake_complete is True
    assert case.intake["booking_reference"] == "BK1023"
    assert case.intake["baggage_tag"] == "BG77821"
    assert case.intake["email"] == "ghada@example.com"
    assert not case.missing_fields


def test_intake_agent_asks_for_missing_fields(services):
    agent = IntakeAgent(services.llm)
    case = CaseState()
    case.user_message = "My baggage did not arrive."
    agent.run(case)
    assert case.intake_complete is False
    assert "booking_reference" in case.missing_fields
    assert case.reply  # a follow-up question was produced
    assert case.agent_status["intake"] == "waiting"


def test_intake_agent_detects_policy_question(services):
    agent = IntakeAgent(services.llm)
    case = CaseState()
    case.user_message = "How much are interim expenses reimbursed for delayed baggage?"
    agent.run(case)
    assert case.intent == "policy_question"


def test_intake_agent_collects_incrementally_over_turns(services):
    agent = IntakeAgent(services.llm)
    case = CaseState()
    case.user_message = "My baggage did not arrive."
    agent.run(case)
    assert not case.intake_complete

    case.user_message = "I am Ghada Mohamed"
    agent.run(case)
    assert case.intake.get("passenger_name") == "Ghada Mohamed"
    assert not case.intake_complete

    case.user_message = "booking BK1023, flight FL123 on 2026-09-20, tag BG77821, email ghada@example.com"
    agent.run(case)
    assert case.intake_complete is True


# ----------------------------------------------------------------- Agent 2
def _filled_case(**overrides) -> CaseState:
    case = CaseState()
    case.intake = {
        "passenger_name": "Ghada Mohamed", "booking_reference": "BK1023", "flight_number": "FL123",
        "flight_date": "2026-09-20", "baggage_tag": "BG77821", "email": "ghada@example.com",
        "issue": "missing baggage",
    }
    case.intake.update(overrides)
    return case


def test_tracking_agent_retrieves_delayed_baggage(services):
    agent = TrackingAgent(services.db_path)
    case = _filled_case()
    agent.run(case)
    assert case.tracking_outcome == "ok"
    assert case.baggage["status"] == "Delayed"
    assert case.agent_status["tracking"] == "done"


def test_tracking_agent_flags_unknown_booking(services):
    agent = TrackingAgent(services.db_path)
    case = _filled_case(booking_reference="BK0000")
    agent.run(case)
    assert case.tracking_outcome == "verify"
    assert "booking_reference" in case.missing_fields


def test_tracking_agent_suggests_correct_tag(services):
    agent = TrackingAgent(services.db_path)
    case = _filled_case(baggage_tag="BG00000")
    agent.run(case)
    assert case.tracking_outcome == "verify"
    assert case.suggested_tag == "BG77821"


def test_tracking_agent_escalates_after_repeated_unknown_tag(services):
    agent = TrackingAgent(services.db_path)
    case = _filled_case(baggage_tag="BG00000")
    agent.run(case)          # 1st failure -> suggestion
    case.intake["baggage_tag"] = "BG11111"
    agent.run(case)          # 2nd failure -> escalate
    assert case.tracking_outcome == "escalate"


# ----------------------------------------------------------------- Agent 3
def test_claim_agent_creates_claim_for_delayed(services):
    tracking = TrackingAgent(services.db_path)
    claim_agent = ClaimAgent(services.db_path, services.llm, services.exports_dir)
    case = _filled_case()
    tracking.run(case)
    claim_agent.decide(case)
    assert case.decision["action"] == "create_claim"
    claim_agent.create_claim(case)
    assert case.claim_created is True
    assert case.claim["claim_id"].startswith("CLM-")
    assert case.claim["status"] == "Open"


def test_claim_agent_no_claim_for_delivered(services):
    tracking = TrackingAgent(services.db_path)
    claim_agent = ClaimAgent(services.db_path, services.llm, services.exports_dir)
    case = _filled_case(booking_reference="BK2045", baggage_tag="BG50412",
                         flight_number="FL210", flight_date="2026-09-19", passenger_name="Omar Hassan")
    tracking.run(case)
    claim_agent.decide(case)
    assert case.decision["action"] == "no_claim_delivered"
    assert case.claim is None


def test_claim_agent_prevents_duplicate(services):
    """BG45120 / booking BK7071 already has an active claim CLM-1001 in the fixtures."""
    tracking = TrackingAgent(services.db_path)
    claim_agent = ClaimAgent(services.db_path, services.llm, services.exports_dir)
    case = _filled_case(booking_reference="BK7071", baggage_tag="BG45120",
                         flight_number="FL123", flight_date="2026-09-20", passenger_name="Sara Mostafa")
    tracking.run(case)
    claim_agent.decide(case)
    assert case.decision["action"] == "duplicate_prevented"
    assert case.claim["claim_id"] == "CLM-1001"


# ----------------------------------------------------------------- Agent 4
def test_communication_agent_generates_grounded_email(services):
    tracking = TrackingAgent(services.db_path)
    claim_agent = ClaimAgent(services.db_path, services.llm, services.exports_dir)
    comm_agent = CommunicationAgent(services.db_path, services.llm, services.policy_path)
    case = _filled_case()
    tracking.run(case)
    claim_agent.decide(case)
    claim_agent.create_claim(case)
    comm_agent.run(case)
    assert case.email is not None
    assert case.email["recipient"] == "ghada@example.com"
    assert case.claim["claim_id"] in case.email["subject"] + case.email["body"]
    assert case.baggage["status"] in case.email["body"] or case.baggage["status"].lower() in case.email["body"].lower()
