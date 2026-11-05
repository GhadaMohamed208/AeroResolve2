"""End-to-end orchestrator tests (Requirement 4: Workflow Orchestration).

These drive ``Orchestrator.run_turn`` exactly as the UI does - full multi-agent
graph execution, shared state, conditional branching, persistence.
"""
from __future__ import annotations

from app.tools.claim_tools import list_records


def test_full_turn_populates_shared_state(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My baggage did not arrive. I am Ghada Mohamed, booking BK1023, "
                                 "flight FL123 on 2026-09-20, baggage tag BG77821, email ghada@example.com")
    assert case.stage == "completed"
    assert case.agent_status == {"intake": "done", "tracking": "done", "claim": "done", "communication": "done"}
    assert case.baggage["status"] == "Delayed"
    assert case.claim["claim_id"].startswith("CLM-")
    assert case.email is not None
    assert case.summary is not None
    assert case.report_path  # PDF case report was generated


def test_multi_turn_conversation_collects_then_completes(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My baggage did not arrive.")
    assert case.stage == "collecting"
    orchestrator.run_turn(case, "I am Omar Hassan, booking BK2045, flight FL210, date 2026-09-19, "
                                 "tag BG50412, email omar.hassan@example.com")
    assert case.stage == "completed"
    assert case.decision["action"] == "no_claim_delivered"
    assert case.claim is None


def test_workflow_survives_and_recovers_from_verification_loop(orchestrator):
    """Invalid tag -> agent asks to verify -> corrected tag -> case completes."""
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My baggage is missing. I am Ghada Mohamed, booking BK1023, "
                                 "flight FL123 on 2026-09-20, tag BG00000, email ghada@example.com")
    assert case.stage == "verifying"
    orchestrator.run_turn(case, "BG77821")
    assert case.stage == "completed"
    assert case.baggage["baggage_tag"] == "BG77821"


def test_claim_persists_across_orchestrator_runs(orchestrator, services):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My baggage is missing. I am Layla Ibrahim, booking BK3078, "
                                 "flight FL305 on 2026-09-21, tag BG90233, email layla.ibrahim@example.com")
    claims = list_records(services.db_path, "claims")
    assert claims.ok
    ids = [c["claim_id"] for c in claims.data["rows"]]
    assert case.claim["claim_id"] in ids


def test_second_report_of_same_bag_does_not_duplicate(orchestrator, services):
    msg = ("My baggage is delayed. I am Ghada Mohamed, booking BK1023, flight FL123 on 2026-09-20, "
           "tag BG77821, email ghada@example.com")
    case1 = orchestrator.new_case()
    orchestrator.run_turn(case1, msg)
    case2 = orchestrator.new_case()
    orchestrator.run_turn(case2, msg)
    assert case1.claim["claim_id"] == case2.claim["claim_id"]
    assert case2.decision["action"] == "duplicate_prevented"
    claims = list_records(services.db_path, "claims")
    tags = [c["baggage_tag"] for c in claims.data["rows"]]
    assert tags.count("BG77821") == 1


def test_policy_question_uses_rag_and_does_not_start_a_case(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "What is the compensation limit for lost baggage?")
    assert case.stage == "idle"
    assert case.policy_answer is not None
    assert "1,500" in case.reply or "1500" in case.reply


def test_workflow_never_crashes_on_empty_message(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "")
    assert case.reply
    assert not case.errors


def test_workflow_never_crashes_on_gibberish(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "asdkjf ;lqkwjelrkj 12309 !@#$%^")
    assert case.reply
    assert not case.errors
