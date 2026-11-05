"""Dedicated tests for conversation memory / session state (as distinct from
persistent long-term memory, retrieval, or embeddings - see README §11).

These prove, through the real Orchestrator (not a mock), that:
  1. the chat log (CaseState.messages) accumulates every turn,
  2. a field given in one turn is remembered and is NOT re-asked in a later turn,
  3. a short follow-up reply is correctly attached to whichever field the
     system is currently waiting for (the exact scenario from the project
     brief: "My baggage is missing." -> "provide your booking reference" ->
     "BK1023" -> understood as belonging to the current case),
  4. state does NOT leak between two independent cases (no accidental
     "memory" of a different passenger).
"""
from __future__ import annotations


def test_chat_log_accumulates_every_turn(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My baggage did not arrive.")
    orchestrator.run_turn(case, "I am Ghada Mohamed")
    roles = [m["role"] for m in case.messages]
    assert roles.count("user") == 2
    assert roles.count("assistant") == 2
    assert case.messages[0]["content"] == "My baggage did not arrive."
    assert case.messages[2]["content"] == "I am Ghada Mohamed"


def test_brief_example_scenario_field_remembered_across_turns(orchestrator):
    """Exact shape of the brief's example, using this project's real ID format
    (BK#### rather than the illustrative "ABC123", which this system's booking
    references are never formatted like - see README §11 for that nuance)."""
    case = orchestrator.new_case()

    orchestrator.run_turn(case, "My baggage is missing.")
    assert case.stage == "collecting"
    assert "booking_reference" in case.missing_fields
    assert "booking" in case.reply.lower() or "reference" in case.reply.lower()

    # The user now replies with JUST the booking reference - nothing else repeated.
    orchestrator.run_turn(case, "BK1023")
    assert case.intake.get("booking_reference") == "BK1023"
    # The system must NOT have forgotten it was collecting a baggage case.
    assert case.stage in ("collecting", "verifying")
    assert "booking_reference" not in case.missing_fields


def test_previously_given_fields_are_never_re_asked(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My baggage did not arrive. I am Ghada Mohamed, booking BK1023.")
    assert case.intake.get("passenger_name") == "Ghada Mohamed"
    assert case.intake.get("booking_reference") == "BK1023"
    first_missing = set(case.missing_fields)
    assert "passenger_name" not in first_missing
    assert "booking_reference" not in first_missing

    # Second turn supplies more info; name/booking must still be remembered.
    orchestrator.run_turn(case, "flight FL123 on 2026-09-20")
    assert case.intake.get("passenger_name") == "Ghada Mohamed"      # not lost
    assert case.intake.get("booking_reference") == "BK1023"          # not lost
    assert case.intake.get("flight_number") == "FL123"               # newly added
    assert "passenger_name" not in case.missing_fields
    assert "booking_reference" not in case.missing_fields


def test_full_case_completes_purely_from_incremental_turns(orchestrator):
    """Every field arrives in a separate message; the case must still complete,
    proving turn-by-turn accumulation works end-to-end, not just extraction of a
    single message that happens to contain everything."""
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My baggage did not arrive.")
    orchestrator.run_turn(case, "I am Ghada Mohamed")
    orchestrator.run_turn(case, "booking BK1023")
    orchestrator.run_turn(case, "flight FL123 on 2026-09-20")
    orchestrator.run_turn(case, "tag BG77821")
    orchestrator.run_turn(case, "email ghada@example.com")
    assert case.stage == "completed"
    assert case.baggage["status"] == "Delayed"
    assert case.claim is not None


def test_verification_correction_is_remembered_not_restarted(orchestrator):
    """An invalid tag triggers a targeted re-ask; the correction must be applied
    to the SAME case (booking/name/flight already given are not lost or re-asked)."""
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My baggage is missing. I am Ghada Mohamed, booking BK1023, "
                                 "flight FL123 on 2026-09-20, tag BG00000, email ghada@example.com")
    assert case.stage == "verifying"
    assert case.intake.get("booking_reference") == "BK1023"   # still remembered

    orchestrator.run_turn(case, "BG77821")
    assert case.stage == "completed"
    assert case.intake.get("booking_reference") == "BK1023"   # still the same case
    assert case.baggage["baggage_tag"] == "BG77821"


def test_state_does_not_leak_between_independent_cases(orchestrator):
    """Two separate CaseState objects (e.g. two browser tabs) must never share data."""
    case_a = orchestrator.new_case()
    orchestrator.run_turn(case_a, "I am Ghada Mohamed, booking BK1023")

    case_b = orchestrator.new_case()
    orchestrator.run_turn(case_b, "My baggage did not arrive.")

    assert case_b.intake.get("passenger_name") is None
    assert case_b.intake.get("booking_reference") is None
    assert case_a.intake.get("passenger_name") == "Ghada Mohamed"  # untouched by case_b


def test_completed_case_starts_fresh_on_new_unrelated_report(orchestrator):
    """After a case completes, a brand-new baggage report resets the case (not an
    unbounded memory of every past case), while still keeping the chat transcript."""
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My baggage did not arrive. I am Ghada Mohamed, booking BK1023, "
                                 "flight FL123 on 2026-09-20, tag BG77821, email ghada@example.com")
    assert case.stage == "completed"
    first_claim = case.claim["claim_id"] if case.claim else None

    orchestrator.run_turn(case, "My baggage is also missing. I am Omar Hassan, booking BK2045, "
                                 "flight FL210 on 2026-09-19, tag BG50412, email omar.hassan@example.com")
    assert case.intake.get("passenger_name") == "Omar Hassan"       # old case's data is gone
    assert case.intake.get("booking_reference") == "BK2045"
    assert len(case.messages) == 4                                   # transcript still accumulates
    if first_claim and case.claim:
        assert case.claim["claim_id"] != first_claim or case.decision["action"] != "create_claim"
