"""The six required Final-Project test scenarios, run end-to-end through the
orchestrator exactly as a passenger would trigger them in the UI.

Also produces a small evaluation summary (workflow success rate, correct
status handling, claim-creation accuracy, tool-call success rate, response
time) printed at the end of the run - see ``test_zzz_print_evaluation_summary``.
"""
from __future__ import annotations

import time

RESULTS: list[dict] = []


def _record(name: str, case, ok: bool) -> None:
    tool_calls = case.tool_calls
    tool_ok = sum(1 for t in tool_calls if t["ok"])
    RESULTS.append({
        "scenario": name,
        "ok": ok,
        "stage": case.stage,
        "tool_calls": len(tool_calls),
        "tool_success_rate": (tool_ok / len(tool_calls)) if tool_calls else 1.0,
        "response_ms": case.last_run_ms,
    })


# ---------------------------------------------------------- Scenario 1: Delayed
def test_scenario_1_delayed_baggage_creates_claim_and_email(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My baggage did not arrive. I am Ghada Mohamed, booking BK1023, "
                                 "flight FL123 on 2026-09-20, baggage tag BG77821, email ghada@example.com")
    ok = (case.baggage["status"] == "Delayed" and case.decision["action"] == "create_claim"
          and case.claim is not None and case.email is not None and case.stage == "completed")
    _record("1. Delayed baggage -> claim -> communication", case, ok)
    assert ok


# --------------------------------------------------------- Scenario 2: Delivered
def test_scenario_2_delivered_baggage_no_claim(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "I can't find my luggage. I am Omar Hassan, booking BK2045, flight FL210, "
                                 "date 2026-09-19, tag BG50412, email omar.hassan@example.com")
    ok = (case.baggage["status"] == "Delivered" and case.decision["action"] == "no_claim_delivered"
          and case.claim is None and case.email is not None and case.stage == "completed")
    _record("2. Delivered baggage -> no claim -> passenger informed", case, ok)
    assert ok


# --------------------------------------------------------- Scenario 3: Not Found
def test_scenario_3_not_found_opens_investigation(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My suitcase is lost. I am Layla Ibrahim, booking BK3078, flight FL305 "
                                 "on 2026-09-21, tag BG90233, email layla.ibrahim@example.com")
    ok = (case.baggage["status"] == "Not Found" and case.decision["action"] == "create_claim"
          and case.claim["claim_type"] == "Lost Baggage Investigation"
          and case.claim["status"] == "Under Investigation" and case.stage == "completed")
    _record("3. Not found -> investigation claim -> escalation email", case, ok)
    assert ok


# --------------------------------------------------------- Scenario 4: Missing info
def test_scenario_4_missing_information_prompts_follow_up(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My baggage did not arrive.")
    ok = (case.stage == "collecting" and bool(case.missing_fields) and bool(case.reply)
          and case.baggage is None and case.claim is None)
    _record("4. Missing information -> intake asks follow-up", case, ok)
    assert ok


# --------------------------------------------------------- Scenario 5: Invalid tag
def test_scenario_5_invalid_baggage_tag_triggers_verification(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My baggage is missing. I am Ghada Mohamed, booking BK1023, flight FL123 "
                                 "on 2026-09-20, tag BG00000, email ghada@example.com")
    ok = (case.stage == "verifying" and "baggage_tag" in case.missing_fields
          and case.claim is None and case.baggage is None)
    _record("5. Invalid baggage tag -> system asks to verify", case, ok)
    assert ok


# --------------------------------------------------------- Scenario 6: Duplicate
def test_scenario_6_duplicate_claim_is_prevented(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "My baggage is delayed. I am Sara Mostafa, booking BK7071, flight FL123 "
                                 "on 2026-09-20, tag BG45120, email sara@example.com")
    ok = (case.decision["action"] == "duplicate_prevented" and case.claim["claim_id"] == "CLM-1001"
          and case.claim_created is False and case.stage == "completed")
    _record("6. Duplicate claim -> existing claim reused, no duplicate", case, ok)
    assert ok


# --------------------------------------------------------------- extra: timing
def test_scenario_extra_response_time_is_reasonable(orchestrator):
    case = orchestrator.new_case()
    start = time.perf_counter()
    orchestrator.run_turn(case, "My baggage did not arrive. I am Ghada Mohamed, booking BK1023, "
                                 "flight FL123 on 2026-09-20, baggage tag BG77821, email ghada@example.com")
    elapsed = time.perf_counter() - start
    _record("7. Response time (rule-based mode)", case, elapsed < 5.0)
    assert elapsed < 5.0  # generous ceiling; rule-based mode typically completes in well under 1s


# ------------------------------------------------------------ evaluation report
def test_zzz_print_evaluation_summary():
    """Not a real assertion - prints the metrics table described in the project brief.

    Named ``test_zzz_*`` so pytest (which collects and runs tests in file order by
    default) executes it after the six scenarios above have populated ``RESULTS``.
    """
    if not RESULTS:
        return
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["ok"])
    avg_tool_success = sum(r["tool_success_rate"] for r in RESULTS) / total
    avg_ms = sum(r["response_ms"] for r in RESULTS) / total
    print("\n\n=== AeroResolve Evaluation Summary ===")
    print(f"{'Scenario':55s} {'Result':8s} {'Tools':6s} {'ms':8s}")
    for r in RESULTS:
        print(f"{r['scenario']:55s} {'PASS' if r['ok'] else 'FAIL':8s} "
              f"{r['tool_calls']:<6d} {r['response_ms']:<8.1f}")
    print("-" * 80)
    print(f"Workflow success rate : {passed}/{total} ({100 * passed / total:.0f}%)")
    print(f"Tool-call success rate: {100 * avg_tool_success:.0f}% (average across scenarios)")
    print(f"Average response time : {avg_ms:.1f} ms")
    assert True
