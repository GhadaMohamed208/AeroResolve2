"""Agent 3 - Claim Agent.

Takes the *retrieved* baggage status and decides - by explicit, auditable rules - whether a claim,
an investigation, monitoring or manual review is needed, then performs the action through tools.

    Delivered            -> no claim, inform passenger
    Delayed              -> create claim (Delayed Baggage)
    Not Found            -> create investigation claim (Lost Baggage Investigation)
    Under Investigation  -> create claim if none exists, otherwise link the existing one
    In Transit / At Airport -> no claim; monitor and inform (follow-up record)
    Unknown (tag unresolved) -> escalate to manual review

The LLM only writes the human-readable analysis; it never chooses the action.
"""
from __future__ import annotations

from app.llm.client import LLMClient
from app.llm.guard import grounded_generate
from app.tools.base import call_tool
from app.tools.claim_tools import CLAIM_TYPE_BY_STATUS, create_baggage_claim, create_follow_up, get_existing_claim
from app.tools.data_tools import export_claims_csv
from app.workflow.state import CaseState

AGENT = "Claim Agent"

_SYSTEM = ("You are the Claim Agent of an airline baggage-support prototype. Explain the case decision to a "
           "support colleague in 2 short sentences using ONLY the facts provided. Never invent identifiers, "
           "dates, locations or statuses.")

REASONS = {
    "Delivered": "The baggage is already delivered, so no claim is created (policy section 8).",
    "Delayed": "The baggage is delayed, so a delayed-baggage claim is required.",
    "Not Found": "The baggage cannot be located, so a lost-baggage investigation claim is required.",
    "Under Investigation": "The baggage is already under investigation, so a claim must exist for it.",
    "In Transit": "The baggage is still moving between airports; no claim is needed yet, so it is monitored.",
    "At Airport": "The baggage has been located at an airport; no claim is needed, delivery is being arranged.",
}


class ClaimAgent:
    name = AGENT

    def __init__(self, db_path: str, llm: LLMClient, exports_dir: str):
        self.db_path, self.llm, self.exports_dir = db_path, llm, exports_dir

    # ------------------------------------------------------ status decision
    def decide(self, case: CaseState) -> None:
        case.set_agent("claim", "running")
        bag = case.baggage
        status = bag["status"]
        existing = call_tool(case, AGENT, get_existing_claim, self.db_path, baggage_tag=bag["baggage_tag"])
        if not existing.ok:
            return self._fail(case, existing)
        case.existing_claim = existing.data["claim"]

        if status == "Delivered":
            decision = dict(action="no_claim_delivered", claim_type=None, follow_up_type=None)
        elif status in ("In Transit", "At Airport"):
            decision = dict(action="monitor", claim_type=None,
                            follow_up_type="Monitor Baggage" if status == "In Transit" else "Arrange Delivery")
        elif status in CLAIM_TYPE_BY_STATUS:
            action = "duplicate_prevented" if case.existing_claim else "create_claim"
            decision = dict(action=action, claim_type=CLAIM_TYPE_BY_STATUS[status], follow_up_type=None)
        else:
            decision = dict(action="manual_review", claim_type=None, follow_up_type="Manual Review")
        decision["status"] = status
        decision["reason"] = REASONS.get(status, "The status is not recognised, so the case needs manual review.")
        if decision["action"] == "duplicate_prevented":
            decision["reason"] += f" An active claim ({case.existing_claim['claim_id']}) already exists, so no duplicate is created."
            case.claim = case.existing_claim
        decision["claim_required"] = decision["action"] in ("create_claim", "duplicate_prevented")
        case.decision = decision
        case.analysis = self._analysis(case)
        case.log(AGENT, "decision", detail=f"status={status} -> {decision['action']}")
        if decision["action"] in ("no_claim_delivered", "duplicate_prevented"):
            case.set_agent("claim", "done")      # nothing further to execute

    # ----------------------------------------------------------- actions
    def create_claim(self, case: CaseState) -> None:
        bag, decision = case.baggage, case.decision
        result = call_tool(
            case, AGENT, create_baggage_claim, self.db_path, baggage_tag=bag["baggage_tag"],
            booking_reference=case.booking["booking_reference"], passenger_name=case.booking["passenger_name"],
            flight_number=case.booking["flight_number"], flight_date=case.booking["flight_date"],
            claim_type=decision["claim_type"], contact_email=case.intake["email"])
        if not result.ok:
            return self._fail(case, result)
        case.claim = result.data["claim"]
        if result.data["duplicate"]:
            decision["action"] = "duplicate_prevented"
            decision["reason"] += f" An active claim ({case.claim['claim_id']}) already exists."
            case.claim_created = False
            case.log(AGENT, "duplicate_prevented", "warn", case.claim["claim_id"])
        else:
            case.claim_created = True
            case.log(AGENT, "claim_created", detail=f"{case.claim['claim_id']} ({case.claim['claim_type']})")
            fu = call_tool(case, AGENT, create_follow_up, self.db_path, baggage_tag=bag["baggage_tag"],
                           booking_reference=case.booking["booking_reference"], follow_up_type="Status Update",
                           note=f"Send a status update for {case.claim['claim_id']} within 48 hours.",
                           claim_id=case.claim["claim_id"], due_hours=48)
            case.follow_up = fu.data if fu.ok else None
            call_tool(case, AGENT, export_claims_csv, self.db_path, exports_dir=self.exports_dir)
        case.analysis = self._analysis(case)
        case.set_agent("claim", "done")

    def follow_up(self, case: CaseState) -> None:
        """Monitoring / manual-review follow-up record (no claim)."""
        d = case.decision
        tag = case.intake["baggage_tag"]
        if d["action"] == "manual_review":
            note, hours = f"Tag {tag} could not be matched to any record; human agent to contact the passenger.", 24
        else:
            note = (f"Monitor baggage {tag} ({d['status']}) and notify the passenger when the status changes."
                    if d["action"] == "monitor" else "Follow up.")
            hours = 12 if d["status"] == "In Transit" else 24
        fu = call_tool(case, AGENT, create_follow_up, self.db_path, baggage_tag=tag,
                       booking_reference=case.booking["booking_reference"], follow_up_type=d["follow_up_type"],
                       note=note, due_hours=hours)
        if fu.ok:
            case.follow_up = fu.data
            case.log(AGENT, "follow_up_created", detail=f"{d['follow_up_type']} due {fu.data['due_at']}")
        else:  # non-fatal: the passenger still gets informed
            case.log(AGENT, "follow_up_failed", "warn", fu.message or fu.error_code or "")
        case.set_agent("claim", "done")

    def escalate(self, case: CaseState) -> None:
        """Unknown/unresolvable baggage -> manual review."""
        case.set_agent("claim", "running")
        case.decision = dict(
            status="Unknown", action="manual_review", claim_type=None, follow_up_type="Manual Review",
            claim_required=False,
            reason=f"Baggage tag {case.intake.get('baggage_tag')} could not be matched to any record, so the case is escalated for manual review.")
        case.analysis = case.decision["reason"]
        self.follow_up(case)
        case.log(AGENT, "escalated_manual_review", "warn", case.decision["reason"])

    # ------------------------------------------------------------ helpers
    def _analysis(self, case: CaseState) -> str:
        bag, d = case.baggage, case.decision
        claim = case.claim
        base = (f"Baggage {bag['baggage_tag']} is {bag['status']} (last seen: {bag['last_known_location']}, "
                f"updated {bag['last_updated']}). {d['reason']}")
        prompt = (f"Facts: tag={bag['baggage_tag']}; status={bag['status']}; location={bag['last_known_location']}; "
                  f"last_updated={bag['last_updated']}; decision={d['action']}; rule={d['reason']}; "
                  f"claim_id={claim['claim_id'] if claim else 'none'}")
        text, _ = grounded_generate(
            self.llm, _SYSTEM, prompt, required=[bag["baggage_tag"], bag["status"]],
            allowed_ids=case.facts_ids, fallback=lambda: base, case=case, agent=AGENT, label="analysis")
        return text

    def _fail(self, case: CaseState, result) -> None:
        case.stage = "error"
        case.halt = True
        case.set_agent("claim", "error")
        case.errors.append(f"{result.error_code}: {result.message}")
        case.reply = ("I found your baggage but couldn't complete the claim step because of a system error. "
                      "Nothing was duplicated - please send **retry** in a moment, or contact support if it persists.")
        case.log(AGENT, "action_failed", "error", f"{result.error_code}: {result.message}")
