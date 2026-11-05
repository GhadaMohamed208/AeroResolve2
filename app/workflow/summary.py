"""Data processing & summarisation: structured case summary + LLM narrative + PDF report."""
from __future__ import annotations

from app.llm.guard import grounded_generate
from app.services import Services
from app.tools.base import call_tool
from app.tools.data_tools import generate_case_pdf
from app.workflow.state import CaseState

AGENT = "Case Summary"

NEXT_STEP = {
    "create_claim": "Track the claim; passenger receives status updates at least every 48 hours.",
    "duplicate_prevented": "No action - an active claim already exists.",
    "no_claim_delivered": "None - baggage delivered. Damage reports are handled at the airport service office.",
    "monitor": "Monitor the baggage and notify the passenger when its status changes.",
    "manual_review": "Human agent to review the case and contact the passenger within one business day.",
}
_SYSTEM = ("You summarise baggage cases for airline support staff in exactly 2 short sentences, using ONLY the "
           "facts provided. Never invent identifiers, dates, locations or statuses.")


class CaseSummarizer:
    def __init__(self, services: Services):
        self.s = services

    def run(self, case: CaseState) -> None:
        d = case.decision
        b = case.baggage or {}
        booking = case.booking or {}
        flight = case.flight or {}
        claim = case.claim
        fields = {
            "Passenger": booking.get("passenger_name", case.intake.get("passenger_name")),
            "Booking": booking.get("booking_reference", case.intake.get("booking_reference")),
            "Flight": f"{booking.get('flight_number', '')} ({flight.get('origin', '?')} to {flight.get('destination', '?')}), {booking.get('flight_date', '')}",
            "Baggage Tag": b.get("baggage_tag") or case.intake.get("baggage_tag"),
            "Status": b.get("status", "Unknown (tag not found)"),
            "Last Known Location": b.get("last_known_location") or "n/a",
            "Last Updated": b.get("last_updated") or "n/a",
            "Decision": d["reason"],
            "Claim": claim["claim_id"] if claim else "None required" if d["action"] in ("no_claim_delivered", "monitor") else "None (manual review)",
            "Claim Type": claim["claim_type"] if claim else "-",
            "Claim Status": claim["status"] if claim else "-",
            "Follow-up": f"{case.follow_up['follow_up_type']} (due {case.follow_up['due_at']} UTC)" if case.follow_up else "-",
            "Next Step": NEXT_STEP[d["action"]],
            "Contact Email": case.intake.get("email"),
        }
        tag, status = fields["Baggage Tag"], b.get("status")
        req = [tag] + ([status] if status else []) + ([claim["claim_id"]] if claim else [])
        fallback = lambda: case.analysis or d["reason"]
        prompt = "Facts: " + "; ".join(f"{k}={v}" for k, v in fields.items() if k != "Contact Email")
        narrative, src = grounded_generate(self.s.llm, _SYSTEM, prompt, required=req,
                                           allowed_ids=case.facts_ids, fallback=fallback, case=case,
                                           agent=AGENT, label="narrative")
        case.summary = {"fields": fields, "narrative": narrative, "narrative_source": src,
                        "claim_id": claim["claim_id"] if claim else None, "baggage_tag": tag}
        case.summary_text = "Baggage Case Summary\n" + "-" * 20 + "\n" + "\n".join(
            f"{k}: {v}" for k, v in fields.items() if k != "Contact Email") + f"\n\nAnalysis: {narrative}"
        pdf = call_tool(case, AGENT, generate_case_pdf, case_summary=case.summary, email=case.email,
                        exports_dir=self.s.exports_dir)
        case.report_path = pdf.data["path"] if pdf.ok else None
        if not pdf.ok:
            case.log(AGENT, "pdf_failed", "warn", pdf.message or pdf.error_code or "")
        case.log(AGENT, "summary_created", detail=f"narrative by {src}")
