"""Agent 2 - Baggage Tracking Agent.

Retrieves booking, flight and baggage information ONLY through tools that query the project
database. It never fabricates a status: whatever it hands to the next agents is a tool result.
"""
from __future__ import annotations

from app.i18n import FIELD_LABELS_AR
from app.tools.base import call_tool
from app.tools.baggage_tools import (
    check_booking, find_baggage_by_booking, get_flight_information, track_baggage,
)
from app.workflow.state import FIELD_LABELS, CaseState

AGENT = "Tracking Agent"
MAX_TAG_ATTEMPTS = 2   # after this many unknown-tag attempts the case goes to manual review

# Arabic translations of the mismatch reason, keyed by which field the tool
# flagged (``result.data["field"]``) - the tool's own message stays English
# (it is a dev-facing error_code/message pair, logged to the trace), but the
# passenger-facing wrapper below must speak the passenger's own language.
_MISMATCH_AR = {
    "booking_reference": "رقم الحجز ده مش موجود عندنا.",
    "passenger_name": "الاسم مش مطابق للحجز.",
    "flight_number": "رقم الرحلة مش مطابق للحجز.",
    "flight_date": "تاريخ الرحلة مش مطابق للحجز.",
}


class TrackingAgent:
    name = AGENT

    def __init__(self, db_path: str):
        self.db_path = db_path

    def run(self, case: CaseState) -> None:
        f = case.intake
        case.set_agent("tracking", "running")

        booking = call_tool(case, AGENT, check_booking, self.db_path,
                            booking_reference=f["booking_reference"], passenger_name=f["passenger_name"],
                            flight_number=f["flight_number"], flight_date=f["flight_date"])
        if not booking.ok:
            return self._failure(case, booking)
        case.booking = booking.data
        case.log(AGENT, "booking_verified", detail=f"{booking.data['booking_reference']} matches passenger, flight and date")

        flight = call_tool(case, AGENT, get_flight_information, self.db_path,
                           flight_number=f["flight_number"], flight_date=f["flight_date"])
        if not flight.ok:
            return self._failure(case, flight)
        case.flight = flight.data

        bag = call_tool(case, AGENT, track_baggage, self.db_path, baggage_tag=f["baggage_tag"])
        if not bag.ok:
            if bag.error_code == "TAG_NOT_FOUND":
                return self._unknown_tag(case)
            return self._failure(case, bag)
        if bag.data["booking_reference"] != case.booking["booking_reference"]:
            case.tag_failures += 1
            f.pop("baggage_tag", None)
            msg = ("تاج الشنطة ده مسجل على حجز تاني. من فضلك راجع الرقم المطبوع على إيصال الشنطة وابعته تاني."
                   if case.language == "ar" else
                   "That baggage tag is registered to a different booking. "
                   "Please check the tag on your baggage receipt and send it again.")
            return self._verify(case, msg, "baggage_tag")
        case.baggage = bag.data
        case.tracking_outcome = "ok"
        case.set_agent("tracking", "done")
        case.log(AGENT, "baggage_retrieved", detail=f"{bag.data['baggage_tag']}: {bag.data['status']} @ {bag.data['last_known_location']}")

    # ------------------------------------------------------------- outcomes
    def _failure(self, case: CaseState, result) -> None:
        if result.error_code in ("DB_ERROR", "TOOL_EXCEPTION", "BAD_TOOL_RESULT"):
            case.tracking_outcome = "error"
            case.stage = "error"
            case.halt = True
            case.set_agent("tracking", "error")
            case.errors.append(f"{result.error_code}: {result.message}")
            case.reply = ("في مشكلة وصول لنظام الشنط دلوقتي، فمقدرتش أدور على شنطتك. بياناتك محفوظة - "
                          "ابعت **retry** بعد شوية."
                          if case.language == "ar" else
                          "I'm having trouble reaching the baggage system right now, so I couldn't check your "
                          "bag. Your details are saved - please send **retry** in a moment.")
            case.log(AGENT, "tool_failure", "error", f"{result.error_code}")
            return
        field = (result.data or {}).get("field", "booking_reference")
        case.intake.pop(field, None)
        if case.language == "ar":
            msg = _MISMATCH_AR.get(field, "البيانات دي مش مطابقة لحجزك.")
            self._verify(case, f"{msg} لأسباب أمنية مينفعش أشارك تفاصيل الحجز، فمن فضلك راجع البيانات.", field)
        else:
            self._verify(case, f"{result.message} For your security I can't share booking details, so "
                               f"please double-check the information.", field)

    def _verify(self, case: CaseState, message: str, field: str) -> None:
        case.tracking_outcome = "verify"
        case.stage = "verifying"
        case.missing_fields = [field]
        case.last_asked = [field]
        case.intake_complete = False
        case.set_agent("tracking", "waiting")
        case.set_agent("intake", "waiting")
        case.log(AGENT, "verification_needed", "warn", f"{field}: {message}")
        if case.language == "ar":
            label = FIELD_LABELS_AR.get(field, field)
            case.reply = f"{message}\n\nمن فضلك ابعت **{label}** الصحيح."
        else:
            case.reply = f"{message}\n\nPlease send your correct **{FIELD_LABELS[field]}**."

    def _unknown_tag(self, case: CaseState) -> None:
        case.tag_failures += 1
        tag = case.intake.get("baggage_tag")
        by_booking = call_tool(case, AGENT, find_baggage_by_booking, self.db_path,
                               booking_reference=case.booking["booking_reference"])
        tags = (by_booking.data or {}).get("tags", []) if by_booking.ok else []
        if case.tag_failures >= MAX_TAG_ATTEMPTS:
            case.tracking_outcome = "escalate"
            case.set_agent("tracking", "done")
            case.log(AGENT, "tag_unresolved", "warn", f"tag {tag} not found after {case.tag_failures} attempts - escalating")
            return
        case.intake.pop("baggage_tag", None)
        case.suggested_tag = tags[0] if tags else None
        if case.language == "ar":
            msg = f"معرفتش ألاقي سجل شنطة بالتاج **{tag}**."
            if case.suggested_tag:
                msg += (f" حجزك مسجل عليه تاج شنطة **{case.suggested_tag}** - ده هو؟ "
                        f"رد **yes** عشان أستخدمه، أو ابعت التاج الصح.")
        else:
            msg = f"I couldn't find a baggage record for tag **{tag}**."
            if case.suggested_tag:
                msg += (f" Your booking has bag tag **{case.suggested_tag}** registered - is that the one? "
                        f"Reply **yes** to use it, or send the correct tag.")
        if case.suggested_tag:
            case.tracking_outcome = "verify"
            case.stage = "verifying"
            case.missing_fields = ["baggage_tag"]
            case.last_asked = ["baggage_tag"]
            case.intake_complete = False
            case.set_agent("tracking", "waiting")
            case.set_agent("intake", "waiting")
            case.log(AGENT, "verification_needed", "warn", f"unknown tag {tag}; suggesting {case.suggested_tag}")
            case.reply = msg
            return
        suffix = " من فضلك راجع التاج المطبوع على إيصال الشنطة." if case.language == "ar" \
            else " Please check the tag printed on your baggage receipt."
        self._verify(case, msg + suffix, "baggage_tag")
