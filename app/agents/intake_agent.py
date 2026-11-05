"""Agent 1 - Baggage Intake Agent.

Responsibilities
* detect the passenger's intent (baggage problem / policy question / other)
* extract structured fields from natural language (LLM + deterministic rules)
* validate every value (format checks) and refuse to continue until all required
  information is present
* ask targeted follow-up questions for whatever is missing or invalid

LLM output is *grounded*: an extracted booking reference, tag, flight, e-mail, date or
name is only accepted if it also validates AND actually appears in the passenger's own
text. The model therefore cannot invent identifiers.
"""
from __future__ import annotations

import re

from app.i18n import AR_BAGGAGE_ISSUE_RE, AR_POLICY_RE, FIELD_HINTS_AR, FIELD_LABELS_AR, detect_language
from app.llm.client import LLMClient, LLMError
from app.llm.guard import grounded_generate
from app.validation import (
    detect_invalid, extract_fields, norm_booking, norm_date, norm_email, norm_flight, norm_name, norm_tag,
)
from app.workflow.state import FIELD_LABELS, REQUIRED_FIELDS, TERMINAL_STAGES, CaseState

AGENT = "Intake Agent"

_STATUS_RE = re.compile(
    r"(where\s+is|track|status\s+of|did\s?n[o']?t\s+arrive|did\s+not\s+arrive|not\s+arrive|hasn'?t\s+arrived|"
    r"has\s+not\s+arrived|never\s+arrived|missing|lost|delay|can'?t\s+find|cannot\s+find|not\s+received|"
    r"left\s+behind|didn'?t\s+get|did\s+not\s+get|no\s+sign)", re.I)
_BAGGAGE_RE = re.compile(r"\b(bag|bags|baggage|luggage|suitcase|suitcases|belongings)\b", re.I)
_POLICY_RE = re.compile(
    r"(polic(?:y|ies)|compensat|reimburs|allowance|entitled|refund|\brules?\b|\blimits?\b|deadline|how\s+long|"
    r"how\s+many|how\s+much|what\s+happens|what\s+if|expenses|receipts|documents?\s+(?:needed|required)|"
    r"claim\s+(?:process|form|deadline)|time\s*limit)", re.I)
_QUESTION_START = re.compile(r"\s*(what|how|when|can|do|does|am|is|are|will|which|who|should)\b", re.I)
_YES_RE = re.compile(r"^\s*(yes|yeah|yep|yup|correct|sure|that'?s\s+(?:it|right|mine|correct)|it\s+is|right)\b[\s.!]*$", re.I)

FIELD_HINTS = {
    "passenger_name": "your full name as it appears on the booking",
    "booking_reference": "your booking reference (e.g. BK1234)",
    "flight_number": "your flight number (e.g. FL123)",
    "flight_date": "the flight date (YYYY-MM-DD, e.g. 2026-09-20)",
    "baggage_tag": "your baggage tag number (e.g. BG12345, printed on your baggage receipt)",
    "email": "an email address where we can send updates",
}
_REQ_WORDS = {"passenger_name": "name", "booking_reference": "booking", "flight_number": "flight",
              "flight_date": "date", "baggage_tag": "tag", "email": "email"}
# Arabic equivalents, used for the grounding check (required fact == must appear
# in the LLM's own output) when the reply is written in Arabic - the English
# words above would never appear in a fluent Arabic reply, which would make
# the grounding guard reject every Arabic LLM answer and silently force the
# deterministic (English) fallback. This keeps "ask for missing fields" LLM
# output honest in both languages.
_REQ_WORDS_AR = {"passenger_name": "اسم", "booking_reference": "حجز", "flight_number": "رحلة",
                 "flight_date": "تاريخ", "baggage_tag": "تاج", "email": "بريد"}

_SYSTEM_EXTRACT = (
    "You are the Baggage Intake Agent of an airline baggage-support prototype. Read the passenger's message "
    "and extract information. Reply with ONE JSON object and nothing else. Never guess or invent values: "
    "use null for anything the passenger did not explicitly state. The passenger may write in Arabic or "
    "English (or mix Latin IDs into an Arabic sentence) - extract fields correctly in either case.")
_SYSTEM_ASK = (
    "You are a warm, concise airline baggage-support assistant. Never invent facts, booking data, statuses "
    "or promises. Never mention any identifier that the passenger did not give you. Detect the language the "
    "passenger is writing in (Arabic or English) and reply fluently in that SAME language - never switch "
    "the passenger to the other language.")


class IntakeAgent:
    name = AGENT

    def __init__(self, llm: LLMClient):
        self.llm = llm

    # ------------------------------------------------------------------ main
    def run(self, case: CaseState) -> None:
        text = case.user_message.strip()
        self._update_language(case, text)
        collecting = bool(case.last_asked) or any(case.intake.get(k) for k in REQUIRED_FIELDS)
        rule = extract_fields(text, expecting=case.last_asked)
        llm_out = self._llm_extract(case, text)
        intent = self._intent(text, rule, llm_out, collecting)
        case.intent = intent
        case.log(AGENT, "intent_detected", detail=f"{intent} (fields found by rules: {sorted(rule) or 'none'})")
        if intent != "baggage_issue":
            return

        if case.stage in TERMINAL_STAGES:      # a finished case + a new report => start a new case
            case.reset_case()
            case.user_message, case.intent = text, intent
            rule = extract_fields(text, expecting=[])
            collecting = False
        case.set_agent("intake", "running")
        case.stage = "collecting"
        before = {k for k in REQUIRED_FIELDS if case.intake.get(k)}

        fields = case.intake
        # a bare "yes" confirms the tag we suggested earlier
        if case.suggested_tag and _YES_RE.match(text):
            fields["baggage_tag"] = case.suggested_tag
            case.log(AGENT, "suggestion_confirmed", detail=f"tag {case.suggested_tag}")
        if case.suggested_tag and "baggage_tag" in rule:
            case.suggested_tag = None
        fields.update(rule)                              # newest user-provided value wins (corrections)
        for key, value in llm_out.items():               # LLM fills gaps only
            if key in REQUIRED_FIELDS and not fields.get(key):
                fields[key] = value
                case.log(AGENT, "llm_extracted", detail=f"{key} (verified against user text)")
        fields["issue"] = self._issue(text, llm_out, fields)

        notes = list(detect_invalid(text, rule).values())
        missing = [k for k in REQUIRED_FIELDS if not fields.get(k)]
        case.missing_fields = missing
        if missing:
            case.last_asked = missing
            case.intake_complete = False
            case.set_agent("intake", "waiting")
            case.reply = self._ask(case, missing, notes, had_data=bool(before))
            case.log(AGENT, "asked_for_missing", "warn", ", ".join(FIELD_LABELS[m] for m in missing))
            return
        case.last_asked = []
        case.intake_complete = True
        case.stage = "processing"
        case.set_agent("intake", "done")
        case.log(AGENT, "intake_complete", detail=f"all {len(REQUIRED_FIELDS)} required fields validated")

    # -------------------------------------------------------------- language
    @staticmethod
    def _update_language(case: CaseState, text: str) -> None:
        """Only flip the sticky ``case.language`` when the current message is Arabic,
        or is English *and* long enough to be a real judgement (so a bare reply like
        "BK1023" doesn't flip an Arabic conversation back to English)."""
        detected = detect_language(text)
        if detected == "ar":
            case.language = "ar"
        elif len(text.split()) >= 2:
            case.language = "en"

    # ---------------------------------------------------------------- intent
    def _intent(self, text: str, rule: dict, llm_out: dict, collecting: bool) -> str:
        if rule:
            return "baggage_issue"
        is_question = "?" in text or "؟" in text or bool(_QUESTION_START.match(text))
        if is_question and (_POLICY_RE.search(text) or AR_POLICY_RE.search(text)):
            return "policy_question"
        if _STATUS_RE.search(text) or _BAGGAGE_RE.search(text) or AR_BAGGAGE_ISSUE_RE.search(text):
            return "baggage_issue"
        if AR_POLICY_RE.search(text):
            return "policy_question"
        if llm_out.get("intent") in ("baggage_issue", "policy_question"):
            return llm_out["intent"]
        return "baggage_issue" if collecting else "other"

    @staticmethod
    def _issue(text: str, llm_out: dict, fields: dict) -> str:
        low = text.lower()
        if "delay" in low:
            return "delayed baggage"
        if re.search(r"lost|missing|not\s+arrive|didn'?t\s+arrive|did\s+not\s+arrive|never\s+arrived|hasn'?t\s+arrived", low):
            return "missing baggage"
        return fields.get("issue") or llm_out.get("issue") or "missing baggage"

    # ------------------------------------------------------------ LLM helper
    def _llm_extract(self, case: CaseState, text: str) -> dict:
        if not self.llm.enabled or not text:
            return {}
        prompt = (
            f'Passenger message:\n"""{text}"""\n\n'
            "Return JSON with exactly these keys:\n"
            '{"intent": "baggage_issue" | "policy_question" | "other", "passenger_name": string|null, '
            '"booking_reference": string|null, "flight_number": string|null, "flight_date": "YYYY-MM-DD"|null, '
            '"baggage_tag": string|null, "email": string|null, "issue": short string|null}')
        try:
            data = self.llm.complete_json(_SYSTEM_EXTRACT, prompt)
        except LLMError as exc:
            case.log(AGENT, "llm_fallback", "warn", f"extraction: LLM unavailable ({exc}) - rules only")
            case.llm_events.append({"agent": AGENT, "message": f"extraction failed: {exc}"})
            return {}
        return self._ground(data, text)

    @staticmethod
    def _ground(data: dict, text: str) -> dict:
        compact = re.sub(r"[\s\-]", "", text).upper()
        low = text.casefold()
        out: dict = {}
        if (v := norm_booking(data.get("booking_reference"))) and v in compact:
            out["booking_reference"] = v
        if (v := norm_tag(data.get("baggage_tag"))) and v in compact:
            out["baggage_tag"] = v
        if (v := norm_flight(data.get("flight_number"))) and v in compact:
            out["flight_number"] = v
        if (v := norm_email(data.get("email"))) and v in low:
            out["email"] = v
        if (v := norm_date(data.get("flight_date"))) and v[:4] in text:
            out["flight_date"] = v
        if (v := norm_name(data.get("passenger_name"))) and all(t.casefold() in low for t in v.split()):
            out["passenger_name"] = v
        if isinstance(data.get("issue"), str) and 0 < len(data["issue"]) <= 120:
            out["issue"] = data["issue"].strip()
        if data.get("intent") in ("baggage_issue", "policy_question", "other"):
            out["intent"] = data["intent"]
        return out

    # ------------------------------------------------------------- follow-up
    def _ask(self, case: CaseState, missing: list[str], notes: list[str], had_data: bool) -> str:
        lang = case.language

        def template() -> str:
            if lang == "ar":
                intro = "تمام، سجلت البيانات دي." if had_data else "آسفين إن شنطتك لسه ما وصلتش - هساعدك نحل الموضوع."
                lines = [intro]
                if notes:
                    lines.append(" ".join(notes))
                lines.append("عشان أدور على شنطتك محتاج منك:")
                lines += [f"- {FIELD_HINTS_AR[m]}" for m in missing]
                return "\n".join(lines)
            intro = ("Thanks, I've noted that." if had_data
                     else "I'm sorry your baggage hasn't arrived - I'll help you get this sorted out.")
            lines = [intro]
            if notes:
                lines.append(" ".join(notes))
            lines.append("To look up your bag I still need:")
            lines += [f"- {FIELD_HINTS[m]}" for m in missing]
            return "\n".join(lines)

        labels = FIELD_LABELS_AR if lang == "ar" else FIELD_LABELS
        hints = FIELD_HINTS_AR if lang == "ar" else FIELD_HINTS
        lang_name = "Arabic" if lang == "ar" else "English"
        prompt = (
            f"Write a short, friendly reply (max 90 words) IN {lang_name.upper()} asking the passenger for the "
            "missing items below. Use a bulleted list. Do not add any other questions or promises.\n"
            f"Missing items: {[hints[m] for m in missing]}\n"
            f"Extra note to include (may be empty): {' '.join(notes)}\n"
            f"Already provided (do not ask again): {[labels[k] for k in REQUIRED_FIELDS if case.intake.get(k)]}")
        req_words = _REQ_WORDS_AR if lang == "ar" else _REQ_WORDS
        text, _src = grounded_generate(
            self.llm, _SYSTEM_ASK, prompt, required=[req_words[m] for m in missing],
            allowed_ids=case.facts_ids, fallback=template, case=case, agent=AGENT, label="follow-up question")
        return text
