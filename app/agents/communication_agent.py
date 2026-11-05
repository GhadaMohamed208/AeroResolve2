"""Agent 4 - Communication / Follow-up Agent.

Builds the passenger email from the *actual workflow results* (tool data in the case state).
Hallucination protection:
  * subject and footer are deterministic;
  * the body may be LLM-written, but every required fact must appear and no unknown identifier may
    appear (``app.llm.guard``) - otherwise a deterministic template is used;
  * a passage of the policy PDF is retrieved (RAG) and used as the "policy reference".
"""
from __future__ import annotations

from app.llm.client import LLMClient
from app.llm.guard import grounded_generate
from app.tools.base import call_tool
from app.tools.claim_tools import save_email
from app.tools.data_tools import first_sentences, search_policy
from app.workflow.state import CaseState

AGENT = "Communication Agent"
FOOTER = {
    "en": ("\n\n-- \nThis message was produced by AeroResolve, a prototype demonstration that uses synthetic data "
           "and is not affiliated with any airline."),
    "ar": ("\n\n-- \nالرسالة دي اتبعتت من AeroResolve، نسخة تجريبية بتستخدم بيانات وهمية وغير تابعة لأي شركة طيران."),
}

_SYSTEM = (
    "You write passenger e-mails for an airline baggage service. Use ONLY the facts provided. Never invent "
    "identifiers, dates, statuses, locations, amounts or promises beyond the 'What happens next' text. "
    "Write plain text, under 170 words. Detect the language the passenger used (Arabic or English, given below) "
    "and write the ENTIRE email fluently in that language: in English start with 'Dear <first name>,' and end "
    "with 'Kind regards,' on its own line followed by 'AeroResolve Baggage Support (prototype)'; in Arabic use an "
    "equally warm, professional greeting and sign-off. Body only - no subject line.")

POLICY_QUERY = {
    "claim_registered": "delayed baggage interim expenses reimbursed receipts",
    "investigation_opened": "lost baggage investigation searches 21 days compensation",
    "duplicate_claim": "only one active claim can exist for each baggage tag",
    "delivered": "baggage already delivered no claim damaged report",
    "monitoring": "baggage in transit at airport monitored progress without opening claim",
    "manual_review": "escalation manual review human agent one business day",
}
NEXT_STEPS = {
    "claim_registered": "Our tracing team is working to forward your baggage to you. You will receive a status update at least every 48 hours until the claim is closed.",
    "investigation_opened": "Your case has been escalated to the Central Tracing team, which will search for your baggage. You will receive a status update at least every 48 hours until the claim is closed.",
    "duplicate_claim": "We already have an active claim for this baggage, so no new claim was created and you do not need to submit anything else.",
    "delivered": "Our records show that this baggage has already been delivered, so no claim has been created. If you believe it is damaged or items are missing, please report it at the airport baggage service office or reply with photographs.",
    "monitoring": "Your baggage has been located and is being monitored. No claim is needed at this stage and we will inform you as soon as its status changes.",
    "manual_review": "We could not match this baggage tag to our records, so your report has been escalated to a human agent, who will contact you within one business day.",
}
NEXT_STEPS_AR = {
    "claim_registered": "فريق التتبع شغال على إرجاع شنطتك. هتوصلك تحديثات كل 48 ساعة كحد أقصى لحد ما المطالبة تتقفل.",
    "investigation_opened": "حالتك اتحولت لفريق التتبع المركزي اللي هيدور على شنطتك. هتوصلك تحديثات كل 48 ساعة كحد أقصى لحد ما المطالبة تتقفل.",
    "duplicate_claim": "عندنا بالفعل مطالبة شغالة لنفس الشنطة، فمعملناش مطالبة جديدة ومش محتاج تعمل حاجة تانية.",
    "delivered": "سجلاتنا بتقول إن الشنطة دي وصلت بالفعل، فمعملناش مطالبة. لو فيه تلفيات أو حاجات ناقصة، بلّغ مكتب خدمة الشنط في المطار أو رد بصور.",
    "monitoring": "تم تحديد مكان شنطتك وبنتابعها. مفيش داعي لمطالبة دلوقتي وهنبلغك أول ما الحالة تتغير.",
    "manual_review": "معرفناش نربط تاج الشنطة ده بسجلاتنا، فبلاغك اتحول لموظف بيتابعه هيتواصل معاك خلال يوم عمل واحد.",
}


class CommunicationAgent:
    name = AGENT

    def __init__(self, db_path: str, llm: LLMClient, policy_path: str):
        self.db_path, self.llm, self.policy_path = db_path, llm, policy_path

    # ---------------------------------------------------------------- main
    def run(self, case: CaseState) -> None:
        case.set_agent("communication", "running")
        lang = case.language
        kind = self.kind(case)
        facts = self._facts(case, kind)
        policy = self._policy(case, kind)
        recipient = case.intake["email"]
        subject = self._subject(kind, facts, lang)

        required = [facts["baggage_tag"]] if kind != "manual_review" else [case.intake["baggage_tag"]]
        if facts.get("claim_id") and kind in ("claim_registered", "investigation_opened", "duplicate_claim"):
            required.append(facts["claim_id"])
        if facts.get("location"):
            required.append(facts["location"].split(" - ")[0])
        next_step = NEXT_STEPS_AR[kind] if lang == "ar" else NEXT_STEPS[kind]
        lang_name = "Arabic" if lang == "ar" else "English"
        prompt = (f"Facts (authoritative, from the airline database):\n{facts}\n\n"
                  f"Write the email IN {lang_name.upper()}.\n"
                  f"What happens next (convey this): {next_step}\n"
                  f"Policy reference (English source - may be paraphrased briefly, translate if writing in Arabic): "
                  f"{policy or 'n/a'}")
        body, source = grounded_generate(
            self.llm, _SYSTEM, prompt, required=required, allowed_ids=case.facts_ids,
            fallback=lambda: self._template(kind, facts, policy, lang), case=case, agent=AGENT, label="email")
        body = body.strip() + FOOTER.get(lang, FOOTER["en"])
        case.email = {"recipient": recipient, "subject": subject, "body": body, "kind": kind, "source": source,
                      "status": "Preview (not sent)"}
        saved = call_tool(case, AGENT, save_email, self.db_path, claim_id=facts.get("claim_id"),
                          baggage_tag=case.intake.get("baggage_tag"), recipient=recipient, subject=subject,
                          body=body)
        if not saved.ok:
            case.log(AGENT, "email_not_persisted", "warn", saved.message or saved.error_code or "")
        case.set_agent("communication", "done")
        case.log(AGENT, "email_prepared", detail=f"{kind} - written by {source}")

    # ------------------------------------------------------------- helpers
    @staticmethod
    def kind(case: CaseState) -> str:
        action = case.decision["action"]
        if action == "create_claim":
            return "claim_registered" if case.decision["claim_type"] == "Delayed Baggage" else "investigation_opened"
        return {"duplicate_prevented": "duplicate_claim", "no_claim_delivered": "delivered",
                "monitor": "monitoring", "manual_review": "manual_review"}[action]

    @staticmethod
    def _facts(case: CaseState, kind: str) -> dict:
        b = case.baggage or {}
        flight = case.flight or {}
        claim = case.claim or {}
        booking = case.booking or {}
        name = booking.get("passenger_name") or case.intake.get("passenger_name", "Passenger")
        return {
            "passenger_first_name": name.split()[0],
            "baggage_tag": b.get("baggage_tag") or case.intake.get("baggage_tag"),
            "status": b.get("status", "Unknown"),
            "location": b.get("last_known_location"),
            "last_updated": b.get("last_updated"),
            "flight": f"{booking.get('flight_number', '')} ({flight.get('origin', '?')} to {flight.get('destination', '?')}) on {booking.get('flight_date', '')}".strip(),
            "claim_id": claim.get("claim_id"),
            "claim_type": claim.get("claim_type"),
            "claim_status": claim.get("status"),
        }

    @staticmethod
    def _subject(kind: str, f: dict, lang: str = "en") -> str:
        if lang == "ar":
            return {
                "claim_registered": f"تم تسجيل مطالبة الشنطة {f['claim_id']}",
                "investigation_opened": f"تم فتح تحقيق للشنطة {f['claim_id']} - الحالة مُحوَّلة",
                "duplicate_claim": f"مطالبة شنطة موجودة بالفعل {f['claim_id']}",
                "delivered": f"تحديث الشنطة: {f['baggage_tag']} تم تسليمها",
                "monitoring": f"تحديث الشنطة: {f['baggage_tag']} - {f['status']}",
                "manual_review": "تم استلام بلاغك - محتاج مراجعة يدوية",
            }[kind]
        return {
            "claim_registered": f"Baggage Claim {f['claim_id']} Registered",
            "investigation_opened": f"Baggage Investigation {f['claim_id']} Opened - Escalated",
            "duplicate_claim": f"Existing Baggage Claim {f['claim_id']} On File",
            "delivered": f"Baggage Update: {f['baggage_tag']} Has Been Delivered",
            "monitoring": f"Baggage Update: {f['baggage_tag']} Is {f['status']}",
            "manual_review": "Baggage Report Received - Manual Review Required",
        }[kind]

    def _policy(self, case: CaseState, kind: str) -> str:
        res = call_tool(case, AGENT, search_policy, self.policy_path, query=POLICY_QUERY[kind], k=1)
        if not res.ok:
            case.log(AGENT, "policy_unavailable", "warn", res.message or "")
            return ""
        hit = res.data["hits"][0]
        return f"Policy reference (Section {hit['section']} - {hit['title']}): {first_sentences(hit['text'], 2)}"

    @staticmethod
    def _template(kind: str, f: dict, policy: str, lang: str = "en") -> str:
        if lang == "ar":
            lead = {
                "claim_registered": f"مطالبة الشنطة {f['claim_id']} اتسجلت بنجاح.",
                "investigation_opened": f"اتفتح تحقيق في شنطتك، مطالبة رقم {f['claim_id']}.",
                "duplicate_claim": f"لقينا مطالبة شغالة بالفعل، {f['claim_id']}، لنفس الشنطة.",
                "delivered": f"خبر كويس - شنطتك {f['baggage_tag']} مسجلة إنها اتسلمت.",
                "monitoring": f"راجعنا حالة شنطتك {f['baggage_tag']}.",
                "manual_review": f"استلمنا بلاغك عن تاج الشنطة {f['baggage_tag']}.",
            }[kind]
            lines = [f"عزيزي/عزيزتي {f['passenger_first_name']}،", "", lead, ""]
            if kind != "manual_review":
                lines += [f"تاج الشنطة: {f['baggage_tag']}", f"الرحلة: {f['flight']}",
                          f"الحالة الحالية: {f['status']}",
                          f"آخر موقع معروف: {f['location']} (آخر تحديث {f['last_updated']})"]
                if f.get("claim_id") and kind != "delivered":
                    lines.append(f"المطالبة: {f['claim_id']} ({f['claim_type']}، الحالة: {f['claim_status']})")
                lines.append("")
            lines += [f"الخطوة الجاية: {NEXT_STEPS_AR[kind]}", ""]
            if policy:
                lines += [policy, ""]
            lines += ["مع تحياتنا،", "فريق دعم شنط AeroResolve (نسخة تجريبية)"]
            return "\n".join(lines)

        lead = {
            "claim_registered": f"Your baggage claim {f['claim_id']} has been registered successfully.",
            "investigation_opened": f"A baggage investigation, claim {f['claim_id']}, has been opened for your baggage.",
            "duplicate_claim": f"We found an existing baggage claim, {f['claim_id']}, for your baggage.",
            "delivered": f"Good news - your baggage {f['baggage_tag']} is recorded as delivered.",
            "monitoring": f"We have checked the status of your baggage {f['baggage_tag']}.",
            "manual_review": f"We received your baggage report for tag {f['baggage_tag']}.",
        }[kind]
        lines = [f"Dear {f['passenger_first_name']},", "", lead, ""]
        if kind != "manual_review":
            lines += [f"Baggage tag: {f['baggage_tag']}", f"Flight: {f['flight']}",
                      f"Current status: {f['status']}",
                      f"Last known location: {f['location']} (updated {f['last_updated']})"]
            if f.get("claim_id") and kind != "delivered":
                lines.append(f"Claim: {f['claim_id']} ({f['claim_type']}, status: {f['claim_status']})")
            lines.append("")
        lines += [f"What happens next: {NEXT_STEPS[kind]}", ""]
        if policy:
            lines += [policy, ""]
        lines += ["Kind regards,", "AeroResolve Baggage Support (prototype)"]
        return "\n".join(lines)
