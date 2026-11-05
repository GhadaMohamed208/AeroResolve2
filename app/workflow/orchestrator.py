r"""Workflow orchestration (LangGraph).

    User
      -> Intake Agent ---- incomplete / invalid ------------------> ask user (END)
      -> Tracking Agent -- mismatch / unknown tag --------------> ask user (END)
                         \- unknown tag twice ------------------> Escalation (manual review)
      -> Status Decision (Claim Agent)
            Delayed / Not Found / Under Investigation ----------> Claim creation (+ follow-up record)
            In Transit / At Airport ----------------------------> Follow-up (monitoring)
            Delivered / duplicate ------------------------------> (no action)
      -> Communication Agent (email preview, persisted)
      -> Case Summary (+ PDF report) -> final reply

The orchestrator owns: execution order, shared state (``CaseState``), conditional branching,
per-node error handling and the final response. Any exception inside a node is caught, logged to the
trace and turned into a friendly message - the app never crashes.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.claim_agent import ClaimAgent
from app.agents.communication_agent import CommunicationAgent
from app.agents.intake_agent import FIELD_HINTS, IntakeAgent
from app.agents.tracking_agent import TrackingAgent
from app.i18n import FIELD_LABELS_AR, expand_policy_query
from app.llm.guard import grounded_generate
from app.services import Services
from app.tools.base import call_tool
from app.tools.data_tools import first_sentences, search_policy
from app.workflow.state import FIELD_LABELS, TERMINAL_STAGES, CaseState
from app.workflow.summary import CaseSummarizer

log = logging.getLogger(__name__)


class GraphState(TypedDict):
    case: CaseState


class Orchestrator:
    def __init__(self, services: Services):
        self.s = services
        self.intake = IntakeAgent(services.llm)
        self.tracking = TrackingAgent(services.db_path)
        self.claim = ClaimAgent(services.db_path, services.llm, services.exports_dir)
        self.comm = CommunicationAgent(services.db_path, services.llm, services.policy_path)
        self.summarizer = CaseSummarizer(services)
        self.graph = self._build_graph()

    # ------------------------------------------------------------ public API
    def new_case(self) -> CaseState:
        return CaseState()

    def run_turn(self, case: CaseState, user_message: str) -> CaseState:
        """Process one user message; returns the (mutated) case with ``case.reply`` filled in."""
        start = time.perf_counter()
        message = (user_message or "").strip()
        case.messages.append({"role": "user", "content": message})
        case.begin_turn(message)
        try:
            if not message:
                case.reply = "Please describe your baggage problem, for example: *My baggage did not arrive.*"
            else:
                self.graph.invoke({"case": case})
        except Exception as exc:  # noqa: BLE001 - never let the UI crash
            log.exception("workflow failure")
            case.errors.append(f"workflow: {exc}")
            case.stage = "error"
            case.reply = "Sorry - something unexpected went wrong. Your details are kept; please try again."
        case.last_run_ms = round((time.perf_counter() - start) * 1000, 1)
        case.messages.append({"role": "assistant", "content": case.reply})
        return case

    def run_conversation(self, messages: list[str], case: CaseState | None = None) -> CaseState:
        case = case or self.new_case()
        for m in messages:
            self.run_turn(case, m)
        return case

    # ------------------------------------------------------------- the graph
    def _build_graph(self):
        g = StateGraph(GraphState)
        g.add_node("intake", self._node("intake", self._intake, "intake"))
        g.add_node("policy", self._node("policy", self._policy_answer))
        g.add_node("tracking", self._node("tracking", self.tracking.run, "tracking"))
        g.add_node("status_decision", self._node("status_decision", self.claim.decide, "claim"))
        g.add_node("claim_creation", self._node("claim_creation", self.claim.create_claim, "claim"))
        g.add_node("follow_up", self._node("follow_up", self.claim.follow_up, "claim"))
        g.add_node("escalation", self._node("escalation", self.claim.escalate, "claim"))
        g.add_node("communication", self._node("communication", self.comm.run, "communication"))
        g.add_node("case_summary", self._node("case_summary", self._summarise))

        g.add_edge(START, "intake")
        g.add_conditional_edges("intake", self._after_intake,
                                {"policy": "policy", "tracking": "tracking", "end": END})
        g.add_conditional_edges("tracking", self._after_tracking,
                                {"decide": "status_decision", "escalate": "escalation", "end": END})
        g.add_conditional_edges("status_decision", self._after_decision,
                                {"create_claim": "claim_creation", "follow_up": "follow_up",
                                 "communicate": "communication", "end": END})
        g.add_conditional_edges("claim_creation", lambda s: "end" if s["case"].halt else "communicate",
                                {"communicate": "communication", "end": END})
        g.add_edge("follow_up", "communication")
        g.add_edge("escalation", "communication")
        g.add_conditional_edges("communication", lambda s: "end" if s["case"].halt else "summary",
                                {"summary": "case_summary", "end": END})
        g.add_edge("case_summary", END)
        g.add_edge("policy", END)
        return g.compile()

    # -------------------------------------------------------------- routers
    @staticmethod
    def _after_intake(state: GraphState) -> str:
        case = state["case"]
        if case.halt:
            return "end"
        if case.intent == "policy_question":
            return "policy"
        if case.intent == "baggage_issue" and case.intake_complete:
            return "tracking"
        return "end"

    @staticmethod
    def _after_tracking(state: GraphState) -> str:
        case = state["case"]
        if case.halt:
            return "end"
        return {"ok": "decide", "escalate": "escalate"}.get(case.tracking_outcome or "", "end")

    @staticmethod
    def _after_decision(state: GraphState) -> str:
        case = state["case"]
        if case.halt:
            return "end"
        action = case.decision["action"]
        if action == "create_claim":
            return "create_claim"
        if action in ("monitor", "manual_review"):
            return "follow_up"
        return "communicate"

    # ---------------------------------------------------------------- nodes
    def _node(self, name: str, fn: Callable[[CaseState], None], agent_key: str | None = None):
        def run(state: GraphState) -> GraphState:
            case = state["case"]
            t0 = time.perf_counter()
            try:
                fn(case)
                case.log(name, "node_finished", detail=f"{(time.perf_counter() - t0) * 1000:.0f} ms")
            except Exception as exc:  # noqa: BLE001
                log.exception("node %s failed", name)
                case.errors.append(f"{name}: {type(exc).__name__}: {exc}")
                case.halt = True
                case.stage = "error"
                if agent_key:
                    case.set_agent(agent_key, "error")
                case.log(name, "exception", "error", f"{type(exc).__name__}: {exc}")
                case.reply = ("Something went wrong while processing your case. Your details are saved - "
                              "please send **retry**, or contact support if it keeps happening.")
            return {"case": case}
        return run

    def _intake(self, case: CaseState) -> None:
        self.intake.run(case)
        if case.intent == "other":
            if case.language == "ar":
                case.reply = ("أهلاً! أنا مساعد AeroResolve لشنط السفر. قولّي حصل ايه مع شنطتك - مثلاً "
                              "*\"شنطتي لسه ما وصلتش\"* - أو اسألني سؤال عن سياسة الشنط.")
            else:
                case.reply = ("Hello! I'm the AeroResolve baggage assistant. Tell me what happened to your baggage - "
                              "for example *\"My baggage did not arrive\"* - or ask a question about the baggage policy.")

    def _summarise(self, case: CaseState) -> None:
        self.summarizer.run(case)
        case.stage = "escalated" if case.decision["action"] == "manual_review" else "completed"
        case.reply = self.final_reply(case)

    def _policy_answer(self, case: CaseState) -> None:
        q = case.user_message
        # The policy PDF and its BM25 index are English-only (see app/rag/policy_rag.py
        # and README "Arabic / bilingual support"). An Arabic query is expanded with
        # English keyword equivalents (app.i18n.expand_policy_query - a lookup table,
        # not an embedding model) so it can still match the English index; nothing in
        # the document itself is translated.
        search_q = expand_policy_query(q) if case.language == "ar" else q
        res = call_tool(case, "Policy Retrieval", search_policy, self.s.policy_path, query=search_q, k=2)
        if not res.ok:
            case.reply = (
                ("معلش، معنديش إجابة لده في سياسة الشنط دلوقتي. موظف حقيقي ممكن يساعدك في السؤال ده. "
                 if case.language == "ar" else
                 "I couldn't find that in the baggage policy. A human agent can help with that question. ")
                + self._resume_hint(case))
            case.log("Policy Retrieval", "no_result", "warn", res.error_code or "")
            return
        hits = res.data["hits"]
        case.policy_answer = {"question": q, "hits": hits}
        context = "\n".join(f"[Section {h['section']} - {h['title']}] {h['text']}" for h in hits)
        lang_name = "Arabic" if case.language == "ar" else "English"
        prompt = (f"Policy excerpts (English source document):\n{context}\n\nPassenger question: {q}\n"
                  f"Answer in at most 3 sentences, IN {lang_name.upper()}, using only the excerpts.")
        if case.language == "ar":
            fallback = lambda: (f"حسب سياسة الشنط (قسم {hits[0]['section']} - {hits[0]['title']}): "
                                f"{first_sentences(hits[0]['text'], 3)}")
        else:
            fallback = lambda: (f"According to the baggage policy (Section {hits[0]['section']} - "
                                f"{hits[0]['title']}): {first_sentences(hits[0]['text'], 3)}")
        text, src = grounded_generate(
            self.s.llm, "You answer airline baggage-policy questions using ONLY the provided excerpts, which are in "
            "English. If they do not contain the answer, say so. Never mention identifiers. Detect the language of "
            "the passenger's question (Arabic or English) and write your answer fluently in that same language, "
            "translating/paraphrasing the English excerpts as needed - never answer in the wrong language.",
            prompt, required=[], allowed_ids=[],
            fallback=fallback, case=case, agent="Policy Retrieval", label="policy answer")
        case.reply = text + self._resume_hint(case)
        case.log("Policy Retrieval", "answered", detail=f"sections {[h['section'] for h in hits]} - written by {src}")

    @staticmethod
    def _resume_hint(case: CaseState) -> str:
        if case.stage in TERMINAL_STAGES or not case.missing_fields:
            return ""
        if case.language == "ar":
            wanted = "، ".join(FIELD_LABELS_AR[m] for m in case.missing_fields)
            return f"\n\nلما تكون جاهز، لسه محتاج منك {wanted} عشان أكمل معاك في حالة الشنطة."
        wanted = ", ".join(FIELD_LABELS[m] for m in case.missing_fields)
        return f"\n\nWhenever you're ready, I still need your {wanted} to continue with your baggage case."

    # ------------------------------------------------------------ final text
    @staticmethod
    def final_reply(case: CaseState) -> str:
        """Deterministic, tool-grounded answer shown in the chat. Bilingual (Arabic/English) -
        both branches state exactly the same tool-retrieved facts, only the wording differs."""
        d, b, c = case.decision, case.baggage, case.claim
        action = d["action"]
        if case.language == "ar":
            mail = f"إيميل تأكيد لـ **{case.intake['email']}** جاهز في تبويب *معاينة الإيميل*."
            head = ""
            if b:
                head = (f"لقيت شنطتك **{b['baggage_tag']}**: الحالة **{b['status']}**، آخر موقع معروف "
                        f"**{b['last_known_location']}** (آخر تحديث {b['last_updated']}).\n\n")
            if action == "create_claim":
                what = (f"بما إنها {b['status']}، سجلت مطالبة **{c['claim_id']}** "
                        f"({c['claim_type']}، الحالة **{c['status']}**) وعملت متابعة لها.")
            elif action == "duplicate_prevented":
                what = (f"فيه مطالبة شغالة بالفعل، **{c['claim_id']}** (الحالة **{c['status']}**)، لنفس الشنطة - "
                        f"فمعملتش مطالبة تانية.")
            elif action == "no_claim_delivered":
                what = "الشنطة ده مسجلة إنها **اتسلمت**، فمعملناش مطالبة."
            elif action == "monitor":
                what = f"تم تحديد مكانها وهي **{b['status']}**. مفيش داعي لمطالبة؛ هتابع الحالة وأبلغك لما تتغير."
            else:
                what = (f"معرفتش أربط تاج **{case.intake.get('baggage_tag')}** بأي سجل بعد محاولتين، "
                        f"فحولت بلاغك لموظف بشري **للمراجعة اليدوية**.")
            return f"{head}{what}\n\n{mail}"

        mail = f"A confirmation email to **{case.intake['email']}** is ready in the *Email Preview* tab."
        head = ""
        if b:
            head = (f"I found your baggage **{b['baggage_tag']}**: status **{b['status']}**, last known location "
                    f"**{b['last_known_location']}** (updated {b['last_updated']}).\n\n")
        if action == "create_claim":
            what = (f"Because it is {b['status'].lower()}, I registered claim **{c['claim_id']}** "
                    f"({c['claim_type']}, status **{c['status']}**) and scheduled a follow-up.")
        elif action == "duplicate_prevented":
            what = (f"There is already an active claim, **{c['claim_id']}** (status **{c['status']}**), for this "
                    f"baggage - so I did **not** create a duplicate.")
        elif action == "no_claim_delivered":
            what = "This baggage is recorded as **delivered**, so no claim was created."
        elif action == "monitor":
            what = f"It has been located and is **{b['status']}**. No claim is needed; I set up monitoring so you are told when it changes."
        else:
            what = (f"I couldn't match tag **{case.intake.get('baggage_tag')}** to any record after two attempts, "
                    f"so I escalated your report to a human agent for **manual review**.")
        return f"{head}{what}\n\n{mail}"
