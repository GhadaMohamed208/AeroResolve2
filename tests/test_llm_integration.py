"""Proves the LLM integration is genuinely wired into the workflow - not just present in
the code but never exercised.

This sandbox has no network access to a real LLM provider (Ollama/Groq/OpenAI/Anthropic
all require outbound network calls this environment cannot make), so these tests use a
``FakeProvider`` - a real ``Provider`` subclass, going through the exact same
``LLMClient`` / ``grounded_generate`` machinery every agent uses - to prove, offline and
deterministically, that:

  1. every agent that is supposed to call the LLM actually does (not decorative code),
  2. correctly-grounded LLM output is accepted and used verbatim,
  3. hallucinated output (an unauthorised identifier, or a missing required fact) is
     detected and rejected, falling back to the deterministic template instead.

If you configure a real provider (``LLM_PROVIDER=ollama`` or ``openai``), the exact same
``grounded_generate`` call sites are exercised - only the HTTP transport underneath
``Provider.chat()`` differs.
"""
from __future__ import annotations

from app.agents.claim_agent import ClaimAgent
from app.agents.communication_agent import CommunicationAgent
from app.agents.intake_agent import IntakeAgent
from app.agents.tracking_agent import TrackingAgent
from app.llm.client import LLMClient, Provider
from app.llm.guard import check_grounding, grounded_generate
from app.workflow.orchestrator import Orchestrator
from app.workflow.state import CaseState
from app.workflow.summary import CaseSummarizer


class FakeProvider(Provider):
    """A real ``Provider`` that returns a scripted answer instead of calling a network API."""
    name = "fake"
    model = "fake-1"

    def __init__(self, reply: str | Exception):
        self._reply = reply

    def chat(self, system, prompt, json_mode=False):
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply

    def health(self):
        return True, "fake provider (test double)"


def fake_llm(reply: str | Exception) -> LLMClient:
    client = LLMClient(FakeProvider(reply))
    assert client.enabled, "LLMClient must report enabled=True for a non-rules provider"
    return client


# --------------------------------------------------------- guard.grounded_generate itself
def test_grounded_generate_calls_the_llm_and_uses_valid_output():
    llm = fake_llm("Your bag BG77821 is Delayed.")
    text, source = grounded_generate(
        llm, "sys", "prompt", required=["BG77821", "Delayed"], allowed_ids=["BG77821"],
        fallback=lambda: "FALLBACK SHOULD NOT BE USED")
    assert source == "llm"
    assert text == "Your bag BG77821 is Delayed."


def test_grounded_generate_rejects_unauthorised_identifier():
    """The LLM invents a claim ID that was never given to it -> must be rejected."""
    llm = fake_llm("Your claim CLM-9999 is confirmed.")  # CLM-9999 is not in allowed_ids
    text, source = grounded_generate(
        llm, "sys", "prompt", required=[], allowed_ids=["BG77821"],
        fallback=lambda: "SAFE TEMPLATE")
    assert source == "fallback:guard"
    assert text == "SAFE TEMPLATE"


def test_grounded_generate_rejects_missing_required_fact():
    llm = fake_llm("Everything is fine.")  # never mentions the required status
    text, source = grounded_generate(
        llm, "sys", "prompt", required=["Delayed"], allowed_ids=[], fallback=lambda: "SAFE TEMPLATE")
    assert source == "fallback:guard"
    assert text == "SAFE TEMPLATE"


def test_grounded_generate_falls_back_on_llm_transport_error():
    from app.llm.client import LLMError
    llm = fake_llm(LLMError("network unreachable"))
    text, source = grounded_generate(
        llm, "sys", "prompt", required=[], allowed_ids=[], fallback=lambda: "SAFE TEMPLATE")
    assert source == "fallback:llm_error"
    assert text == "SAFE TEMPLATE"


def test_check_grounding_directly():
    ok, problems = check_grounding("Bag BG1 is Delayed", required=["BG1", "Delayed"], allowed_ids=["BG1"])
    assert ok and not problems
    ok, problems = check_grounding("Bag BG1 is Delayed, see also CLM-1", required=["BG1"], allowed_ids=["BG1"])
    assert not ok and any("CLM-1" in p for p in problems)


# ---------------------------------------------------------------- Agent 1: Intake Agent
def test_intake_agent_llm_extraction_is_grounded_against_user_text(services):
    """The passenger's name is written in a way the regex rules do NOT catch (no 'I am' /
    'name is' prefix), so it can ONLY be picked up via the LLM path - proving the LLM
    extraction call is real and its output is actually merged into the case."""
    llm = fake_llm('{"intent":"baggage_issue","passenger_name":"Ghada Mohamed",'
                   '"booking_reference":null,"flight_number":null,"flight_date":null,'
                   '"baggage_tag":null,"email":null,"issue":"missing baggage"}')
    agent = IntakeAgent(llm)
    case = CaseState()
    case.user_message = ("Ghada Mohamed here - booking BK1023, flight FL123 on 2026-09-20, "
                          "tag BG77821, email ghada@example.com")
    agent.run(case)
    assert case.intake.get("passenger_name") == "Ghada Mohamed"  # only the LLM path could have found this
    assert case.intake_complete is True


def test_intake_agent_llm_cannot_smuggle_a_name_not_in_the_text(services):
    """The LLM invents a name that never appears in the passenger's message -> must be rejected."""
    llm = fake_llm('{"intent":"baggage_issue","passenger_name":"Someone Else",'
                   '"booking_reference":null,"flight_number":null,"flight_date":null,'
                   '"baggage_tag":null,"email":null,"issue":null}')
    agent = IntakeAgent(llm)
    case = CaseState()
    case.user_message = "booking BK1023, flight FL123 on 2026-09-20, tag BG77821, email ghada@example.com"
    agent.run(case)
    assert case.intake.get("passenger_name") is None  # rejected: not present in the user's text
    assert "passenger_name" in case.missing_fields


# ---------------------------------------------------------------- Agent 3: Claim Agent
def test_claim_agent_analysis_is_written_by_llm_when_grounded(services):
    llm = fake_llm("Bag BG77821 is Delayed so a claim was opened.")
    tracking = TrackingAgent(services.db_path)
    claim_agent = ClaimAgent(services.db_path, llm, services.exports_dir)
    case = CaseState()
    case.intake = {"passenger_name": "Ghada Mohamed", "booking_reference": "BK1023",
                   "flight_number": "FL123", "flight_date": "2026-09-20", "baggage_tag": "BG77821",
                   "email": "ghada@example.com", "issue": "missing baggage"}
    tracking.run(case)
    claim_agent.decide(case)
    assert case.analysis == "Bag BG77821 is Delayed so a claim was opened."


def test_claim_agent_analysis_falls_back_when_llm_hallucinates_a_claim_id(services):
    llm = fake_llm("This links to claim CLM-4242 automatically.")  # not a real, allowed claim id yet
    tracking = TrackingAgent(services.db_path)
    claim_agent = ClaimAgent(services.db_path, llm, services.exports_dir)
    case = CaseState()
    case.intake = {"passenger_name": "Ghada Mohamed", "booking_reference": "BK1023",
                   "flight_number": "FL123", "flight_date": "2026-09-20", "baggage_tag": "BG77821",
                   "email": "ghada@example.com", "issue": "missing baggage"}
    tracking.run(case)
    claim_agent.decide(case)
    assert "CLM-4242" not in case.analysis
    assert "BG77821" in case.analysis  # deterministic fallback still states the real facts


# ------------------------------------------------------------ Agent 4: Communication Agent
def _completed_delayed_case(services, llm):
    tracking = TrackingAgent(services.db_path)
    claim_agent = ClaimAgent(services.db_path, llm, services.exports_dir)
    case = CaseState()
    case.intake = {"passenger_name": "Ghada Mohamed", "booking_reference": "BK1023",
                   "flight_number": "FL123", "flight_date": "2026-09-20", "baggage_tag": "BG77821",
                   "email": "ghada@example.com", "issue": "missing baggage"}
    tracking.run(case)
    claim_agent.decide(case)
    claim_agent.create_claim(case)
    return case


def test_communication_agent_email_body_written_by_llm_when_grounded(services):
    case = _completed_delayed_case(services, fake_llm("placeholder"))
    claim_id = case.claim["claim_id"]
    llm = fake_llm(f"Dear Ghada,\n\nYour claim {claim_id} for bag BG77821 (Delayed, last seen at "
                   f"Cairo Airport) is confirmed.\n\nKind regards,\nAeroResolve Baggage Support (prototype)")
    comm = CommunicationAgent(services.db_path, llm, services.policy_path)
    comm.run(case)
    assert case.email["source"] == "llm"
    assert claim_id in case.email["body"]


def test_communication_agent_falls_back_when_llm_invents_a_location(services):
    case = _completed_delayed_case(services, fake_llm("placeholder"))
    claim_id = case.claim["claim_id"]
    # The LLM invents a location never given to it - the guard must catch that the
    # REQUIRED real location ("Cairo Airport") is missing and reject this output.
    llm = fake_llm(f"Dear Ghada,\n\nYour claim {claim_id} is confirmed, bag last seen in Tokyo.\n\n"
                   f"Kind regards,\nAeroResolve Baggage Support (prototype)")
    comm = CommunicationAgent(services.db_path, llm, services.policy_path)
    comm.run(case)
    assert case.email["source"] != "llm"
    assert "Cairo Airport" in case.email["body"]  # deterministic template used instead


# --------------------------------------------------------------------- Case Summarizer
def test_case_summary_narrative_written_by_llm_when_grounded(services):
    case = _completed_delayed_case(services, fake_llm("placeholder"))
    claim_id = case.claim["claim_id"]
    llm = fake_llm(f"Bag BG77821 is Delayed and claim {claim_id} was opened for the passenger.")
    summarizer = CaseSummarizer(type("S", (), {"llm": llm, "exports_dir": services.exports_dir})())
    summarizer.run(case)
    assert case.summary["narrative_source"] == "llm"
    assert claim_id in case.summary["narrative"]


# --------------------------------------------------------------------- Policy RAG + LLM
def test_policy_answer_uses_retrieval_then_llm_generation(services):
    """Proves the RAG call site is a genuine retrieve-then-generate pipeline: BM25 finds the
    section, and (when a provider is configured) the LLM writes the answer from it."""
    llm = fake_llm("Interim expenses are reimbursed up to the daily policy limit with receipts.")
    orch = Orchestrator(type("S", (), {
        "llm": llm, "db_path": services.db_path, "policy_path": services.policy_path,
        "exports_dir": services.exports_dir, "data_dir": services.data_dir})())
    case = orch.new_case()
    orch.run_turn(case, "How much are interim expenses reimbursed for delayed baggage?")
    assert case.policy_answer is not None
    assert case.policy_answer["hits"], "BM25 retrieval must return at least one policy section"
    assert "Interim expenses are reimbursed" in case.reply  # the LLM-generated answer was used
