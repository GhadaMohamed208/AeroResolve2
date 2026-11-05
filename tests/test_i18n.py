"""Tests for Arabic/English bilingual support (app/i18n.py + its use across
the agents, validation and orchestrator).

These exist because the bilingual behaviour is a real, user-facing
requirement ("the assistant must understand Arabic and reply in Arabic") and
must be proven to actually work end-to-end with the default, no-LLM
Rule-based provider - not just asserted in documentation.
"""
from __future__ import annotations

from app.i18n import detect_language, expand_policy_query, stepper_stage
from app.validation import extract_fields, norm_name
from app.workflow.state import CaseState


# --------------------------------------------------------------- detection
def test_detect_language_arabic_script():
    assert detect_language("شنطتي ضاعت") == "ar"


def test_detect_language_english():
    assert detect_language("My baggage is missing") == "en"


def test_detect_language_latin_ids_inside_arabic_sentence_stays_arabic():
    # A passenger writing in Arabic but quoting Latin-script IDs (booking
    # refs, tags, emails) should still be detected as Arabic overall.
    assert detect_language("رقم الحجز BK1023 وتاج الشنطة BG77821") == "ar"


# --------------------------------------------------------------- validation
def test_norm_name_accepts_arabic_script():
    assert norm_name("غادة محمد") == "غادة محمد"


def test_norm_name_still_accepts_latin_script():
    assert norm_name("ghada mohamed") == "Ghada Mohamed"


def test_extract_fields_from_arabic_message():
    text = ("اسمي غادة محمد، رقم الحجز BK1023، رقم الرحلة FL123، "
            "تاريخ الرحلة 2026-09-20، تاج الشنطة BG77821، إيميلي ghada@example.com")
    found = extract_fields(text)
    assert found["booking_reference"] == "BK1023"
    assert found["flight_number"] == "FL123"
    assert found["baggage_tag"] == "BG77821"
    assert found["flight_date"] == "2026-09-20"
    assert found["email"] == "ghada@example.com"
    assert found["passenger_name"] == "غادة محمد"


# --------------------------------------------------------- policy expansion
def test_expand_policy_query_adds_english_synonyms_for_arabic_terms():
    expanded = expand_policy_query("كام يوم مهلة التعويض عن التأخير؟")
    assert "compensation" in expanded
    assert "deadline" in expanded or "limit" in expanded


def test_expand_policy_query_leaves_english_text_unchanged():
    assert expand_policy_query("How many days for compensation?") == "How many days for compensation?"


# --------------------------------------------------------------- UI stepper
def test_stepper_stage_idle_is_zero():
    case = CaseState()
    assert stepper_stage(case) == 0


def test_stepper_stage_delivered_is_last():
    case = CaseState(stage="completed", baggage={"status": "Delivered"})
    assert stepper_stage(case) == 3


def test_stepper_stage_at_airport_is_located():
    case = CaseState(stage="completed", baggage={"status": "At Airport"})
    assert stepper_stage(case) == 2


def test_stepper_stage_delayed_is_searching():
    case = CaseState(stage="completed", baggage={"status": "Delayed"})
    assert stepper_stage(case) == 1


# ------------------------------------------------------- end-to-end (rules)
def test_full_arabic_conversation_end_to_end_rule_based(orchestrator):
    """A whole Arabic baggage report, with zero LLM calls (RulesProvider),
    should be understood, resolved and answered fluently in Arabic."""
    case = orchestrator.new_case()
    msg = ("شنطتي ضاعت، اسمي Ghada Mohamed، رقم الحجز BK1023، رقم الرحلة FL123، "
           "تاريخ الرحلة 2026-09-20، تاج الشنطة BG77821، إيميلي ghada@example.com")
    orchestrator.run_turn(case, msg)

    assert case.language == "ar"
    assert case.stage == "completed"
    assert case.baggage is not None and case.baggage["baggage_tag"] == "BG77821"
    # The reply must actually be written in Arabic, not just internally
    # tagged as such - check for Arabic script in the visible text.
    assert detect_language(case.reply) == "ar"


def test_arabic_policy_question_is_answered_in_arabic(orchestrator):
    case = orchestrator.new_case()
    orchestrator.run_turn(case, "كام يوم المهلة للمطالبة بتعويض عن تأخير الشنطة؟")
    assert case.language == "ar"
    assert case.reply
    assert detect_language(case.reply) == "ar"
