"""Lightweight Arabic/English bilingual support.

This is intentionally small: a regex-based language detector, a handful of
Arabic intent keywords that mirror the English ones in the agents, and a
keyword-expansion table used before BM25 policy search (the policy index's
tokenizer only matches Latin a-z/0-9 - see ``app/rag/policy_rag.py`` - so an
Arabic query is expanded with English equivalents before it reaches BM25;
nothing is translated or embedded, this is a lookup table, not an NLP model).

No translation library, no embedding model and no vector database is used
here - see README "Arabic / bilingual support" for exactly what this does
and does not do.
"""
from __future__ import annotations

import re

_ARABIC_RE = re.compile(r"[؀-ۿ]")


def detect_language(text: str) -> str:
    """Return 'ar' if the text contains Arabic script, else 'en'.

    This only looks at the current message. Callers that want "sticky"
    language across a multi-turn conversation (so a short reply like a bare
    booking reference doesn't flip the conversation back to English) should
    only update their stored language when this returns a confident result
    for a message that has enough content to judge - see
    ``IntakeAgent._update_language``.
    """
    return "ar" if _ARABIC_RE.search(text or "") else "en"


# Arabic baggage-issue / tracking keywords, mirroring intake_agent._STATUS_RE
# and _BAGGAGE_RE. Deliberately short and broad rather than exhaustive.
AR_BAGGAGE_ISSUE_RE = re.compile(
    "شنط|حقيبة|حقائب|أمتعة|امتعه|أمتعتي|امتعتي|ضاع|ضايع|ضاعت|مفقود|فقدت|اتأخر|متأخر|تأخرت|"
    "لسه\\s*ما\\s*وصل|لم\\s*تصل|ما\\s*وصل|مش\\s*لاقي|مش\\s*لاقيه|مش\\s*موجود|مش\\s*موجوده|فين\\s*شنطتي|"
    "اين\\s*حقيبتي|وين\\s*شنطتي"
)

# Arabic policy-question keywords, mirroring intake_agent._POLICY_RE.
AR_POLICY_RE = re.compile(
    "سياس|سياسة|تعويض|استرجاع|استرداد|قانون|قواعد|كام\\s*يوم|اد[يى]\\s*مدة|مهلة|حد\\s*أقصى|"
    "مطالب|استحقاق|مصروفات|فواتير|ايصالات"
)

# Arabic -> English keyword equivalents, merged into the BM25 query so it
# can still match the (English-only) policy PDF tokens. Not exhaustive -
# only the terms a passenger is realistically likely to use.
AR_POLICY_SYNONYMS: dict[str, str] = {
    "تعويض": "compensation reimbursement",
    "تعويضات": "compensation reimbursement",
    "استرجاع": "refund reimbursement",
    "استرداد": "refund reimbursement",
    "تأخير": "delay delayed",
    "متأخر": "delayed",
    "مفقود": "lost missing not found",
    "ضاع": "lost missing",
    "ضايع": "lost missing",
    "كام يوم": "how many days",
    "مهلة": "deadline time limit",
    "حد أقصى": "limit maximum",
    "مطالبة": "claim",
    "مطالب": "claim entitled",
    "حقيبة": "baggage bag",
    "حقائب": "baggage bags",
    "شنطة": "baggage bag",
    "شنط": "baggage bags",
    "استلام": "delivery delivered",
    "تسليم": "delivery",
    "مصروفات": "expenses",
    "فواتير": "receipts expenses",
    "ايصالات": "receipts",
    "إيصالات": "receipts",
    "وثائق": "documents",
    "مستندات": "documents",
}


def expand_policy_query(text: str) -> str:
    """Append English keyword equivalents for Arabic terms found in *text*.

    The BM25 index only tokenizes a-z0-9, so an all-Arabic query would
    otherwise match nothing. Returns *text* unchanged when no known term is
    found (including when the text is already English).
    """
    extra = [eng for ar, eng in AR_POLICY_SYNONYMS.items() if ar in text]
    return f"{text} {' '.join(extra)}" if extra else text


# Bilingual field labels / hints, used by the Intake Agent's deterministic
# (non-LLM) "what's still missing" message so Rule-based mode also works in
# Arabic - the LLM path is instructed (via each agent's system prompt) to
# reply in whichever language the passenger used, but the template fallback
# must not silently fall back to English when the LLM is unavailable/rejected.
FIELD_LABELS_AR = {
    "passenger_name": "الاسم",
    "booking_reference": "رقم الحجز",
    "flight_number": "رقم الرحلة",
    "flight_date": "تاريخ الرحلة",
    "baggage_tag": "رقم تاج الشنطة",
    "email": "البريد الإلكتروني",
}
FIELD_HINTS_AR = {
    "passenger_name": "اسمك بالكامل زي ما هو مكتوب في الحجز",
    "booking_reference": "رقم الحجز (مثال: BK1234)",
    "flight_number": "رقم الرحلة (مثال: FL123)",
    "flight_date": "تاريخ الرحلة (YYYY-MM-DD، مثال: 2026-09-20)",
    "baggage_tag": "رقم تاج الشنطة (مثال: BG12345 - مطبوع على إيصال الشنطة)",
    "email": "بريد إلكتروني نقدر نبعتلك عليه التحديثات",
}


def field_labels(lang: str) -> dict:
    return FIELD_LABELS_AR if lang == "ar" else None  # caller falls back to FIELD_LABELS (en)


# UI strings for the passenger-facing Streamlit view (bilingual, since this
# is a UI copy concern, not something an LLM should decide at runtime).
UI_TEXT = {
    "ar": {
        "chip_track": "📌 أين شنطتي الآن؟",
        "chip_report": "📝 التبليغ عن شنطة مفقودة",
        "chip_delivery": "🚚 طلب توصيل لمقر الإقامة",
        "chat_placeholder": "اكتب مشكلة شنطتك هنا... (عربي أو English)",
        "stepper": [
            "تم استلام البلاغ",
            "جاري البحث في المطار",
            "تم التحديد ونقل الشنطة",
            "جاهزة للتسليم",
        ],
    },
    "en": {
        "chip_track": "📌 Where is my bag now?",
        "chip_report": "📝 Report missing baggage",
        "chip_delivery": "🚚 Request delivery to my address",
        "chat_placeholder": "Describe your baggage problem... (Arabic or English)",
        "stepper": [
            "Report received",
            "Searching at the airport",
            "Located & being transferred",
            "Ready for delivery",
        ],
    },
}


def stepper_stage(case) -> int:
    """Map the case's real, tool-retrieved status to one of the 4 passenger-facing
    stepper stages (0-3). This is a presentation-only mapping over facts the
    Tracking/Claim agents already put in ``case`` - it decides nothing new."""
    status = (case.baggage or {}).get("status")
    if case.stage in ("idle", "collecting", "verifying"):
        return 0
    if status == "Delivered":
        return 3
    if status == "At Airport":
        return 2
    if status in ("In Transit", "Delayed", "Not Found", "Under Investigation"):
        return 1
    return 1 if case.stage in ("processing", "completed", "escalated") else 0
