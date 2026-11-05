"""Deterministic validation and extraction helpers.

The Intake Agent combines LLM extraction with these rules. IDs are ALWAYS
validated here, so a model can never smuggle a malformed or invented value
into the workflow.
"""
from __future__ import annotations

import re
from datetime import date

_ARABIC_CHAR_RE = re.compile(r"[؀-ۿ]")

BOOKING_RE = re.compile(r"^BK\d{3,6}$")
FLIGHT_RE = re.compile(r"^[A-Z]{2}\d{2,4}$")
TAG_RE = re.compile(r"^BG\d{4,8}$")
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
_MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
                "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12})

NAME_STOP = {
    "and", "my", "the", "booking", "flight", "email", "e-mail", "tag", "baggage", "bag", "bags",
    "luggage", "suitcase", "from", "on", "at", "with", "i", "is", "was", "did", "has", "have", "had",
    "not", "missing", "lost", "delayed", "looking", "calling", "writing", "trying", "here", "flying",
    "travelling", "traveling", "reference", "ref", "number", "date", "yes", "no", "hello", "hi",
    "hey", "thanks", "thank", "please", "so", "very", "really", "just", "still", "waiting", "sorry",
    "upset", "worried", "contact", "phone", "reporting", "reported", "a", "an", "to", "for", "in",
    # Arabic connectors/context words that could otherwise be swept into a name match.
    "و", "في", "على", "عند", "من", "الى", "إلى", "مع", "عن", "لازم", "محتاج", "محتاجة", "عايز", "عايزة",
    "رقم", "الحجز", "حجز", "رحلة", "الرحلة", "طيران", "يوم", "تاريخ", "بريد", "ايميل", "إيميل", "البريد",
    "مشكلة", "شنطة", "شنطتي", "حقيبة", "حقيبتي", "حقائب", "أمتعة", "امتعة", "تاج", "رقم",
    "ضاعت", "ضاع", "مفقودة", "مفقود", "فقدت", "اتأخرت", "متأخرة", "وصلت", "وصل", "لسه", "ما", "مش",
    "انا", "أنا", "ايه", "إيه", "هو", "هي", "ده", "دي", "عندي", "لي", "له", "لها",
}

_EMAIL_FIND = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
_BOOKING_FIND = re.compile(r"\bBK[\s\-]?\d{3,6}\b", re.I)
_TAG_FIND = re.compile(r"\bBG[\s\-]?\d{4,8}\b", re.I)
_FLIGHT_CTX = re.compile(r"flight\s*(?:number|no\.?|#)?\s*[:\-]?\s*(?:is\s+)?([A-Za-z]{2})\s?-?(\d{2,4})\b", re.I)
_FLIGHT_FL = re.compile(r"\b(FL)\s?-?(\d{2,4})\b", re.I)
_NOT_AIRLINE = {"on", "to", "at", "in", "is", "no", "of", "by", "or", "my", "it", "as", "we", "so"}

_ISO = re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_DMY = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b")
_DMONY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([A-Za-z]{3,9})\.?,?\s+(\d{4})\b")
_MDY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")

# A "name token" is either a Latin word or an Arabic word (؀-ۿ), so a
# passenger can give their name in either script. The two scripts are never
# required to mix within one token, only across tokens in the same name.
_NAME_TOKEN = r"(?:[A-Za-z][A-Za-z'\-\.]*|[؀-ۿ][؀-ۿ'\-\.]*)"
_NAME_EXPLICIT = re.compile(
    r"(?:my\s+(?:full\s+)?name\s+is|(?:full\s+|passenger\s+)?name\s*(?:is\s*)?[:\-]|(?:full\s+|passenger\s+)?name\s+is"
    r"|passenger\s*:|اسمي|اسمى|الاسم\s*[:\-]?|انا\s+اسمي|أنا\s+اسمي)\s*"
    rf"({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,4}})", re.I)
_NAME_IMPLICIT = re.compile(
    rf"(?:\bi\s+am\b|\bi'm\b|\bim\b|\bthis\s+is\b|\bانا\b|\bأنا\b)\s+({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}})", re.I)
_NAME_ONLY = re.compile(rf"^{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}}$")


def _clean_id(value) -> str:
    return re.sub(r"[\s\-]", "", str(value or "")).upper()


def norm_booking(value) -> str | None:
    s = _clean_id(value)
    return s if BOOKING_RE.match(s) else None


def norm_flight(value) -> str | None:
    s = _clean_id(value)
    return s if FLIGHT_RE.match(s) else None


def norm_tag(value) -> str | None:
    s = _clean_id(value)
    return s if TAG_RE.match(s) else None


def norm_email(value) -> str | None:
    s = str(value or "").strip().strip(".,;:<>()")
    return s.lower() if EMAIL_RE.match(s) else None


def _mk_date(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def find_date(text) -> str | None:
    """Find the first valid date in free text and return it as YYYY-MM-DD."""
    text = str(text or "")
    if m := _ISO.search(text):
        return _mk_date(int(m[1]), int(m[2]), int(m[3]))
    if m := _DMY.search(text):
        d, mo = int(m[1]), int(m[2])
        if mo > 12 >= d:  # clearly month/day/year
            d, mo = mo, d
        return _mk_date(int(m[3]), mo, d)
    if m := _DMONY.search(text):
        mo = _MONTHS.get(m[2].lower())
        if mo:
            return _mk_date(int(m[3]), mo, int(m[1]))
    if m := _MDY.search(text):
        mo = _MONTHS.get(m[1].lower())
        if mo:
            return _mk_date(int(m[3]), mo, int(m[2]))
    return None


norm_date = find_date


_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-\.]*|[؀-ۿ][؀-ۿ'\-\.]*")


_PUNCT = ".,;:،؛؟!"  # Latin + Arabic sentence punctuation


def norm_name(value) -> str | None:
    tokens = [t for t in re.split(r"\s+", str(value or "").strip()) if t]
    tokens = [t.strip(_PUNCT) for t in tokens]
    tokens = [t for t in tokens if t]
    if not tokens or len(tokens) > 5:
        return None
    if any(t.lower() in NAME_STOP for t in tokens):
        return None
    if not all(_NAME_TOKEN_RE.fullmatch(t) for t in tokens):
        return None
    name = " ".join(tokens)
    is_arabic = any(_ARABIC_CHAR_RE.search(t) for t in tokens)
    if is_arabic:
        return name  # Arabic script has no case to normalise
    return name.title() if name.islower() or name.isupper() else name


def _trim_name(raw: str) -> str | None:
    kept = []
    for tok in raw.split():
        stripped = tok.strip(_PUNCT)
        if stripped.lower() in NAME_STOP:
            break
        kept.append(stripped)
    return norm_name(" ".join(kept)) if kept else None


def extract_name(text: str, expecting=()) -> str | None:
    for pattern in (_NAME_EXPLICIT, _NAME_IMPLICIT):
        if m := pattern.search(text):
            name = _trim_name(m.group(1))
            if name:
                return name
    if "passenger_name" in expecting:
        stripped = text.strip().strip(_PUNCT)
        if _NAME_ONLY.match(stripped):
            return norm_name(stripped)
    return None


def extract_fields(text: str, expecting=()) -> dict:
    """Rule-based extraction of every field we know how to validate."""
    found: dict = {}
    work = text or ""
    if m := _EMAIL_FIND.search(work):
        if email := norm_email(m.group(0)):
            found["email"] = email
        work = work.replace(m.group(0), " ")
    if m := _BOOKING_FIND.search(work):
        found["booking_reference"] = norm_booking(m.group(0))
        work = work.replace(m.group(0), " ")
    if m := _TAG_FIND.search(work):
        found["baggage_tag"] = norm_tag(m.group(0))
        work = work.replace(m.group(0), " ")
    flight = None
    for m in _FLIGHT_CTX.finditer(work):
        if m.group(1).lower() not in _NOT_AIRLINE:
            flight = norm_flight(m.group(1) + m.group(2))
            break
    if not flight and (m := _FLIGHT_FL.search(work)):
        flight = norm_flight(m.group(1) + m.group(2))
    if flight:
        found["flight_number"] = flight
    if d := find_date(work):
        found["flight_date"] = d
    if name := extract_name(work, expecting if not found else ()):
        found["passenger_name"] = name
    return {k: v for k, v in found.items() if v}


_INVALID_MSG = {
    "en": {
        "email": "That email address doesn't look valid.",
        "booking_reference": "That booking reference isn't in the expected format (e.g. BK1234).",
        "baggage_tag": "That baggage tag isn't in the expected format (e.g. BG12345).",
        "flight_date": "I couldn't read that date - please use YYYY-MM-DD (e.g. 2026-09-20).",
    },
    "ar": {
        "email": "البريد الإلكتروني ده شكله مش صح.",
        "booking_reference": "رقم الحجز ده مش بالصيغة المتوقعة (مثال: BK1234).",
        "baggage_tag": "رقم تاج الشنطة ده مش بالصيغة المتوقعة (مثال: BG12345).",
        "flight_date": "مقدرتش أقرا التاريخ ده - اكتبه بصيغة YYYY-MM-DD (مثال: 2026-09-20).",
    },
}


def detect_invalid(text: str, found: dict, lang: str = "en") -> dict:
    """Return {field: hint} for values the user *tried* to give but are malformed."""
    problems: dict = {}
    text = text or ""
    msg = _INVALID_MSG.get(lang, _INVALID_MSG["en"])
    if "@" in text and "email" not in found:
        problems["email"] = msg["email"]
    if "booking_reference" not in found and re.search(r"\bBK[\s\-]?\w+", text, re.I):
        problems["booking_reference"] = msg["booking_reference"]
    if "baggage_tag" not in found and re.search(r"\bBG[\s\-]?\w+", text, re.I):
        problems["baggage_tag"] = msg["baggage_tag"]
    if "flight_date" not in found and re.search(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b", text):
        problems["flight_date"] = msg["flight_date"]
    return problems
