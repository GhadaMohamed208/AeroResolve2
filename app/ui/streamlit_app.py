"""AeroResolve - Streamlit web interface.

This is the passenger-facing view: a clean "airline support portal" screen.
Internal/developer details (LLM provider config, agent names, raw tool-call
trace, the LangGraph diagram, guided test-demo buttons) are deliberately NOT
shown here - a passenger only needs to see their own case status. For a
developer/demo view of those internals, read the code directly or see the
project README; this file intentionally does not expose a hidden "dev mode"
toggle, since that would just re-introduce the same clutter for end users.
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from app.database.db import reset_db
from app.i18n import UI_TEXT, stepper_stage
from app.services import build_services
from app.workflow.orchestrator import Orchestrator
from app.workflow.state import CaseState

STATUS_COLORS = {
    "Delivered": "#16a34a", "Delayed": "#d97706", "In Transit": "#2563eb", "At Airport": "#0d9488",
    "Not Found": "#dc2626", "Under Investigation": "#7c3aed", "Unknown": "#64748b",
}
STATUS_LABEL_AR = {
    "Delivered": "تم التسليم", "Delayed": "متأخرة", "In Transit": "في الطريق", "At Airport": "في المطار",
    "Not Found": "غير موجودة", "Under Investigation": "تحت التحقيق", "Unknown": "غير معروفة",
}

# Passenger-facing quick-action chips (replace the old developer "guided
# demo" buttons). Each sends a natural-language message, in whichever UI
# language is selected, into the same chat the passenger can type into -
# nothing here is a special/fake code path, it is exactly what typing the
# same sentence would do.
CHIPS = [
    {"key": "track",
     "label": {"ar": "📌 أين شنطتي الآن؟", "en": "📌 Where's my bag now?"},
     "msg": {"ar": "فين شنطتي دلوقتي؟", "en": "Where is my baggage right now?"}},
    {"key": "report",
     "label": {"ar": "📝 التبليغ عن شنطة مفقودة", "en": "📝 Report a missing bag"},
     "msg": {"ar": "عايز أبلغ عن شنطة مفقودة.", "en": "I'd like to report a missing bag."}},
    {"key": "delivery",
     "label": {"ar": "🚚 طلب توصيل لمقر الإقامة", "en": "🚚 Request delivery to my residence"},
     "msg": {"ar": "شنطتي اتلاقت، عايز أطلب توصيلها لمقر إقامتي.",
             "en": "My baggage has been found - I'd like to request delivery to my residence."}},
]

TXT = {
    "ar": {
        "title": "🧳 AeroResolve", "tagline": "منصة دعم ومتابعة الشنط المفقودة أو المتأخرة",
        "sidebar_title": "🧳 AeroResolve",
        "case_ref": "رقم المطالبة", "case_tag": "رقم تاج الشنطة", "no_case_yet": "لسه معندكش بلاغ مسجل.",
        "status_title": "حالة بلاغك", "new_case": "بلاغ جديد", "reset_data": "إعادة تعيين البيانات",
        "reset_help": "استرجاع بيانات العرض التجريبي (هيحذف أي مطالبات اتعملت)",
        "disclaimer": "نسخة تجريبية بتستخدم بيانات وهمية بالكامل. مش تابعة لأي شركة طيران حقيقية. "
                       "من فضلك متكتبش بيانات شخصية حقيقية.",
        "welcome": "أهلاً! أنا مساعد AeroResolve. احكيلي حصل إيه مع شنطتك، أو استخدم الأزرار اللي تحت.",
        "tabs": ["📦 الشنطة", "📝 المطالبة", "✉️ الإيميل"],
        "baggage_empty": "لسه معندناش بيانات عن شنطتك. هتظهر هنا أول ما نلاقيها.",
        "baggage_status": "حالة الشنطة", "tag": "تاج الشنطة", "last_loc": "آخر موقع معروف",
        "last_upd": "آخر تحديث", "booking": "الحجز", "passenger_flight": "الراكب / الرحلة",
        "claim_empty": "لسه مفيش قرار بخصوص المطالبة.", "claim_created": "تم تسجيل المطالبة",
        "claim_existing": "عندك مطالبة مسجلة بالفعل - مفيش مطالبة تانية اتعملت.",
        "claim_none": "مفيش مطالبة اتعملت.", "follow_up": "متابعة", "pdf": "⬇️ تحميل تقرير الحالة (PDF)",
        "email_empty": "هتظهر هنا رسالة التأكيد بعد ما نخلص معالجة بلاغك.",
        "email_to": "إلى", "email_subject": "الموضوع", "email_download": "⬇️ تحميل الإيميل",
        "lang_toggle": "اللغة",
    },
    "en": {
        "title": "🧳 AeroResolve", "tagline": "Baggage support & resolution for missing or delayed bags",
        "sidebar_title": "🧳 AeroResolve",
        "case_ref": "Claim Reference", "case_tag": "Baggage Tag", "no_case_yet": "No report on file yet.",
        "status_title": "Your report status", "new_case": "New report", "reset_data": "Reset data",
        "reset_help": "Restore the demo database (deletes any claims created)",
        "disclaimer": "Prototype using entirely synthetic data. Not affiliated with any real airline. "
                       "Please do not enter real personal information.",
        "welcome": "Hi! I'm the AeroResolve assistant. Tell me what happened to your baggage, or use a button below.",
        "tabs": ["📦 Baggage", "📝 Claim", "✉️ Email"],
        "baggage_empty": "We don't have your baggage details yet. They'll appear here once found.",
        "baggage_status": "Baggage status", "tag": "Baggage tag", "last_loc": "Last known location",
        "last_upd": "Last updated", "booking": "Booking", "passenger_flight": "Passenger / Flight",
        "claim_empty": "No claim decision yet.", "claim_created": "Claim registered",
        "claim_existing": "You already have an active claim - no duplicate was created.",
        "claim_none": "No claim was created.", "follow_up": "Follow-up", "pdf": "⬇️ Download case report (PDF)",
        "email_empty": "Your confirmation email will appear here once your report is processed.",
        "email_to": "To", "email_subject": "Subject", "email_download": "⬇️ Download email",
        "lang_toggle": "Language",
    },
}

CSS = """
<style>
.stApp, .block-container {background: #F8FAFC;}
.block-container {padding-top: 1.2rem; max-width: 1280px;}
.hero {background: linear-gradient(120deg,#0B2A4A 0%,#114272 55%,#0EA5E9 130%); color: white;
       padding: 22px 28px; border-radius: 18px; margin-bottom: 16px; box-shadow: 0 4px 16px rgba(11,42,74,.15);}
.hero h1 {margin: 0; font-size: 1.9rem; letter-spacing: .3px; color: white;}
.hero p {margin: 6px 0 0 0; opacity: .92;}
.card {border: 1px solid #eef2f7; border-radius: 16px; padding: 16px 18px; background: #fff;
       margin-bottom: 10px; box-shadow: 0 1px 3px rgba(15,23,42,.06), 0 1px 2px rgba(15,23,42,.04);}
.card .k {font-size: .74rem; color: #64748b; text-transform: uppercase; letter-spacing: .5px;}
.card .v {font-size: 1.05rem; font-weight: 700; color: #0B2A4A;}
.chip-status {display:inline-block; padding: 4px 14px; border-radius: 999px; color: white; font-weight: 700; font-size: .88rem;}
.mail {border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px 18px; background: #F8FAFC;
       white-space: pre-wrap; font-size: .92rem;}
.disclaimer {font-size: .74rem; color: #94a3b8;}
.sidebar-case {border: 1px solid #eef2f7; border-radius: 14px; padding: 12px 14px; background: #fff;
       margin-bottom: 12px; box-shadow: 0 1px 3px rgba(15,23,42,.05);}
.sidebar-case .row {display:flex; justify-content:space-between; font-size:.84rem; padding: 3px 0; color:#334155;}
.sidebar-case .row b {color:#0B2A4A;}
.stepper {display:flex; flex-direction:column; margin: 6px 0 4px 0;}
.step {display:flex; align-items:flex-start; gap:10px; position:relative; padding-bottom:22px;}
.step:last-child {padding-bottom:0;}
.step .dot {width:14px; height:14px; border-radius:50%; background:#e2e8f0; flex-shrink:0; margin-top:3px;
            box-shadow:0 0 0 3px #fff;}
.step.done .dot {background:#0EA5E9;}
.step.active .dot {background:#0B2A4A; box-shadow:0 0 0 5px rgba(11,42,74,.14);}
.step:not(:last-child)::before {content:""; position:absolute; inset-inline-start:6px; top:16px; bottom:0;
            width:2px; background:#e2e8f0;}
.step.done:not(:last-child)::before {background:#0EA5E9;}
.step .label {font-size:.84rem; color:#64748b; font-weight:500; line-height:1.35;}
.step.active .label {color:#0B2A4A; font-weight:700;}
.step.done .label {color:#0f172a;}
.stButton>button {border-radius: 999px !important; border: 1px solid #bae6fd !important; background: #fff !important;
       color: #0B2A4A !important; font-weight: 600 !important; box-shadow: none !important;}
.stButton>button:hover {background: #eff6ff !important; border-color: #0EA5E9 !important; color: #0B2A4A !important;}
[data-testid="stChatInput"] textarea {border-radius: 14px !important;}
</style>
"""


# ----------------------------------------------------------------------- setup
@st.cache_resource(show_spinner=False)
def get_orchestrator() -> Orchestrator:
    return Orchestrator(build_services())


def status_chip(status: str, lang: str) -> str:
    color = STATUS_COLORS.get(status, "#64748b")
    label = STATUS_LABEL_AR.get(status, status) if lang == "ar" else status
    return f'<span class="chip-status" style="background:{color}">{html.escape(label)}</span>'


def kv(label: str, value) -> str:
    return f'<div class="card"><div class="k">{html.escape(label)}</div><div class="v">{html.escape(str(value))}</div></div>'


# -------------------------------------------------------------- sidebar bits
def render_sidebar_case(case: CaseState, t: dict, lang: str) -> None:
    ref = (case.claim or {}).get("claim_id") or "—"
    tag = (case.baggage or {}).get("baggage_tag") or case.intake.get("baggage_tag") or "—"
    if ref == "—" and tag == "—":
        st.caption(t["no_case_yet"])
        return
    st.markdown(
        f'<div class="sidebar-case">'
        f'<div class="row"><span>{html.escape(t["case_ref"])}</span><b>{html.escape(str(ref))}</b></div>'
        f'<div class="row"><span>{html.escape(t["case_tag"])}</span><b>{html.escape(str(tag))}</b></div>'
        f'</div>', unsafe_allow_html=True)


def render_stepper(case: CaseState, lang: str) -> None:
    stage = stepper_stage(case)
    labels = UI_TEXT[lang]["stepper"]
    parts = []
    for i, label in enumerate(labels):
        state = "done" if i < stage else ("active" if i == stage else "")
        parts.append(f'<div class="step {state}"><div class="dot"></div><div class="label">{html.escape(label)}</div></div>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------- rendering
def render_baggage(case: CaseState, t: dict, lang: str) -> None:
    if not case.baggage and not case.booking:
        st.info(t["baggage_empty"])
        return
    b, bk, fl = case.baggage, case.booking or {}, case.flight or {}
    if b:
        st.markdown(f"**{t['baggage_status']}** &nbsp; {status_chip(b['status'], lang)}", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.markdown(kv(t["tag"], b["baggage_tag"]) + kv(t["last_loc"], b["last_known_location"]), unsafe_allow_html=True)
        c2.markdown(kv(t["last_upd"], b["last_updated"]) + kv(t["booking"], b["booking_reference"]), unsafe_allow_html=True)
    if bk:
        st.markdown(kv(t["passenger_flight"], f"{bk['passenger_name']} · {bk['flight_number']} "
                       f"({fl.get('origin', '?')} → {fl.get('destination', '?')}) · {bk['flight_date']}"),
                    unsafe_allow_html=True)


def render_claim(case: CaseState, t: dict, lang: str) -> None:
    if not case.decision:
        st.info(t["claim_empty"])
        return
    if case.claim:
        c = case.claim
        if case.claim_created:
            st.success(f"{t['claim_created']}: **{c['claim_id']}**")
        else:
            st.warning(f"{t['claim_existing']} (**{c['claim_id']}**)")
        cols = {"ar": {"Claim ID": "رقم المطالبة", "Passenger": "الراكب", "Flight": "الرحلة",
                       "Baggage tag": "تاج الشنطة", "Claim type": "نوع المطالبة", "Claim status": "حالة المطالبة",
                       "Last known location": "آخر موقع معروف"},
               "en": {}}[lang] if lang == "ar" else {}
        rows = {"Claim ID": c["claim_id"], "Passenger": c["passenger_name"], "Flight": c["flight_number"],
                "Baggage tag": c["baggage_tag"], "Claim type": c["claim_type"], "Claim status": c["status"],
                "Last known location": c["last_known_location"]}
        if lang == "ar":
            rows = {cols.get(k, k): v for k, v in rows.items()}
        st.dataframe(pd.DataFrame([rows]).T.rename(columns={0: "—"}), width="stretch")
    else:
        st.info(f"{t['claim_none']} {case.decision['reason']}")
    if case.follow_up:
        f = case.follow_up
        st.caption(f"{t['follow_up']}: {f['follow_up_type']} · {f['due_at']} UTC")
    if case.report_path and Path(case.report_path).exists():
        st.download_button(t["pdf"], Path(case.report_path).read_bytes(),
                           file_name=Path(case.report_path).name, mime="application/pdf")


def render_email(case: CaseState, t: dict) -> None:
    if not case.email:
        st.info(t["email_empty"])
        return
    e = case.email
    st.markdown(f"**{t['email_to']}:** {html.escape(e['recipient'])}  \n**{t['email_subject']}:** {html.escape(e['subject'])}")
    st.markdown(f'<div class="mail">{html.escape(e["body"])}</div>', unsafe_allow_html=True)
    st.download_button(t["email_download"], f"To: {e['recipient']}\nSubject: {e['subject']}\n\n{e['body']}",
                       file_name="passenger_email.txt")


# ------------------------------------------------------------------------ app
def main() -> None:
    st.set_page_config(page_title="AeroResolve", page_icon="🧳", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    orch = get_orchestrator()
    if "case" not in st.session_state:
        st.session_state.case = orch.new_case()
    case: CaseState = st.session_state.case
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "ar"
    # Once the passenger has actually written to the assistant, let the chrome
    # follow the language the assistant detected for them automatically - the
    # sidebar toggle below can still override it at any time.
    if case.messages:
        st.session_state.ui_lang = case.language
    lang = st.session_state.ui_lang
    t = TXT[lang]

    prompt: str | None = None
    with st.sidebar:
        st.markdown(f"### {t['sidebar_title']}")
        st.caption(t["tagline"])
        tog1, tog2 = st.columns(2)
        if tog1.button("العربية", width="stretch", type=("primary" if lang == "ar" else "secondary")):
            st.session_state.ui_lang = "ar"
            st.rerun()
        if tog2.button("English", width="stretch", type=("primary" if lang == "en" else "secondary")):
            st.session_state.ui_lang = "en"
            st.rerun()
        st.divider()
        st.markdown(f"**{t['status_title']}**")
        render_sidebar_case(case, t, lang)
        render_stepper(case, lang)
        st.divider()
        c1, c2 = st.columns(2)
        if c1.button(t["new_case"], width="stretch"):
            st.session_state.case = orch.new_case()
            st.rerun()
        if c2.button(t["reset_data"], width="stretch", help=t["reset_help"]):
            reset_db(orch.s.db_path, orch.s.data_dir)
            st.session_state.case = orch.new_case()
            st.rerun()
        st.markdown(f'<p class="disclaimer">{html.escape(t["disclaimer"])}</p>', unsafe_allow_html=True)

    st.markdown(f'<div class="hero"><h1>{t["title"]}</h1><p>{t["tagline"]}</p></div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])
    with left:
        chat_box = st.container(height=480)   # filled below, after any new message is processed
        chip_cols = st.columns(len(CHIPS))
        for col, chip in zip(chip_cols, CHIPS):
            if col.button(chip["label"][lang], key=f"chip_{chip['key']}", width="stretch"):
                prompt = chip["msg"][lang]
    typed = st.chat_input(UI_TEXT[lang]["chat_placeholder"])
    prompt = prompt or typed

    if prompt:
        with st.spinner("…" if lang == "ar" else "Working…"):
            orch.run_turn(case, prompt)

    with chat_box:
        if not case.messages:
            st.chat_message("assistant", avatar="🧳").markdown(t["welcome"])
        for m in case.messages:
            st.chat_message(m["role"], avatar="🧳" if m["role"] == "assistant" else None).markdown(m["content"])

    with right:
        tabs = st.tabs(t["tabs"])
        with tabs[0]:
            render_baggage(case, t, lang)
        with tabs[1]:
            render_claim(case, t, lang)
        with tabs[2]:
            render_email(case, t)
