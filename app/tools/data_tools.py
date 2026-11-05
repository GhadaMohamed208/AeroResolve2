"""Data tools: policy retrieval (RAG), CSV export and PDF case-report generation."""
from __future__ import annotations

import csv
import re
from pathlib import Path

from app.database.db import get_conn
from app.rag.policy_rag import load_policy_index
from app.tools.base import ToolResult, safe_tool


@safe_tool
def search_policy(policy_path, query, k=2) -> ToolResult:
    """Retrieve the most relevant sections of the baggage policy PDF."""
    if not Path(policy_path).exists():
        return ToolResult(False, error_code="POLICY_UNAVAILABLE", message="The baggage policy document is missing.")
    hits = load_policy_index(policy_path).search(query, k=k)
    if not hits:
        return ToolResult(False, error_code="POLICY_NO_MATCH", message="No policy section matched the question.")
    return ToolResult(True, data={"query": query, "hits": hits})


def first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n])


@safe_tool
def export_claims_csv(db_path, exports_dir) -> ToolResult:
    """Dashboard/report action: export all claims to CSV (stand-in for a Google Sheet sync)."""
    out = Path(exports_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "claims_export.csv"
    with get_conn(db_path) as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM claims ORDER BY seq").fetchall()]
    if not rows:
        return ToolResult(False, error_code="NO_CLAIMS", message="There are no claims to export.")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return ToolResult(True, data={"path": str(path), "rows": len(rows)})


@safe_tool
def generate_case_pdf(case_summary, email, exports_dir) -> ToolResult:
    """Generate a PDF case report (summary + passenger email)."""
    from xml.sax.saxutils import escape
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib import colors

    out = Path(exports_dir)
    out.mkdir(parents=True, exist_ok=True)
    ident = case_summary.get("claim_id") or case_summary.get("baggage_tag") or "case"
    path = out / f"case_report_{ident}.pdf"
    styles = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=styles["Normal"], fontSize=10, leading=14)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm, title="AeroResolve Case Report")
    story = [Paragraph("AeroResolve - Baggage Case Report", styles["Title"]),
             Paragraph("Prototype with synthetic data - not affiliated with any airline", styles["Italic"]),
             Spacer(1, 12)]
    rows = [[k, str(v)] for k, v in case_summary.get("fields", {}).items()]
    table = Table(rows, colWidths=[5 * cm, 11 * cm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF3FA"))]))
    story += [table, Spacer(1, 12)]
    if case_summary.get("narrative"):
        story += [Paragraph("Case narrative", styles["Heading3"]),
                  Paragraph(escape(case_summary["narrative"]), body), Spacer(1, 10)]
    if email:
        story += [Paragraph("Passenger email (preview)", styles["Heading3"]),
                  Paragraph(f"<b>To:</b> {escape(email['recipient'])}", body),
                  Paragraph(f"<b>Subject:</b> {escape(email['subject'])}", body), Spacer(1, 6)]
        for para in email["body"].split("\n\n"):
            story.append(Paragraph(escape(para).replace("\n", "<br/>"), body))
            story.append(Spacer(1, 4))
    doc.build(story)
    return ToolResult(True, data={"path": str(path)})
