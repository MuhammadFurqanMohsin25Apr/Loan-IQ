from __future__ import annotations

import json
import os
from io import BytesIO
from datetime import datetime
from urllib import error, request
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..schemas import LoanApplication


def _build_prompt(case: LoanApplication) -> str:
    payload = case.payload
    scoring = case.scoring
    return f"""You are summarizing a loan case for a human officer.

Return concise markdown only with these sections:
## Summary
## Key risk signals
## Model linkage
## Manual action

Rules:
- Do not approve or reject the loan automatically.
- Clearly state that the final decision is manual by the officer or worker.
- Mention the case is driven by three model services: ANN credit risk, LSTM cash flow, and CNN document authenticity.
- Keep it short and practical.

Case data:
- Case ID: {case.id}
- Applicant: {payload.applicant_name}
- Purpose: {payload.purpose}
- Requested amount: {payload.requested_amount:.2f}
- Tenure: {payload.tenure_months} months
- Final score: {scoring.final_score:.3f}
- Risk band: {scoring.risk_band}
- Recommendation: {scoring.recommendation}
- Confidence: {scoring.confidence:.3f}
- Tabular score: {scoring.tabular_score:.3f}
- Cash-flow score: {scoring.cash_flow_score:.3f}
- Authenticity score: {scoring.authenticity_score:.3f}
- Officer decision: {case.officer_decision or 'Pending'}
- Officer comment: {case.officer_comment or 'No comment recorded'}
- Risk drivers: {json.dumps(scoring.shap_explanation)}
- Time steps: {json.dumps(scoring.time_step_signal)}
- Document rationale: {scoring.document_rationale}
"""


def _gemini_summary(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 500,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=30) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise RuntimeError(f"Gemini request failed with HTTP {exc.code}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Gemini request failed: {exc.reason}") from exc

    candidates = response_data.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if text.strip():
            return text.strip()

    raise RuntimeError("Gemini returned no summary text")


def _fallback_summary(case: LoanApplication) -> str:
    payload = case.payload
    scoring = case.scoring
    return f"""# Loan IQ Case Summary

## Summary
- Applicant: {payload.applicant_name}
- Requested amount: {payload.requested_amount:.2f}
- Final score: {scoring.final_score:.3f}
- Risk band: {scoring.risk_band}

## Key risk signals
- Tabular score: {scoring.tabular_score:.3f}
- Cash-flow score: {scoring.cash_flow_score:.3f}
- Authenticity score: {scoring.authenticity_score:.3f}

## Model linkage
- ANN credit risk, LSTM cash flow, and CNN document authenticity are used for scoring.

## Manual action
- Officer or worker must manually approve or reject this loan.
- Current officer decision: {case.officer_decision or 'Pending'}
"""


def generate_case_report(case: LoanApplication) -> str:
    generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    prompt = _build_prompt(case)
    try:
        summary = _gemini_summary(prompt)
    except Exception:
        summary = _fallback_summary(case)

    return f"{summary}\n\n## Metadata\n- Generated at: {generated_at}\n- Model version: {case.scoring.model_outputs[0].model_version if case.scoring.model_outputs else 'unknown'}\n- Source: Gemini API when available; otherwise local fallback\n- Decision mode: manual officer approval required\n"


def _report_sections(case: LoanApplication) -> list[tuple[str, list[str]]]:
    payload = case.payload
    scoring = case.scoring
    return [
        (
            "Applicant Summary",
            [
                f"Case ID: {case.id}",
                f"Applicant: {payload.applicant_name}",
                f"Email: {payload.applicant_email or 'Not provided'}", 
                f"Employment type: {payload.employment_type.replace('_', ' ').title()}",
                f"Purpose: {payload.purpose}",
                f"Requested amount: {payload.requested_amount:.2f}",
                f"Tenure: {payload.tenure_months} months",
            ],
        ),
        (
            "Model Assessment",
            [
                f"Final score: {scoring.final_score:.3f}",
                f"Risk band: {scoring.risk_band}",
                f"Recommendation: {scoring.recommendation}",
                f"Confidence: {scoring.confidence:.3f}",
                f"Tabular score: {scoring.tabular_score:.3f}",
                f"Cash-flow score: {scoring.cash_flow_score:.3f}",
                f"Authenticity score: {scoring.authenticity_score:.3f}",
            ],
        ),
        (
            "Manual Action",
            [
                f"Current status: {case.status}",
                f"Officer decision: {case.officer_decision or 'Pending'}",
                f"Officer comment: {case.officer_comment or 'No comment recorded'}",
                "Final approval or rejection must be performed by the officer or worker.",
            ],
        ),
    ]


def _escape_lines(lines: list[str]) -> str:
    return "<br/>".join(f"&bull; {escape(line)}" for line in lines)


def build_case_report_pdf(case: LoanApplication) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title=f"Loan IQ Case {case.id}",
        author="Loan IQ",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleAccent",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#102a43"),
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleAccent",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#627d98"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "SectionAccent",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BodyAccent",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1f2937"),
    )
    note_style = ParagraphStyle(
        "NoteAccent",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334e68"),
    )
    right_style = ParagraphStyle(
        "RightAccent",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#486581"),
    )
    left_style = ParagraphStyle(
        "LeftAccent",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#243b53"),
    )

    prompt = _build_prompt(case)
    try:
        summary = _gemini_summary(prompt)
        source_label = "Gemini summary"
    except Exception:
        summary = _fallback_summary(case)
        source_label = "Local fallback summary"

    sections = _report_sections(case)
    story = [
        Paragraph("Loan IQ Case Report", title_style),
        Paragraph("Formal officer review packet generated for manual decisioning.", subtitle_style),
    ]

    header_table = Table(
        [[
            Paragraph(f"Case ID: <b>{case.id}</b>", left_style),
            Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", right_style),
        ]],
        colWidths=[90 * mm, 70 * mm],
    )
    header_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.extend([header_table, Spacer(1, 8)])

    safe_summary = escape(summary).replace("\n", "<br/>")
    summary_box = Table(
        [[Paragraph(safe_summary, body_style)]],
        colWidths=[160 * mm],
    )
    summary_box.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#f59e0b")),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.extend([summary_box, Spacer(1, 10)])

    for section_title, lines in sections:
        story.append(Paragraph(section_title, section_style))
        bullets = _escape_lines(lines)
        section_box = Table(
            [[Paragraph(bullets, body_style)]],
            colWidths=[160 * mm],
        )
        section_box.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d9e2ec")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.extend([section_box, Spacer(1, 8)])

    if case.scoring.shap_explanation:
        story.append(Paragraph("Top Explainability Drivers", section_style))
        explain_rows = [["Feature", "Impact"]]
        for item in case.scoring.shap_explanation[:5]:
            impact = float(item["impact"])
            explain_rows.append([str(item["feature"]), f"{impact:+.3f}"])
        explain_table = Table(explain_rows, colWidths=[110 * mm, 50 * mm])
        explain_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102a43")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d9e2ec")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9e2ec")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.extend([explain_table, Spacer(1, 10)])

    footer_note = Paragraph(
        f"Source: {source_label}. Final decision remains manual for the officer or worker.",
        note_style,
    )
    story.append(footer_note)

    def _decorate(canvas, _doc) -> None:
        canvas.saveState()
        width, height = A4
        canvas.setFillColor(colors.HexColor("#102a43"))
        canvas.rect(0, height - 18 * mm, width, 18 * mm, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 12)
        canvas.drawString(18 * mm, height - 12 * mm, "Loan IQ")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(width - 18 * mm, height - 11.5 * mm, f"Case {case.id}")
        canvas.setFillColor(colors.HexColor("#486581"))
        canvas.setFont("Helvetica", 7)
        canvas.drawString(18 * mm, 10 * mm, "Manual officer decision required")
        canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    document.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    return buffer.getvalue()
