"""
dds_pdf_report.py
Generates two artefacts from a completed DDS pipeline run:
  1. A shed action report  (human-readable, one chain per section)
  2. An ATIL-format failure register  (2026-27 column layout, one row per chain)

Usage:
    from dds_pdf_report import build_report
    pdf_bytes = build_report(
        vehicle, log_start, log_end,
        sorted_results, localizer_annotations, atil_engine,
        sessions_list,
        report_type="action"   # or "atil_register"
    )
    # pdf_bytes is ready for st.download_button
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Palette (matches app dark theme as closely as print allows) ───────────────
C_BLACK   = colors.HexColor("#0d0d0d")
C_RED     = colors.HexColor("#c0392b")
C_AMBER   = colors.HexColor("#b7950b")
C_GREEN   = colors.HexColor("#1a7a4a")
C_GRAY    = colors.HexColor("#555555")
C_LGRAY   = colors.HexColor("#cccccc")
C_ROWEVEN = colors.HexColor("#f5f5f5")

W, H = A4


# ── Style helpers ─────────────────────────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()
    custom = {
        "Title": ParagraphStyle("DocTitle", parent=base["Title"],
                                fontSize=16, spaceAfter=4, textColor=C_BLACK),
        "Sub":   ParagraphStyle("Sub", parent=base["Normal"],
                                fontSize=9, textColor=C_GRAY, spaceAfter=2),
        "H2":    ParagraphStyle("H2", parent=base["Heading2"],
                                fontSize=11, spaceBefore=8, spaceAfter=3,
                                textColor=C_BLACK),
        "H3":    ParagraphStyle("H3", parent=base["Heading3"],
                                fontSize=9, spaceBefore=4, spaceAfter=2,
                                textColor=C_GRAY),
        "Body":  ParagraphStyle("Body", parent=base["Normal"],
                                fontSize=8.5, leading=13, spaceAfter=4),
        "Mono":  ParagraphStyle("Mono", parent=base["Code"],
                                fontSize=7.5, leading=11, spaceAfter=3,
                                fontName="Courier"),
        "Small": ParagraphStyle("Small", parent=base["Normal"],
                                fontSize=7, textColor=C_GRAY),
        "CardID":ParagraphStyle("CardID", parent=base["Normal"],
                                fontSize=22, fontName="Helvetica-Bold",
                                textColor=C_BLACK, spaceAfter=2),
    }
    return custom


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_GRAY)
    canvas.drawString(15*mm, 8*mm,
        f"ELS/BL Shed · DDS Fault Analysis Engine · Generated {datetime.now().strftime('%d-%b-%Y %H:%M')}")
    canvas.drawRightString(W - 15*mm, 8*mm, f"Page {doc.page}")
    canvas.restoreState()


def _sev_color(severity):
    s = (severity or "").upper()
    if s == "HIGH":   return C_RED
    if s == "MEDIUM": return C_AMBER
    return C_GREEN


# ── Report 1: Shed Action Report ──────────────────────────────────────────────
def _build_action_report(buf, vehicle, log_start, log_end,
                         sorted_results, localizer_annotations, atil_engine,
                         sessions_list):
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=18*mm, bottomMargin=18*mm,
    )
    S = _styles()
    story = []

    # ── Cover block ───────────────────────────────────────────────────────────
    story.append(Paragraph("Locomotive DDS Fault Analysis Report", S["Title"]))
    story.append(Paragraph(
        f"Vehicle: <b>{vehicle}</b> &nbsp;·&nbsp; "
        f"Log period: {log_start} → {log_end} &nbsp;·&nbsp; "
        f"Sessions: {len(sessions_list)}",
        S["Sub"]
    ))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_LGRAY, spaceAfter=6))

    high   = [r for r in sorted_results if getattr(r, "severity", "").upper() == "HIGH"]
    medium = [r for r in sorted_results if getattr(r, "severity", "").upper() == "MEDIUM"]
    persis = [r for r in sorted_results if getattr(r, "outcome", "") == "PERSISTING"]

    summary_data = [
        ["HIGH chains", "MEDIUM chains", "Persisting", "Total chains"],
        [str(len(high)), str(len(medium)), str(len(persis)), str(len(sorted_results))],
    ]
    summary_tbl = Table(summary_data, colWidths=[42*mm]*4)
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), C_BLACK),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,1), 9),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_ROWEVEN, colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.3, C_LGRAY),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ]))
    story.append(summary_tbl)
    story.append(Spacer(1, 8*mm))

    # ── One block per chain ───────────────────────────────────────────────────
    for i, r in enumerate(sorted_results, 1):
        severity  = getattr(r, "severity",  "HIGH").upper()
        chain_id  = getattr(r, "chain_id",  "UNKNOWN")
        name      = getattr(r, "name",      chain_id)
        outcome   = getattr(r, "outcome",   "UNKNOWN")

        chain_uid = getattr(r, "_uid", None)
        ann       = localizer_annotations.get(chain_uid) if chain_uid else None
        tb        = ann.get("trigger_board") if ann else None
        guidance  = ann.get("graduated_guidance", "") if ann else ""
        b_id      = getattr(tb, "board_id",    "—") if tb else "—"
        b_desc    = getattr(tb, "description", "") if tb else ""
        b_rep     = getattr(tb, "replaces",    "") if tb else ""

        atil = atil_engine.process_chain(
            chain_id=chain_id,
            trigger_text=getattr(r, "trigger_text", ""),
            pcb_verdict=getattr(
                ann.get("pcb_suspect") if ann else None, "verdict", None),
        )

        sev_col = _sev_color(severity)
        persist_note = "  [PERSISTING ACROSS SESSIONS]" if outcome == "PERSISTING" else ""

        block = []

        # Chain header row
        header_data = [[
            Paragraph(f"<b>{i}. {name}{persist_note}</b>", S["Body"]),
            Paragraph(f"<font color='#{sev_col.hexval()[2:]}'>&#9679;</font> {severity}",
                      S["Body"]),
            Paragraph(f"`{chain_id}`", S["Mono"]),
        ]]
        hdr_tbl = Table(header_data, colWidths=[110*mm, 25*mm, 45*mm])
        hdr_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,-1), colors.HexColor("#eeeeee")),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("LINEBELOW",    (0,0), (-1,-1), 1.2, sev_col),
        ]))
        block.append(hdr_tbl)

        # Two-column detail
        left_items = []
        if getattr(r, "trigger_time", None):
            left_items.append(
                f"<b>Trigger:</b> {r.trigger_time.strftime('%d-%b-%Y  %H:%M:%S')}")
        if getattr(r, "trigger_text", None):
            left_items.append(f"<b>Root cause event:</b> <font name='Courier'>{r.trigger_text}</font>")
        if outcome:
            left_items.append(f"<b>Status:</b> {outcome}")

        right_items = []
        right_items.append(f"<b>Target:</b> {b_id}")
        if b_desc:
            right_items.append(b_desc)
        if b_rep and b_rep != "Physical inspection required":
            right_items.append(f"<i>Rack/location: {b_rep}</i>")
        if atil.candidates:
            top = atil.candidates[0]
            right_items.append(
                f"<b>Fleet evidence:</b> {top.pct:.0f}% — {top.name}")
            for c in atil.candidates[1:3]:
                right_items.append(f"  {c.pct:.0f}% — {c.name}")
        if guidance:
            right_items.append(f"<b>Action:</b> {guidance}")

        detail_data = [[
            Paragraph("<br/>".join(left_items),  S["Body"]),
            Paragraph("<br/>".join(right_items), S["Body"]),
        ]]
        detail_tbl = Table(detail_data, colWidths=[90*mm, 90*mm])
        detail_tbl.setStyle(TableStyle([
            ("VALIGN",       (0,0), (-1,-1), "TOP"),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("LEFTPADDING",  (0,0), (0,-1),  6),
            ("LEFTPADDING",  (1,0), (1,-1),  8),
            ("LINEAFTER",    (0,0), (0,-1),  0.3, C_LGRAY),
        ]))
        block.append(detail_tbl)

        action_text = getattr(r, "action", None) or getattr(r, "description", "")
        if action_text:
            block.append(Paragraph("<b>Diagnostic &amp; Shed Procedure</b>", S["H3"]))
            block.append(Paragraph(action_text, S["Body"]))

        block.append(Spacer(1, 3*mm))
        story.append(KeepTogether(block))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


# ── Report 2: ATIL-format failure register ────────────────────────────────────
# Columns mirror the 2026-27 sheet:
# S.N. | Loco Shed | Loco Type | Loco No. | Date | Type of Failure |
# Main Equipment | Failed Item | Investigation Details

def _build_atil_register(buf, vehicle, log_start, log_end,
                          sorted_results, localizer_annotations, atil_engine,
                          sessions_list):
    doc = SimpleDocTemplate(
        buf, pagesize=landscape(A4),
        leftMargin=12*mm, rightMargin=12*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )
    S = _styles()
    story = []

    # Title block
    story.append(Paragraph(
        "FORMAT FOR LISTING OF FAILURE EVENT — IGBT based POWER CONVERTER (SR), "
        "AUX. CONVERTER (BUR) &amp; VEHICLE CONTROL ELECTRONICS (VCU)",
        S["Sub"]
    ))
    story.append(Paragraph(
        f"ELS/BL Shed &nbsp;·&nbsp; Vehicle: <b>{vehicle}</b> &nbsp;·&nbsp; "
        f"DDS Log: {log_start} → {log_end} &nbsp;·&nbsp; "
        f"Generated: {datetime.now().strftime('%d-%b-%Y')}",
        S["Sub"]
    ))
    story.append(Spacer(1, 4*mm))

    # Column headers — two-row like the real register
    COL_W = [10*mm, 14*mm, 16*mm, 18*mm, 22*mm, 30*mm, 38*mm, 38*mm, 80*mm]
    header_row1 = [
        "S.N.", "Loco Shed*", "Loco Type*", "Loco No.*",
        "Date of Failure*", "Type of Fault",
        "Main Equipment*", "Failed Item / Sub-assembly*",
        "Investigation Details",
    ]

    rows = [header_row1]

    for i, r in enumerate(sorted_results, 1):
        chain_id  = getattr(r, "chain_id",  "")
        outcome   = getattr(r, "outcome",   "")
        severity  = getattr(r, "severity",  "")

        chain_uid = getattr(r, "_uid", None)
        ann       = localizer_annotations.get(chain_uid) if chain_uid else None
        tb        = ann.get("trigger_board") if ann else None
        b_id      = getattr(tb, "board_id",    "—") if tb else "—"
        b_desc    = getattr(tb, "description", "") if tb else ""

        atil = atil_engine.process_chain(
            chain_id=chain_id,
            trigger_text=getattr(r, "trigger_text", ""),
            pcb_verdict=getattr(
                ann.get("pcb_suspect") if ann else None, "verdict", None),
        )

        # Derive main equipment from chain_id prefix heuristic
        cid_upper = chain_id.upper()
        if "BUR" in cid_upper:
            main_eq = "Aux. Converter (BUR)"
        elif "SR" in cid_upper or "TRACTION" in cid_upper or "BOGIE" in cid_upper:
            main_eq = "Traction Converter (SR)"
        elif "VCU" in cid_upper or "CCUO" in cid_upper:
            main_eq = "TCN-VCU"
        else:
            main_eq = "—"

        trig_time = getattr(r, "trigger_time", None)
        date_str  = trig_time.strftime("%d-%m-%Y") if trig_time else log_end

        fault_type = "DDS-Detected"
        if outcome == "PERSISTING":
            fault_type = "Line Failure / Persisting"
        elif severity == "HIGH":
            fault_type = "Shed Arising"

        # Investigation text — root cause + top ATIL candidate
        inv_parts = []
        trig_text = getattr(r, "trigger_text", "")
        if trig_text:
            inv_parts.append(f"DDS trigger: {trig_text}.")
        action_text = getattr(r, "action", None) or getattr(r, "description", "")
        if action_text:
            # Trim to ~200 chars for register column
            short = action_text[:200].rsplit(" ", 1)[0] + "…" \
                    if len(action_text) > 200 else action_text
            inv_parts.append(short)
        if atil.candidates:
            top = atil.candidates[0]
            inv_parts.append(f"Fleet evidence: {top.pct:.0f}% — {top.name}.")

        rows.append([
            str(i),
            "BL",
            "WAG-9/WAP-5",   # shed default; could be derived from vehicle name
            vehicle,
            date_str,
            fault_type,
            main_eq,
            Paragraph(f"<b>{b_id}</b><br/>{b_desc}", S["Small"]),
            Paragraph(" ".join(inv_parts), S["Small"]),
        ])

    tbl = Table(rows, colWidths=COL_W, repeatRows=1)
    row_styles = [
        ("BACKGROUND",   (0, 0), (-1, 0), C_BLACK),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 7),
        ("FONTSIZE",     (0, 1), (-1, -1), 7),
        ("ALIGN",        (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("GRID",         (0, 0), (-1, -1), 0.3, C_LGRAY),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, C_ROWEVEN]),
    ]
    # Colour-code severity in S.N. column
    for idx, r in enumerate(sorted_results, 1):
        sev = getattr(r, "severity", "").upper()
        col = _sev_color(sev)
        row_styles.append(("TEXTCOLOR", (0, idx), (0, idx), col))
        row_styles.append(("FONTNAME",  (0, idx), (0, idx), "Helvetica-Bold"))

    tbl.setStyle(TableStyle(row_styles))
    story.append(tbl)

    def _lf_header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(C_GRAY)
        Lw, Lh = landscape(A4)
        canvas.drawString(12*mm, 8*mm,
            f"ELS/BL Shed · ATIL Failure Register Format · "
            f"Generated {datetime.now().strftime('%d-%b-%Y %H:%M')}")
        canvas.drawRightString(Lw - 12*mm, 8*mm, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_lf_header_footer, onLaterPages=_lf_header_footer)


# ── Public entry point ────────────────────────────────────────────────────────
def build_report(vehicle, log_start, log_end,
                 sorted_results, localizer_annotations, atil_engine,
                 sessions_list, report_type="action"):
    """
    report_type: "action" → shed action report (portrait)
                 "atil_register" → ATIL 2026-27 format (landscape)
    Returns bytes.
    """
    buf = io.BytesIO()
    if report_type == "atil_register":
        _build_atil_register(buf, vehicle, log_start, log_end,
                             sorted_results, localizer_annotations,
                             atil_engine, sessions_list)
    else:
        _build_action_report(buf, vehicle, log_start, log_end,
                             sorted_results, localizer_annotations,
                             atil_engine, sessions_list)
    return buf.getvalue()
