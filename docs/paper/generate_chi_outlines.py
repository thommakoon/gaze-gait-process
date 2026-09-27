"""Generate CHI System and Method writing-outline PDFs for gazeGait."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT_DIR = Path(__file__).resolve().parent


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontSize=18,
            leading=22,
            spaceAfter=6,
            textColor=colors.HexColor("#1a1a1a"),
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCustom",
            parent=base["Normal"],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#555555"),
            spaceAfter=16,
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "H1Custom",
            parent=base["Heading1"],
            fontSize=13,
            leading=16,
            spaceBefore=14,
            spaceAfter=6,
            textColor=colors.HexColor("#111111"),
        ),
        "h2": ParagraphStyle(
            "H2Custom",
            parent=base["Heading2"],
            fontSize=11,
            leading=14,
            spaceBefore=10,
            spaceAfter=4,
            textColor=colors.HexColor("#222222"),
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=base["Normal"],
            fontSize=9.5,
            leading=12.5,
            leftIndent=4,
            spaceAfter=2,
        ),
        "note": ParagraphStyle(
            "NoteCustom",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#333333"),
            backColor=colors.HexColor("#f4f4f4"),
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "CaptionCustom",
            parent=base["Normal"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#444444"),
            spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "FooterCustom",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#777777"),
            alignment=TA_CENTER,
        ),
        "mono": ParagraphStyle(
            "MonoCustom",
            parent=base["Code"],
            fontSize=8,
            leading=11,
            fontName="Courier",
            backColor=colors.HexColor("#f7f7f7"),
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "th": ParagraphStyle(
            "ThCell",
            parent=base["Normal"],
            fontSize=8.5,
            leading=11,
            fontName="Helvetica-Bold",
        ),
        "td": ParagraphStyle(
            "TdCell",
            parent=base["Normal"],
            fontSize=8.5,
            leading=11,
        ),
    }
    return s


def bullets(items, style):
    return ListFlowable(
        [ListItem(Paragraph(i, style), leftIndent=12, bulletColor=colors.HexColor("#333333")) for i in items],
        bulletType="bullet",
        start="•",
        leftIndent=15,
        spaceBefore=2,
        spaceAfter=6,
    )


def table(headers, rows, col_widths):
    s = styles()
    data = [[Paragraph(h, s["th"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(c, s["td"]) for c in row])
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def add_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(0.75 * inch, 0.5 * inch, "gazeGait — CHI writing outline")
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_system_pdf():
    s = styles()
    path = OUT_DIR / "CHI_System_Section_Outline.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="CHI System Section Outline — gazeGait",
        author="gazeGait writing guide",
    )
    story = []

    story.append(Paragraph("CHI Paper — System Section Outline", s["title"]))
    story.append(
        Paragraph(
            "gazeGait · Quest 3 Fitts + OpenEye Neon + dual foot IMUs · Writing guide for §3 System",
            s["subtitle"],
        )
    )

    story.append(Paragraph("Purpose of this section", s["h1"]))
    story.append(
        Paragraph(
            "System describes <b>what you built / assembled</b>: hardware, software, sync, and how streams "
            "are captured on a common timeline. Leave Fitts IDs, factorial design, participants, and statistics "
            "for Method. Related Work covers Fitts theory, modality literature, and locomotion HCI.",
            s["body"],
        )
    )

    story.append(Paragraph("Recommended structure", s["h1"]))
    story.append(
        Paragraph(
            "3 System<br/>"
            "3.1 Overview<br/>"
            "3.2 VR Fitts Application (Quest 3)<br/>"
            "3.3 Gaze Sensing with OpenEye (Neon on Quest 3)<br/>"
            "3.4 Foot-Worn IMUs for Gait<br/>"
            "3.5 Synchronization and Recording Hub<br/>"
            "3.6 Data Products (optional, short)",
            s["mono"],
        )
    )

    story.append(
        Paragraph(
            "<b>Suggested opening sentence:</b> We built a synchronized multimodal capture stack that combines "
            "a Quest 3 angular Fitts application, Neon eye tracking via OpenEye, and dual foot-worn IMUs, "
            "so selection behavior and gait can be analyzed on a common timeline.",
            s["note"],
        )
    )

    # 3.1
    story.append(Paragraph("3.1 Overview", s["h1"]))
    story.append(
        Paragraph(
            "One short paragraph stating that a PC hub orchestrates Quest Fitts, Neon gaze, and dual foot IMUs, "
            "and writes clock offsets so all streams share one timeline.",
            s["body"],
        )
    )
    story.append(Paragraph("<b>Fig 1 — System overview diagram (MUST)</b>", s["h2"]))
    story.append(
        Paragraph(
            "Boxes: Participant → Quest 3 Fitts app | Neon glasses | LF/RF foot IMUs → OpenEye PC GUI → "
            "synced recordings (Quest, Neon, IMU, sync.json). Arrows: TCP commander / time-echo, Neon API, "
            "USB serial IMU. Redraw the README mermaid cleanly for CHI (flat, labeled, no pipeline clutter).",
            s["caption"],
        )
    )

    # 3.2
    story.append(Paragraph("3.2 VR Fitts Application (Quest 3)", s["h1"]))
    story.append(bullets([
        "Unity Main Study on Meta Quest 3",
        "Virtual wall at <b>2 m</b>; ring and rectangle layouts",
        "Three cursors: head ray, hand ray, eye gaze; <b>pinch confirm</b> for all three",
        "In-app logging ~<b>100 Hz</b>",
    ], s["bullet"]))
    story.append(Paragraph("<b>Fig 2 — Task UI in headset (MUST)</b>", s["h2"]))
    story.append(
        Paragraph(
            "Two panels (or one composite): Ring and Rectangle on the wall, with one A/W example labeled. "
            "Prefer a clean HMD screenshot or illustration.",
            s["caption"],
        )
    )
    story.append(Paragraph("<b>Fig 3 — Three modalities (NICE)</b>", s["h2"]))
    story.append(
        Paragraph(
            "Same scene, three insets: head ray / hand ray / eye gaze + pinch. Helps readers see the IVs instantly.",
            s["caption"],
        )
    )

    # 3.3
    story.append(Paragraph("3.3 Gaze Sensing with OpenEye", s["h1"]))
    story.append(bullets([
        "Pupil Labs Neon as external eye tracker on Quest 3",
        "OpenEye maps Neon gaze into the Quest task space",
        "5×5 (25-dot) calibration on the <b>same 2 m plane</b> as Fitts (H ±20°, V +15°…−35°)",
        "Cite OpenEye (ETRA ’26) briefly; do not re-derive the full method here",
    ], s["bullet"]))
    story.append(Paragraph("<b>Fig 4 — Hardware worn view (MUST)</b>", s["h2"]))
    story.append(
        Paragraph(
            "Photo: Quest 3 + Neon mount on a person (side or 3/4). Label Quest, Neon, mount. "
            "This is the “how did you put ET on Quest 3?” figure.",
            s["caption"],
        )
    )
    story.append(Paragraph("<b>Fig 5 — Calibration grid (NICE)</b>", s["h2"]))
    story.append(
        Paragraph(
            "Screenshot/schematic of the 25-dot calib FOV. Useful if eye accuracy is part of the contribution story.",
            s["caption"],
        )
    )

    # 3.4
    story.append(Paragraph("3.4 Foot-Worn IMUs for Gait", s["h1"]))
    story.append(bullets([
        "Dual foot IMUs (LF/RF), QT Py + ICM-20948",
        "Recorded on <b>PC USB serial</b> during walking",
        "Purpose: initial contacts → gait phase for coupling analyses",
        "Standing practice: <b>no foot IMU</b> (state once)",
    ], s["bullet"]))
    story.append(Paragraph("<b>Fig 6 — Foot IMU placement (STRONG)</b>", s["h2"]))
    story.append(
        Paragraph(
            "Photo or line drawing: sensors on both feet, LF/RF labeled, path to PC. "
            "Keep treadmill in frame if possible.",
            s["caption"],
        )
    )

    # 3.5
    story.append(Paragraph("3.5 Synchronization and Recording Hub", s["h1"]))
    story.append(bullets([
        "OpenEye GUI as operator hub",
        "Live <b>sync.json</b>: Quest↔PC and Neon↔PC offsets",
        "Walking record blocked until both offsets exist",
        "All streams later interpreted on the <b>PC clock</b>",
    ], s["bullet"]))
    story.append(Paragraph("<b>Fig 7 — Timing / sync schematic (STRONG)</b>", s["h2"]))
    story.append(
        Paragraph(
            "Simple timeline: Quest clock, Neon clock, IMU receive time → shift by offsets → shared PC time. "
            "Conceptual only (no Hampel / 200 Hz detail).",
            s["caption"],
        )
    )

    # 3.6
    story.append(Paragraph("3.6 Data Products (optional)", s["h1"]))
    story.append(
        Paragraph(
            "Only if space allows or reviewers may doubt multimodal alignment. One paragraph: streams aligned "
            "to a common PC clock and prepared for Fitts, gaze, and gait analyses.",
            s["body"],
        )
    )
    story.append(Paragraph("<b>Fig 8 — High-level data flow (NICE)</b>", s["h2"]))
    story.append(
        Paragraph(
            "00_raw → align → shared grid → Fitts events + gaze + gait features. "
            "Stop at “analysis-ready streams”; do not list every script stage.",
            s["caption"],
        )
    )

    story.append(Paragraph("Figure priority (CHI page budget)", s["h1"]))
    story.append(
        table(
            ["Priority", "Figure", "Why"],
            [
                ["Must", "Fig 1 System overview", "Explains the stack in one glance"],
                ["Must", "Fig 4 Quest + Neon worn photo", "OpenEye contribution made concrete"],
                ["Must", "Fig 2 Ring / Rectangle task", "Bridges System → Method"],
                ["Strong", "Fig 6 Foot IMUs", "Makes gait sensing believable"],
                ["Strong", "Fig 7 Sync timeline", "Justifies multimodal claims"],
                ["Nice", "Fig 3 Three modalities", "Cheap clarity for IVs"],
                ["Nice", "Fig 5 / 8 Calib or pipeline", "Only if space / reviewer risk"],
            ],
            [0.9 * inch, 2.4 * inch, 3.5 * inch],
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("What NOT to put in System", s["h1"]))
    story.append(bullets([
        "Participant N, counterbalancing, A×W as design factors → Method",
        "MT / TP / RM-ANOVA → Method / Results",
        "Full cleaning (Hampel, drop Δt, 200 Hz fill) → Supplement",
        "Long Fitts theory → Related Work",
    ], s["bullet"]))

    story.append(Paragraph("How System relates to other sections", s["h1"]))
    story.append(
        table(
            ["Topic", "Earlier / System", "Method keeps"],
            [
                ["OpenEye Neon→Quest", "System §3.3 + Fig 4", "“Used OpenEye for Neon gaze” + calib note"],
                ["Foot IMU gait", "System §3.4 + Fig 6", "Events vs left-foot phase in Measures"],
                ["Clock sync", "System §3.5 + Fig 7", "One sentence: aligned to PC clock"],
                ["Fitts app / layouts", "System §3.2 + Fig 2", "IDs, hits, factors, procedure"],
            ],
            [1.4 * inch, 2.6 * inch, 2.8 * inch],
        )
    )

    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    return path


def build_method_pdf():
    s = styles()
    path = OUT_DIR / "CHI_Method_Section_Outline.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="CHI Method Section Outline — gazeGait",
        author="gazeGait writing guide",
    )
    story = []

    story.append(Paragraph("CHI Paper — Method / Study Section Outline", s["title"]))
    story.append(
        Paragraph(
            "gazeGait · Standing vs walking × head/hand/eye Fitts · Writing guide for §4 Method",
            s["subtitle"],
        )
    )

    story.append(Paragraph("Purpose of this section", s["h1"]))
    story.append(
        Paragraph(
            "Method (often titled <b>Study</b> or <b>Experiment</b> at CHI) describes <b>how this study used</b> "
            "the System: who participated, task parameters, factorial design, procedure, measures, and analysis. "
            "Do not re-introduce OpenEye architecture, foot-IMU hardware design, or Fitts theory—point back to "
            "Related Work and System.",
            s["body"],
        )
    )

    story.append(Paragraph("Recommended structure", s["h1"]))
    story.append(
        Paragraph(
            "4 Method<br/>"
            "4.1 Participants<br/>"
            "4.2 Apparatus<br/>"
            "4.3 Task<br/>"
            "4.4 Experimental Design<br/>"
            "4.5 Procedure<br/>"
            "4.6 Measures<br/>"
            "4.7 Data Analysis",
            s["mono"],
        )
    )
    story.append(
        Paragraph(
            "<b>Opener paragraph:</b> Within-subjects Fitts pointing in VR, comparing three modalities under "
            "standing vs walking, with synchronized gaze + foot IMU on walking trials.",
            s["note"],
        )
    )

    # 4.1
    story.append(Paragraph("4.1 Participants", s["h1"]))
    story.append(
        Paragraph(
            "Fill when N is final. CHI reviewers look for demographics and ethics early.",
            s["body"],
        )
    )
    story.append(bullets([
        "N, age mean/SD, gender, handedness, vision correction, XR experience",
        "Exclusion criteria (incomplete sync, motion sickness, failed calib)",
        "Ethics / IRB + informed consent",
        "Compensation",
    ], s["bullet"]))

    # 4.2
    story.append(Paragraph("4.2 Apparatus", s["h1"]))
    story.append(
        Paragraph(
            "Keep short. Reference System for OpenEye / IMU / sync detail. Focus on what participants experienced.",
            s["body"],
        )
    )
    story.append(bullets([
        "Meta Quest 3 + custom Unity Fitts app; wall at 2 m",
        "Modalities: HeadPinch, HandPinch, EyePinch (Neon via OpenEye); pinch confirm",
        "Treadmill for walking; dual foot IMUs on walking bouts only",
        "One sentence: streams aligned to PC clock via OpenEye hub",
    ], s["bullet"]))
    story.append(
        Paragraph(
            "Optional Method figure: lab photo (participant on treadmill with Quest + Neon + foot IMUs). "
            "If Fig 4/6 already appear in System, do not duplicate.",
            s["caption"],
        )
    )

    # 4.3
    story.append(Paragraph("4.3 Task", s["h1"]))
    story.append(
        Paragraph(
            "Angular Fitts selection on a virtual wall. Same six A×W pairs for ring and rectangle.",
            s["body"],
        )
    )
    story.append(
        table(
            ["Parameter", "Value"],
            [
                ["Layouts", "Ring (11 targets, step 5); Rectangle (horizontal L/R, height 30°)"],
                ["Widths W", "{3, 4, 5}°"],
                ["Amplitudes A", "{30, 20}°"],
                ["Shannon ID", "log2(A/W+1) ≈ 2.32–3.46"],
                ["Counted hits per ID", "10 (rectangle drops opening target)"],
                ["Per-target timeout", "5.0 s"],
                ["Rectangle ID pause", "0.5 s (not in MT)"],
                ["Quest log rate", "100 Hz"],
            ],
            [1.8 * inch, 5.0 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "CHI tip: one figure with ring + rectangle and A/W callouts is enough (may reuse System Fig 2).",
            s["caption"],
        )
    )

    # 4.4
    story.append(Paragraph("4.4 Experimental Design", s["h1"]))
    story.append(
        Paragraph(
            "Within-subjects factorial. State clearly: standing = practice/baseline; walking = main gait-coupled analysis.",
            s["body"],
        )
    )
    story.append(
        table(
            ["Factor", "Levels"],
            [
                ["Locomotion", "Standing, Walking"],
                ["Modality", "HeadPinch, HandPinch, EyePinch"],
                ["Layout", "Ring, Rectangle"],
            ],
            [1.5 * inch, 5.3 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "<b>12 recordings per person</b> = 4 bouts × 3 modalities:",
            s["body"],
        )
    )
    story.append(
        table(
            ["Bout", "Posture", "Layout", "ring_sets"],
            [
                ["PracticeRing", "standing", "ring", "2"],
                ["PracticeRectangle", "standing", "rectangle", "2"],
                ["Ring", "walking", "ring", "3"],
                ["Rectangle", "walking", "rectangle", "3"],
            ],
            [1.7 * inch, 1.2 * inch, 1.5 * inch, 1.2 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Also state counterbalancing / block order (or fixed order if that was the locked protocol).",
            s["caption"],
        )
    )

    # 4.5
    story.append(Paragraph("4.5 Procedure", s["h1"]))
    story.append(bullets([
        "Consent; fit Neon + Quest; foot IMUs for walk",
        "OpenEye calibration (25 dots) inside Main Study",
        "Standing practice: PracticeRing / PracticeRectangle × 3 modalities (state actual order)",
        "Treadmill familiarization — <b>fill speed policy</b> (fixed / self-selected)",
        "Walking Ring / Rectangle × 3 modalities",
        "Breaks and debrief",
        "Note: calib may be reused from matching practice bout if a walk bout lacked calib",
    ], s["bullet"]))

    # 4.6
    story.append(Paragraph("4.6 Measures", s["h1"]))
    story.append(
        Paragraph(
            "Organize by research question so Results can mirror this subsection.",
            s["body"],
        )
    )
    story.append(Paragraph("Selection performance", s["h2"]))
    story.append(bullets([
        "Movement time (MT), dwell, hit rate",
        "Throughput TP = ID / MT",
        "Effective width We = 4.133σ; IDe; TPe",
        "Ballistic vs homing (if reported)",
    ], s["bullet"]))
    story.append(Paragraph("Cursor / gaze dynamics", s["h2"]))
    story.append(bullets([
        "I-VT fixation and saccade counts (eye / head / hand cursors)",
    ], s["bullet"]))
    story.append(Paragraph("Gait coupling (walking only)", s["h2"]))
    story.append(bullets([
        "Foot initial contacts; Fitts events vs <b>left-foot gait phase %</b>",
        "Foot → pointer transfer H(f) in step band (~0.8–3 Hz)",
    ], s["bullet"]))
    story.append(Paragraph("Trial filtering", s["h2"]))
    story.append(
        Paragraph(
            "Drop training laps and first target of each A×W lap (ring first-dot; rectangle opening L); "
            "apply ISO-style filters as used in analysis scripts. Define MT onset precisely "
            "(e.g., target onset → pinch; include/exclude dwell).",
            s["body"],
        )
    )

    # 4.7
    story.append(Paragraph("4.7 Data Analysis", s["h1"]))
    story.append(
        Paragraph(
            "State the inferential unit explicitly—CHI reviewers care about this.",
            s["body"],
        )
    )
    story.append(bullets([
        "Unit of analysis = <b>participant</b> (person-cell medians/means), not pooled trials",
        "Per cell: standing|walking × modality × layout",
        "Across people: mean ± SE of person values",
        "Planned tests: paired standing vs walking; RM-ANOVA or linear mixed model on person-cells",
        "Nonparametrics (e.g., Wilcoxon) only if N and distributions warrant",
        "Cleaning detail stays out of Method body—one clause on PC-clock alignment is enough; rest → supplement",
    ], s["bullet"]))

    story.append(Paragraph("Practical writing order", s["h1"]))
    story.append(bullets([
        "1. Write 4.4 Design + 4.3 Task first (protocol is locked)",
        "2. Write 4.6 Measures (locks what Results will show)",
        "3. Write 4.2 Apparatus (short; point to System)",
        "4. Write 4.5 Procedure (session timeline)",
        "5. Write 4.7 Analysis (person-level stats plan)",
        "6. Fill 4.1 Participants last (when N is final)",
    ], s["bullet"]))

    story.append(Paragraph("Placeholders still to fill before submission", s["h1"]))
    story.append(bullets([
        "Final N, demographics, ethics ID",
        "Treadmill speed policy",
        "Counterbalancing / block order",
        "Exact MT definition",
        "Primary hypotheses H1–H3 (stand vs walk × modality contrasts)",
    ], s["bullet"]))

    story.append(Paragraph("Division of labor vs previous sections", s["h1"]))
    story.append(
        table(
            ["Content", "Where", "Method role"],
            [
                ["Fitts’ law / ID / We theory", "Related Work", "Your A/W, hits, filters only"],
                ["Eye/head/hand in XR", "Related Work", "Your three modalities"],
                ["Walking / locomotion HCI", "Related Work + Intro", "Stand vs walk as factors"],
                ["OpenEye + Neon on Quest 3", "System", "Used for EyePinch + calib"],
                ["Foot IMU + sync stack", "System", "Gait-phase measures on walk trials"],
            ],
            [2.0 * inch, 1.8 * inch, 3.0 * inch],
        )
    )

    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    return path


if __name__ == "__main__":
    p1 = build_system_pdf()
    p2 = build_method_pdf()
    print(p1)
    print(p2)
