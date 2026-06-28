"""Polished, privacy-conscious executive PDF report generation."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from fraudshield.analysis.eda import (
    infer_amount_columns,
    infer_binary_target_columns,
    infer_fraud_value,
    profile_target,
    target_values,
)
from fraudshield.data.quality import profile_dataset
from fraudshield.risk.scoring import risk_band_counts


if TYPE_CHECKING:
    from fraudshield.modeling.training import TrainingResult
    from fraudshield.risk.scoring import ScoringResult


NAVY = HexColor("#101820")
CYAN = HexColor("#0891B2")
RED = HexColor("#E11D48")
GREEN = HexColor("#059669")
AMBER = HexColor("#D97706")
INK = HexColor("#17212B")
MUTED = HexColor("#607080")
LINE = HexColor("#D9E1E8")
PALE = HexColor("#F4F7F9")


class NumberedCanvas(canvas.Canvas):
    """Canvas that adds a stable header, footer, and Page X of Y."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self) -> None:  # noqa: N802 - ReportLab API name
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        page_count = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_header_footer(page_count)
            super().showPage()
        super().save()

    def _draw_header_footer(self, page_count: int) -> None:
        width, height = A4
        self.saveState()
        if self._pageNumber > 1:
            self.setStrokeColor(LINE)
            self.setLineWidth(0.6)
            self.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
            self.setFillColor(MUTED)
            self.setFont("Helvetica", 7.5)
            self.drawString(18 * mm, height - 11 * mm, "FRAUDSHIELD AI")
            self.drawRightString(
                width - 18 * mm,
                height - 11 * mm,
                "EXECUTIVE RISK REPORT",
            )
        self.setStrokeColor(LINE)
        self.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
        self.setFillColor(MUTED)
        self.setFont("Helvetica", 7.5)
        self.drawString(18 * mm, 9 * mm, "Decision-support prototype - Human review required")
        self.drawRightString(
            width - 18 * mm,
            9 * mm,
            f"Page {self._pageNumber} of {page_count}",
        )
        self.restoreState()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=30,
            textColor=colors.white,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=HexColor("#C7D2DA"),
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=8,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=INK,
            spaceBefore=7,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=13,
            textColor=INK,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10.5,
            textColor=MUTED,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=RED,
        ),
        "center": ParagraphStyle(
            "Center",
            parent=base["BodyText"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=INK,
        ),
    }


def _format_number(value: Any, decimals: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _table(
    data: list[list[Any]],
    widths: list[float],
    *,
    header_background: colors.Color = NAVY,
    row_backgrounds: tuple[colors.Color, colors.Color] = (colors.white, PALE),
) -> LongTable:
    header_style = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=7.4,
        leading=9.5,
        textColor=colors.white,
    )
    body_style = ParagraphStyle(
        "TableBody",
        fontName="Helvetica",
        fontSize=7.4,
        leading=9.5,
        textColor=INK,
    )
    wrapped_data = []
    for row_index, row in enumerate(data):
        style = header_style if row_index == 0 else body_style
        wrapped_data.append([Paragraph(escape(str(cell)), style) for cell in row])
    table = LongTable(wrapped_data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), header_background),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("LEADING", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_index in range(1, len(data)):
        commands.append(
            ("BACKGROUND", (0, row_index), (-1, row_index), row_backgrounds[row_index % 2])
        )
    table.setStyle(TableStyle(commands))
    return table


def _metric_cards(metrics: list[tuple[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    cells: list[Table] = []
    for label, value in metrics:
        cells.append(
            Table(
                [
                    [Paragraph(value, styles["center"])],
                    [Paragraph(label.upper(), styles["small"])],
                ],
                colWidths=[39 * mm],
                hAlign="CENTER",
            )
        )
    cards = Table([cells], colWidths=[42 * mm] * len(cells), hAlign="LEFT")
    cards.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return cards


def _cover(
    styles: dict[str, ParagraphStyle],
    app_version: str,
    generated_at: datetime,
    dataset_name: str,
) -> list[Any]:
    cover_content = [
        Paragraph("FRAUDSHIELD AI", styles["subtitle"]),
        Spacer(1, 4 * mm),
        Paragraph("Executive Fraud Risk Report", styles["title"]),
        Paragraph(
            "Dataset quality, fraud patterns, model evaluation, and investigation priorities",
            styles["subtitle"],
        ),
        Spacer(1, 12 * mm),
        Paragraph(
            f"Version {app_version}<br/>Generated {generated_at.strftime('%d %b %Y, %H:%M UTC')}"
            f"<br/>Dataset: {escape(dataset_name)}",
            styles["subtitle"],
        ),
    ]
    cover = Table([[cover_content]], colWidths=[174 * mm], rowHeights=[92 * mm])
    cover.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 14 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 14 * mm),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return [cover, Spacer(1, 8 * mm)]


def build_executive_report(
    frame: pd.DataFrame,
    *,
    app_version: str,
    dataset_name: str = "In-session transaction dataset",
    training_result: TrainingResult | None = None,
    scoring_result: ScoringResult | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    """Generate an executive PDF without embedding raw transaction records."""
    if frame.empty:
        raise ValueError("The executive report requires at least one transaction row.")
    generated_at = generated_at or datetime.now(UTC)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)

    styles = _styles()
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title="FraudShield AI Executive Fraud Risk Report",
        author="FraudShield AI",
        subject="Fraud risk decision-support summary",
    )
    story: list[Any] = []
    story.extend(_cover(styles, app_version, generated_at, dataset_name))
    story.append(
        Paragraph(
            "DECISION-SUPPORT NOTICE",
            styles["callout"],
        )
    )
    story.append(
        Paragraph(
            "This report summarizes model signals for human review. A score is not proof of "
            "fraud and must not be the sole basis for blocking a transaction or penalizing "
            "a customer.",
            styles["body"],
        )
    )
    story.append(PageBreak())

    quality = profile_dataset(frame)
    story.append(Paragraph("1. Executive Summary", styles["section"]))
    story.append(
        _metric_cards(
            [
                ("Transactions", f"{quality.row_count:,}"),
                ("Features", f"{quality.column_count:,}"),
                ("Data quality", f"{quality.quality_score:.1f}/100"),
                ("Duplicates", f"{quality.duplicate_rows:,}"),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 5 * mm))
    summary_lines = [
        f"- Dataset quality is rated {quality.quality_band.lower()} at "
        f"{quality.quality_score:.1f}/100.",
        f"- Missing data affects {quality.missing_percentage:.2f}% of all cells.",
        f"- {quality.duplicate_rows:,} duplicate rows and "
        f"{len(quality.constant_columns):,} constant columns were detected.",
    ]
    if training_result is not None:
        best_run = training_result.runs[training_result.best_model_name]
        summary_lines.append(
            f"- The current recommended model candidate is {best_run.display_name}, "
            "ranked by holdout PR-AUC."
        )
    if scoring_result is not None:
        scored = scoring_result.frame
        review_count = int((scored["FS_Decision"] == "Manual review").sum())
        critical_count = int((scored["FS_Risk_Band"] == "Critical").sum())
        summary_lines.append(
            f"- {review_count:,} transactions enter manual review; "
            f"{critical_count:,} are in the Critical band."
        )
    for line in summary_lines:
        story.append(Paragraph(line, styles["body"]))

    story.append(Paragraph("2. Data Quality", styles["section"]))
    quality_data = [
        ["Measure", "Result", "Interpretation"],
        ["Rows", f"{quality.row_count:,}", "Transactions available"],
        ["Columns", f"{quality.column_count:,}", "Source fields available"],
        [
            "Missing cells",
            f"{quality.missing_cells:,}",
            f"{quality.missing_percentage:.2f}% of cells",
        ],
        [
            "Duplicate rows",
            f"{quality.duplicate_rows:,}",
            f"{quality.duplicate_percentage:.2f}% of rows",
        ],
        [
            "Empty columns",
            f"{len(quality.empty_columns):,}",
            "Should be removed before training",
        ],
        [
            "Constant columns",
            f"{len(quality.constant_columns):,}",
            "No predictive variation",
        ],
        [
            "Formula-like cells",
            f"{quality.formula_like_cells:,}",
            "Review before spreadsheet export",
        ],
    ]
    story.append(_table(quality_data, [46 * mm, 33 * mm, 92 * mm]))

    targets = infer_binary_target_columns(frame)
    if targets:
        target_column = targets[0]
        values = target_values(frame, target_column)
        fraud_value = infer_fraud_value(values)
        amounts = infer_amount_columns(frame, exclude=(target_column,))
        amount_column = amounts[0] if amounts else None
        fraud_profile = profile_target(frame, target_column, fraud_value, amount_column)
        story.append(Paragraph("3. Labeled Fraud Profile", styles["section"]))
        fraud_data = [
            ["Measure", "Result"],
            ["Target column", target_column],
            ["Selected fraud label", str(fraud_value)],
            ["Valid labeled rows", f"{fraud_profile.valid_transactions:,}"],
            ["Fraud rows", f"{fraud_profile.fraud_transactions:,}"],
            ["Fraud rate", f"{fraud_profile.fraud_rate:.2f}%"],
            ["Missing target rows", f"{fraud_profile.missing_target:,}"],
        ]
        if fraud_profile.fraud_amount is not None:
            fraud_data.extend(
                [
                    ["Fraud-linked amount", f"{fraud_profile.fraud_amount:,.2f}"],
                    [
                        "Fraud amount exposure",
                        f"{fraud_profile.fraud_amount_rate:.2f}%"
                        if fraud_profile.fraud_amount_rate is not None
                        else "N/A",
                    ],
                ]
            )
        story.append(_table(fraud_data, [72 * mm, 99 * mm], header_background=CYAN))

    if training_result is not None:
        story.append(PageBreak())
        story.append(Paragraph("4. Model Evaluation", styles["section"]))
        story.append(
            Paragraph(
                "All preprocessing and fitting occur inside leakage-safe pipelines. The "
                "leaderboard uses an untouched stratified holdout; PR-AUC is the primary "
                "ranking measure.",
                styles["body"],
            )
        )
        leaderboard = training_result.leaderboard
        model_data = [["Model", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]]
        for _, row in leaderboard.iterrows():
            model_data.append(
                [
                    str(row["Model"]),
                    _format_number(row["Precision"]),
                    _format_number(row["Recall"]),
                    _format_number(row["F1"]),
                    _format_number(row["PR-AUC"]),
                    _format_number(row["ROC-AUC"]),
                ]
            )
        story.append(
            _table(
                model_data,
                [47 * mm, 24 * mm, 24 * mm, 22 * mm, 27 * mm, 27 * mm],
                header_background=CYAN,
            )
        )
        story.append(Spacer(1, 4 * mm))
        story.append(
            Paragraph(
                f"Training rows: {training_result.train_rows:,} | Holdout rows: "
                f"{training_result.test_rows:,} | Holdout fraud rows: "
                f"{training_result.test_fraud:,}",
                styles["small"],
            )
        )

    if scoring_result is not None:
        story.append(Paragraph("5. Investigation Queue", styles["section"]))
        scored = scoring_result.frame
        counts = risk_band_counts(scored)
        risk_data = [["Risk band", "Transactions", "Share"]]
        for _, row in counts.iterrows():
            risk_data.append(
                [
                    str(row["Risk band"]),
                    f"{int(row['Transactions']):,}",
                    f"{float(row['Share %']):.2f}%",
                ]
            )
        story.append(_table(risk_data, [57 * mm] * 3, header_background=RED))
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph("Highest-risk queue entries", styles["subsection"]))
        top = scored.nlargest(10, "FS_Risk_Score")
        queue_data = [["Row", "Source index", "Risk score", "Band", "Decision"]]
        for _, row in top.iterrows():
            queue_data.append(
                [
                    str(int(row["FS_Row_Number"])),
                    str(row["FS_Source_Index"])[:30],
                    f"{float(row['FS_Risk_Score']):.2f}",
                    str(row["FS_Risk_Band"]),
                    str(row["FS_Decision"]),
                ]
            )
        story.append(
            _table(
                queue_data,
                [18 * mm, 45 * mm, 30 * mm, 30 * mm, 48 * mm],
                header_background=RED,
            )
        )
        story.append(
            Paragraph(
                f"Scoring model: {scoring_result.model_display_name} | Manual-review threshold: "
                f"{scoring_result.decision_threshold:.2f}",
                styles["small"],
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("6. Governance Recommendations", styles["section"]))
    recommendations = [
        "- Keep human review between model output and any customer-impacting action.",
        "- Select thresholds using fraud loss, investigation capacity, and false-alert cost.",
        "- Validate on future time periods when the production task predicts future transactions.",
        "- Monitor data drift, fraud-rate drift, calibration, recall, and false-alert volume.",
        "- Review proxy variables, access controls, retention rules, and audit logging "
        "before deployment.",
        "- Retrain only through a versioned approval process with repeatable evaluation.",
    ]
    for recommendation in recommendations:
        story.append(Paragraph(recommendation, styles["body"]))

    story.append(Paragraph("7. Methodology and Limitations", styles["section"]))
    limitations = [
        "- Data-quality scoring measures completeness, uniqueness, and non-constant "
        "columns; it is not model accuracy.",
        "- Correlation and feature influence do not establish real-world causation.",
        "- Scores produced for training rows may be optimistic and must not replace "
        "holdout metrics.",
        "- Small datasets can demonstrate workflow but cannot support reliable "
        "performance claims.",
        "- Joblib artifacts must be loaded only from trusted sources because they use "
        "Python pickle.",
        "- Production use requires authentication, encryption, privacy review, monitoring, "
        "and incident response.",
    ]
    for limitation in limitations:
        story.append(Paragraph(limitation, styles["body"]))

    story.append(Paragraph("8. Production Readiness Checklist", styles["section"]))
    readiness_data = [
        ["Control", "Portfolio status", "Production requirement"],
        ["Authentication and authorization", "Not included", "Required"],
        ["Encryption and secret management", "Local only", "Required"],
        ["Decision and access audit logs", "Basic app log", "Required"],
        ["Data retention and deletion policy", "Not included", "Required"],
        ["Performance and drift monitoring", "Not included", "Required"],
        ["Incident response process", "Not included", "Required"],
        ["Human-review operating procedure", "Conceptual", "Required"],
        ["Privacy, legal, and fairness approval", "Not included", "Required"],
    ]
    story.append(
        _table(
            readiness_data,
            [76 * mm, 47 * mm, 48 * mm],
            header_background=AMBER,
        )
    )

    document.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
