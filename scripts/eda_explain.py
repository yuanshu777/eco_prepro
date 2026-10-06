"""Render the English reading guide beside a revised frozen EDA report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REVIEWED_INVENTORY_SHA256 = "147b984c70cbef74c1a5b40f08745a2bb754186678b4f7415530297a5a1a7d60"
FIGURES = {
    "4.1": "01_views",
    "4.2": "02_source_view",
    "4.3": "03_duration",
    "4.4": "04_size_fps",
    "4.5": "08_mimic_structure",
    "4.6": "07_echonet_measurements",
    "4.7": "06_quality",
    "4.8": "09_mask_normal",
    "4.9": "10_mask_dark",
    "4.10": "11_transition",
    "4.11": "12_uncertain",
    "4.12": "13_ev9v_group_heatmap",
    "4.13": "14_ev9v_group_diversity",
    "4.14": "15_ev9v_group_coverage",
    "4.15": "16_ev9v_group_combinations",
    "4.16": "17_ev9v_group_sensitivity",
    "4.17": "18_ev9v_group_merged",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--text", type=Path, default=Path("docs/EDA_EXPLANATION_EN.txt"))
    args = ap.parse_args()
    out = args.out
    inventory = out / "tables/video_inventory.csv"
    if hashlib.sha256(inventory.read_bytes()).hexdigest() != REVIEWED_INVENTORY_SHA256:
        ap.error("The guide describes the reviewed 2026-10-05 inventory only.")
    source = args.text.read_text()
    fields = json.loads((out / "data_dictionary.json").read_text())["fields"]
    if len(fields) != 69:
        ap.error("Expected the reviewed 69-column data dictionary.")

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#17354c")
    body = ParagraphStyle(
        "GuideBody",
        parent=styles["BodyText"],
        fontSize=10.2,
        leading=14.2,
        spaceAfter=8,
        textColor=navy,
    )
    heading = ParagraphStyle(
        "GuideHeading",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        spaceAfter=12,
        textColor=navy,
    )
    cell = ParagraphStyle("GuideCell", parent=body, fontSize=9.1, leading=12, spaceAfter=0)
    story = []
    web = []
    figure_ids = []

    def add_figure(name):
        path = out / "figures" / f"{name}.png"
        with PILImage.open(path) as im:
            width, height = im.size
        scale = min(760 / width, 245 / height)
        story.extend([Image(str(path), width=width * scale, height=height * scale), Spacer(1, 10)])
        web.append(f'<img src="figures/{name}.png" alt="{name}">')
        figure_ids.append(name)

    for index, paragraph in enumerate(source.strip().split("\n\n")):
        is_figure = " | " in paragraph and paragraph.split(" | ")[0] in FIGURES
        is_heading = index == 0 or paragraph.isupper() or is_figure
        if is_heading:
            if story:
                story.append(PageBreak())
                web.append("</section>")
            web.append("<section>")
        story.append(Paragraph(html.escape(paragraph), heading if is_heading else body))
        tag = "h1" if is_heading else "p"
        web.append(f"<{tag}>{html.escape(paragraph)}</{tag}>")
        if is_figure:
            add_figure(FIGURES[paragraph.split(" | ")[0]])
        if paragraph.startswith("MISSINGNESS: HOW TO READ"):
            add_figure("05_missingness")

    items = list(fields.items())
    for start in range(0, len(items), 12):
        story.append(PageBreak())
        title = f"Inventory fields {start + 1}-{min(start + 12, len(items))} of 69"
        story.append(Paragraph(title, heading))
        rows = [[Paragraph("Field", cell), Paragraph("Meaning", cell)]]
        rows += [
            [Paragraph(html.escape(k), cell), Paragraph(html.escape(v), cell)]
            for k, v in items[start : start + 12]
        ]
        table = Table(rows, colWidths=[190, 570], repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf4f6")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                ]
            )
        )
        story.append(table)
        web.append(
            f"</section><section><h1>{title}</h1><table><tr><th>Field</th><th>Meaning</th></tr>"
        )
        web.extend(
            f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>"
            for k, v in items[start : start + 12]
        )
        web.append("</table>")

    def footer(canvas, doc):
        canvas.setFillColor(navy)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(40, 20, "English explanation guide | revised EDA | snapshot 2026-10-05")
        canvas.drawRightString(802, 20, str(doc.page))

    doc = SimpleDocTemplate(
        str(out / "EDA_Explanation_EN.pdf"),
        pagesize=landscape(A4),
        leftMargin=40,
        rightMargin=40,
        topMargin=30,
        bottomMargin=40,
        title="Echocardiography EDA: English Explanation Guide",
        author="eco_prepro",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    css = "body{font-family:Arial,sans-serif;line-height:1.6;color:#17354c;background:#edf4f6}main{max-width:1150px;margin:auto}section{padding:30px;margin:20px 0;background:white}h1{font-size:24px}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ccd;padding:9px;text-align:left;vertical-align:top}td:first-child{font-family:monospace}"
    (out / "EDA_Explanation_EN.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>EDA English Explanation</title>'
        f"<style>{css}</style><main>" + "".join(web) + "</section></main></html>"
    )
    (out / "EDA_Explanation_EN.txt").write_text(
        source
        + "\n\nALL 69 INVENTORY FIELDS\n\n"
        + "\n\n".join(f"{k}: {v}" for k, v in fields.items())
        + "\n"
    )
    (out / "explanation_manifest.json").write_text(
        json.dumps(
            {
                "source_text": str(args.text),
                "source_text_sha256": hashlib.sha256(args.text.read_bytes()).hexdigest(),
                "inventory_sha256": REVIEWED_INVENTORY_SHA256,
                "explained_figures": figure_ids,
                "inventory_fields": len(fields),
                "scope": "English definitions and revised-report interpretation; no new analysis or annotations",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"English guide generated: {len(figure_ids)} figures, {len(fields)} inventory fields.")


if __name__ == "__main__":
    main()
