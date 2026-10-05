"""Render the snapshot-specific English report and HTML for the 2026-10-05 EDA.

The interpretive narrative contains reviewed findings from this frozen inventory.
Reject a different inventory so later downloads cannot inherit stale claims.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import pandas as pd
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

NAVY = colors.HexColor("#17354c")
TEAL = colors.HexColor("#147d80")
SOFT = colors.HexColor("#edf4f6")
REVIEWED_INVENTORY_SHA256 = "147b984c70cbef74c1a5b40f08745a2bb754186678b4f7415530297a5a1a7d60"


def fmt(x, digits=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{digits}f}"


class Report:
    def __init__(self, root):
        self.root = root
        self.story = []
        self.web = []
        styles = getSampleStyleSheet()
        self.body = ParagraphStyle(
            "EDABody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.3,
            leading=14.1,
            textColor=NAVY,
            spaceAfter=7,
        )
        self.title = ParagraphStyle(
            "EDATitle",
            parent=styles["Heading1"],
            fontSize=19,
            leading=23,
            textColor=NAVY,
            spaceAfter=12,
        )
        self.small = ParagraphStyle(
            "EDASmall", parent=self.body, fontSize=8.3, leading=11, spaceAfter=6
        )
        self.table_style = ParagraphStyle(
            "EDACell", parent=self.body, fontSize=9, leading=12, spaceAfter=0
        )
        self.figures = {
            r["name"]: r for r in json.loads((root / "figure_manifest.json").read_text())
        }
        self.page = 0

    def new_page(self, title):
        if self.page:
            self.story.append(PageBreak())
            self.web.append("</section>")
        self.page += 1
        self.story.append(Paragraph(title, self.title))
        self.web.append(f"<section><h1>{html.escape(title)}</h1>")

    def p(self, text, small=False):
        self.story.append(Paragraph(text, self.small if small else self.body))
        self.web.append(f"<p>{text}</p>")

    def table(self, headers, rows, widths=None):
        data = [[Paragraph(str(v), self.table_style) for v in headers]]
        data += [[Paragraph(str(v), self.table_style) for v in row] for row in rows]
        widths = widths or [770 / len(headers)] * len(headers)
        tab = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        tab.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), SOFT),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, TEAL),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.lightgrey),
                ]
            )
        )
        self.story += [tab, Spacer(1, 9)]
        self.web.append(
            "<table><thead><tr>"
            + "".join(f"<th>{v}</th>" for v in headers)
            + "</tr></thead><tbody>"
        )
        self.web += ["<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows]
        self.web.append("</tbody></table>")

    def figure(self, name, maxheight=290):
        path = self.root / "figures" / f"{name}.png"
        w, h = PILImage.open(path).size
        scale = min(770 / w, maxheight / h)
        self.story.append(Image(str(path), width=w * scale, height=h * scale, hAlign="LEFT"))
        self.story.append(Spacer(1, 7))
        self.web.append(
            f'<img src="figures/{name}.png" alt="{html.escape(self.figures[name]["caption"])}">'
        )
        self.p(self.figures[name]["caption"], small=True)

    def finish(self):
        self.web.append("</section>")
        style = """body{font-family:Arial,sans-serif;color:#17354c;background:#edf4f6;margin:0}main{max-width:1180px;margin:auto}section{background:white;padding:35px;margin:22px 0}h1{font-size:26px}p{line-height:1.5}table{border-collapse:collapse;width:100%;margin:20px 0}th{background:#edf4f6}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd;vertical-align:top}img{max-width:100%;height:auto}a{color:#147d80}@media print{section{break-before:page;margin:0}body{background:white}}"""
        (self.root / "EDA_Report_EN.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Echo EDA: data, methods and findings</title><style>'
            + style
            + "</style><main>"
            + "".join(self.web)
            + "</main></html>"
        )

        def footer(canvas, doc):
            canvas.setStrokeColor(colors.HexColor("#c4d3d9"))
            canvas.line(36, 29, 806, 29)
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(NAVY)
            canvas.drawString(
                36,
                18,
                "EV9V / MIMIC-IV-Echo / EchoNet-Dynamic | local snapshot 2026-10-05 | research EDA",
            )
            canvas.drawRightString(806, 18, str(doc.page))

        doc = SimpleDocTemplate(
            str(self.root / "EDA_Report_EN.pdf"),
            pagesize=landscape(A4),
            rightMargin=36,
            leftMargin=36,
            topMargin=30,
            bottomMargin=39,
            title="Echocardiography EDA: Data, Methodology and Results",
            author="eco_prepro",
        )
        doc.build(self.story, onFirstPage=footer, onLaterPages=footer)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.out
    inventory_hash = hashlib.sha256((root / "tables/video_inventory.csv").read_bytes()).hexdigest()
    if inventory_hash != REVIEWED_INVENTORY_SHA256:
        ap.error(
            "This report narrative was reviewed for the frozen 2026-10-05 inventory. "
            "Use that inventory, or update and review the narrative before changing its hash."
        )
    d = pd.read_csv(
        root / "tables/video_inventory.csv",
        low_memory=False,
        dtype={"subject_id": str, "study_id": str},
    )
    inv = json.loads((root / "inventory_summary.json").read_text())
    ana = json.loads((root / "analysis_provenance.json").read_text())
    summary = pd.read_csv(root / "tables/dataset_summary.csv").set_index("dataset")
    durations = pd.read_csv(root / "tables/bmode_duration_summary.csv").set_index("dataset")
    ev = d[d.dataset == "EV9V"]
    mi = d[d.dataset == "MIMIC-IV-Echo"]
    en = d[d.dataset == "EchoNet-Dynamic"]
    qc = pd.read_csv(root / "tables/qc_sample_results.csv")
    pred = pd.read_csv(root / "tables/mimic_predicted_views.csv")
    mask = pd.read_csv(root / "tables/masking_examples.csv")
    trace_missing = int((en.traced_frames == 0).sum())
    report = Report(root)
    report.new_page("0. Project introduction and goal")
    report.p(
        "<b>Echocardiography preprocessing and temporal view analysis.</b> The project develops a reproducible pipeline for reading heterogeneous ultrasound inputs, identifying the imaging region, preserving useful anatomy and timing, and preparing consistent inputs for cardiac-view analysis. EV9V supplies labelled raw-screen clips, MIMIC-IV-Echo supplies clinical DICOM objects, and EchoNet-Dynamic supplies already cropped A4C videos."
    )
    report.p(
        "This EDA examines whether the available data support reliable preprocessing and the study of stable views, possible view changes and uncertain intervals. It quantifies local coverage, formats, timing, label availability and image characteristics. True probe movement, temporary loss of visible evidence and a classifier changing its prediction are treated as different phenomena. No new classifier was trained or run."
    )
    report.p(
        "<b>1. Data overview.</b> Counts below refer to the frozen local inventory, not the sizes of all published datasets. MIMIC was still downloading; the scan started at 15:31 on October 5, 2026 (America/New_York). Later files are outside this snapshot."
    )
    rows = []
    for source, origin in [
        ("EV9V", "Chengdu Medical College; released view-classification clips"),
        ("MIMIC-IV-Echo", "BIDMC / PhysioNet; locally downloaded v1.0.1 subset"),
        ("EchoNet-Dynamic", "Stanford; preprocessed A4C videos"),
    ]:
        s = summary.loc[source]
        rows.append(
            [source, origin, f"{int(s.files):,}", f"{int(s.frames):,}", f"{s.total_hours:.2f}"]
        )
    report.table(
        ["Source", "Origin / role", "Local files", "Frames*", "Dynamic hours*"],
        rows,
        [130, 330, 90, 110, 110],
    )
    report.p(
        f"<b>Total:</b> {len(d):,} distinct file records and {int(d.n_frames.sum()):,} frames. These are not independent patients or observations. *Frame counts are from the manifest / headers; single-frame DICOMs count as one image, and hours include dynamic objects with known timing only. Header success does not establish full pixel decoding.",
        small=True,
    )

    report.new_page("1. Data structure and statistical units")
    report.table(
        ["Input table", "Rows x columns", "One row represents"],
        [
            [
                "EV9V native manifest",
                "5,138 x 14",
                "One labelled video; JPEG frames are a derivative, not extra videos",
            ],
            [
                "EchoNet FileList",
                "10,030 x 9",
                "One video with FPS, frame count, EF, ESV, EDV and split",
            ],
            [
                "EchoNet VolumeTracings",
                "425,010 x 6",
                "One coordinate pair within an annotated frame, not one video",
            ],
            [
                "MIMIC official record list",
                "524,137 x 4",
                "One expected DICOM object; most are not local",
            ],
            [
                "Unified local video_inventory",
                f"{len(d):,} x {len(d.columns)}",
                "One native local file, including stills and uncertain modes",
            ],
        ],
        [210, 150, 410],
    )
    report.p(
        "<b>Hierarchy:</b> MIMIC patient -&gt; study -&gt; stored DICOM loop/image -&gt; frame. EV9V and EchoNet provide video/file identity, but this local release does not provide an independently verifiable patient-linkage table. Frame-level rows must not be treated as independent subjects."
    )
    report.p(
        f"<b>EV9V grouping:</b> removing the final clip suffix yields {ev.original_id.str.rsplit('_', n=1).str[0].nunique():,} recording keys; using the prefix before the resolution string yields {ev.session.nunique():,} coarser groups. Neither is a verified patient ID. Official train/validation/test splits are preserved; the coarser groups do not cross splits."
    )
    report.p(
        "<b>Data flow:</b> frozen file list -&gt; header/manifest validation -&gt; source-aware metadata table -&gt; full-population descriptive plots + fixed-seed sampled pixel QC -&gt; illustrated masking and native-video candidates. Separate tables record sampled frames and proposed temporal intervals. No global resizing, frame dropping, oversampling or class balancing is applied before distribution statistics."
    )
    report.p(
        "<b>Important proxies:</b> released clip labels are not frame-level truth; MIMIC model probabilities are not annotations; screen pixel dimensions are not spatial resolution; brightness and sharpness summaries are not clinical quality scores."
    )

    report.new_page("2. Preprocessing and inclusion decisions")
    report.table(
        ["Operation", "Applied rule", "What is preserved"],
        [
            [
                "Inventory / duplicate handling",
                "Match MIMIC relative paths to its official record list; one row per logical file. Check SOP identifiers.",
                "No repeated MIMIC SOP UID found. EV9V MP4 and JPEG representations are not counted twice.",
            ],
            [
                "Frame order",
                "Use numeric JPEG indices, and validate cache frame indices against original timestamps.",
                "All 5,138 EV9V frame folders match the manifest count and contiguous 0..N-1 numbering.",
            ],
            [
                "Timing",
                "Prefer valid frame intervals; store playback duration and first-to-last span separately.",
                "No default 30 FPS is substituted for missing DICOM timing. Keep alternative rate tags and conflicts.",
            ],
            [
                "Main motion cohort",
                "EV9V and EchoNet videos; MIMIC dynamic objects whose region metadata is entirely 2D tissue.",
                "Other/mixed imaging modes, unknown-mode loops and stills remain in the full inventory.",
            ],
            [
                "Pixel QC",
                "54 EV9V + 60 MIMIC + 36 EchoNet videos; up to 16 ordered frames each.",
                "Raw data unchanged. A 128x128 analysis copy is used only for comparable pixel summaries.",
            ],
            [
                "Mask examples",
                "Run the existing detector on two complete native EV9V MP4s. A fixed mask is applied only if accepted.",
                "Low-confidence input is retained; EchoNet remains a pass-through source.",
            ],
        ],
        [125, 345, 300],
    )
    report.p(
        "<b>Scope of verification:</b> every inventoried file has a readable header; sampled pixels decode in all 150 selected cines. This is not a claim that every frame in all 32,506 files was decoded or that all sources were geometrically processed. No MIMIC layout profile was silently inherited from EV9V.",
        small=True,
    )

    report.new_page("3. Missingness and information availability")
    report.figure("05_missingness", maxheight=260)
    report.table(
        ["Finding", "Quantity / interpretation", "Handling"],
        [
            [
                "MIMIC timing",
                "0/9,489 dynamic objects lack frame timing; 7,849 stills have no temporal rate.",
                "Stills are not included in the missing-video-timing denominator.",
            ],
            [
                "MIMIC view / mode",
                "17,338 objects lack validated view labels; 551 have no region-mode metadata.",
                "Retain Unknown; show model-based view counts separately.",
            ],
            [
                "EchoNet tracings",
                f"{trace_missing}/10,030 videos lack matched tracings (0.060%); one tracing filename is unmatched.",
                "Keep these videos for view/metadata EDA; flag tracing-dependent analyses.",
            ],
            [
                "EV9V mapping",
                "453/5,138 (8.8%) have raw PMPALA labels but no agreed five-family mapping.",
                "Preserve raw nine-class counts and an Unmapped group.",
            ],
        ],
        [135, 360, 275],
    )
    report.p(
        "For single-frame DICOMs, an absent NumberOfFrames tag is interpreted as one image (7,849 objects); this inference is recorded separately. No mean/mode imputation is used. Distinguish not provided, not applicable, unavailable locally, parse failure and conflicting metadata. Expert transition labels are unavailable for all sources; this is the main limitation for a true transition-accuracy claim.",
        small=True,
    )

    report.new_page("4.1 View distributions: keep label sources separate")
    report.figure("01_views", maxheight=255)
    report.p(
        f"<b>EV9V:</b> parasternal long-axis (PLAX) has 1,580 videos ({1580 / len(ev) * 100:.1f}%), parasternal short-axis (PSAX) has 1,332 ({1332 / len(ev) * 100:.1f}%), and apical four-chamber (A4C) has 1,237 ({1237 / len(ev) * 100:.1f}%). PSAX combines PASA, PMVLSA, PPMLSA and PMASA. The 453 PMPALA videos remain separate because their anatomical mapping is unresolved."
    )
    report.p(
        f"<b>MIMIC:</b> {len(pred):,} existing EchoPrime caches match the current file records. The figure uses the argmax of the mean raw 11-class probability over sampled frames. This is an explicitly defined prediction aggregation, not an expert annotation or the true prevalence of cardiac views. {len(mi) - len(pred):,} local objects have no reused prediction in this plot."
    )
    report.p(
        "EV9V has no apical two-chamber (A2C) category in its released labels; this does not establish that A2C anatomy never appears within individual frames."
    )
    report.p(
        "MIMIC cached predictions include Doppler-labelled outputs even though the selected loops have tissue-only region metadata. That mismatch is evidence for reviewing the predictor or mode metadata, not proof of either actual mode or view.",
        small=True,
    )

    report.new_page("4.2 Source-view confounding and weighting")
    report.figure("02_source_view", maxheight=275)
    report.p(
        "<b>The unit changes the answer.</b> Video-weighted class shares give each cine one vote. Time-weighted shares give long cines more influence. EV9V PLAX occupies a larger share of video time than of clip count; the report provides both denominators instead of pooling frames as independent examples."
    )
    report.p(
        "<b>Dataset identity can become a shortcut.</b> EchoNet contributes only A4C and has a distinctive 112x112 cropped canvas, whereas EV9V carries a raw 320x240 screen and MIMIC uses larger clinical DICOMs. A pooled classifier could associate source appearance with view. Later comparisons should include within-source results and shared-view comparisons, particularly A4C."
    )
    report.p(
        "<b>Grouping and splits:</b> keep the official EV9V and EchoNet splits, group MIMIC by actual patient for any future split, and avoid placing JPEG derivatives or frames of one video in separate subsets. Source balancing is a later experimental choice, not an EDA preprocessing step."
    )

    report.new_page("4.3 Video length distributions")
    report.figure("03_duration", maxheight=245)
    rows = []
    for src in ["EV9V", "MIMIC-IV-Echo", "EchoNet-Dynamic"]:
        x = durations.loc[src]
        rows.append(
            [
                src,
                f"{int(x['n']):,}",
                fmt(x["median"]),
                f"{x.p25:.2f}-{x.p75:.2f}",
                f"{int(x.under_2s):,} ({x.under_2s / x['n'] * 100:.1f}%)",
                f"{int(x.at_least_10s):,} ({x.at_least_10s / x['n'] * 100:.1f}%)",
                fmt(x.maximum),
            ]
        )
    report.table(
        ["Cohort", "Videos", "Median s", "IQR s", "Under 2 s", "At least 10 s", "Max s"],
        rows,
        [145, 75, 80, 105, 145, 145, 75],
    )
    report.p(
        "<b>Interpretation:</b> EV9V has a much heavier duration tail. MIMIC B-mode clips are predominantly short saved loops. Long clips provide more opportunity to contain a change, so future candidate rates must be stratified by length or expressed per unit time. The 2 s and 10 s bins are descriptive cutoffs, not clinical definitions.",
        small=True,
    )

    report.new_page("4.4 File type, dimensions and timing")
    report.table(
        ["Source", "Representation", "Files", "Native W x H", "Timing basis", "Primary status"],
        [
            [
                "EV9V",
                "MP4 + matched JPEG derivative",
                "5,138",
                "320 x 240",
                "30 FPS playback",
                "Raw screen; nine clip labels",
            ],
            [
                "MIMIC-IV-Echo",
                "DICOM",
                f"{len(mi):,}",
                "1016 x 708",
                "FrameTime for 9,489 dynamic objects",
                "Clinical stills and loops; views unlabelled",
            ],
            [
                "EchoNet-Dynamic",
                "AVI",
                "10,030",
                "112 x 112",
                "FileList; median 50 FPS",
                "Already cropped/masked; A4C",
            ],
        ],
        [125, 160, 65, 100, 180, 140],
    )
    report.figure("04_size_fps", maxheight=235)
    report.p(
        "<b>Timing conflicts:</b> 8 MIMIC objects differ by more than 5% between frame-time-derived FPS and another rate tag. The analysis uses FrameTime and records the alternatives. Container and table rates agree within 5% for EV9V and EchoNet. Native image size differs strongly by source; no model input size replaces it in these plots.",
        small=True,
    )

    report.new_page("4.5 MIMIC: coverage, study structure and imaging modes")
    report.figure("08_mimic_structure", maxheight=260)
    report.table(
        ["Population / subset", "Count", "Meaning"],
        [
            ["Official file inventory", "524,137", "Expected DICOM objects, not all downloaded"],
            [
                "Local snapshot",
                f"{len(mi):,} ({len(mi) / 524137 * 100:.2f}%)",
                f"{mi.subject_id.nunique()} patients; {inv['local_mimic_started_studies']} studies started; 15 partial files excluded",
            ],
            [
                "Complete studies",
                f"{inv['local_mimic_complete_studies']} studies / {inv['local_mimic_patients_complete']} patients",
                f"{int((mi.study_complete == True).sum()):,} objects; completeness refers to inventory matching",
            ],
            [
                "Dynamic tissue-only cohort",
                f"{int((mi.bmode_cine == True).sum()):,}",
                f"{int(((mi.bmode_cine == True) & (mi.study_complete == True)).sum()):,} are in complete studies",
            ],
        ],
        [230, 170, 370],
    )
    report.p(
        "This download is a convenience subset, not a randomized sample of all MIMIC. All local images have the same 1016x708 canvas and report one of two GE manufacturer names. Do not claim that these distributions characterize all vendors or the full release. A sequence of stored files is not a continuous probe sweep.",
        small=True,
    )

    report.new_page("4.6 EchoNet-Dynamic: measurements and annotations")
    report.figure("07_echonet_measurements", maxheight=250)
    report.table(
        ["Property", "Observed result", "Relevance"],
        [
            [
                "Train / validation / test",
                "7,465 / 1,288 / 1,277",
                "Retain the released split; no new mixing of videos or frames",
            ],
            [
                "EF / EDV / ESV missing values",
                "0 / 0 / 0",
                "Complete in FileList; does not imply universal annotation completeness",
            ],
            [
                "EF distribution",
                f"Median {en.ef.median():.2f}%; IQR {en.ef.quantile(0.25):.2f}-{en.ef.quantile(0.75):.2f}%",
                "Descriptive distribution; no clinical grouping or downstream experiment",
            ],
            [
                "Matched tracing coverage",
                "10,024/10,030 videos; two traced frames per matched video",
                "6 unmatched videos and 1 orphan tracing filename; do not silently drop them",
            ],
        ],
        [190, 260, 320],
    )
    report.p(
        "The released EF agrees numerically with 100*(EDV-ESV)/EDV to within 6.6e-8 percentage points. This checks table consistency only. The original source website describes the videos as pre-cropped and masked; our EDA leaves them intact and uses them as a preprocessing contrast, not an additional multi-view dataset.",
        small=True,
    )

    report.new_page("4.7 Image characteristics: sampled descriptive QC")
    report.figure("06_quality", maxheight=255)
    rows = []
    for src, how in [
        ("EV9V", "2 videos per raw-view x duration stratum"),
        ("MIMIC-IV-Echo", "1 B-mode loop per sampled patient, complete studies"),
        ("EchoNet-Dynamic", "4 videos per released-split x duration stratum"),
    ]:
        q = qc[qc.dataset == src]
        rows.append(
            [
                src,
                len(q),
                how,
                f"{q.brightness.median():.1f}",
                f"{q.nonblack_fraction.median():.3f}",
            ]
        )
    report.table(
        ["Source", "Videos", "Selection, fixed seed 20261005", "Brightness*", "Nonblack*"],
        rows,
        [135, 60, 385, 95, 95],
    )
    report.p(
        f"<b>Coverage:</b> {len(qc)} selected cines; {int(qc.n_decoded.sum()):,} requested frames decoded successfully. EV9V QC reads numerically ordered released JPEG frames; EchoNet and MIMIC read video/DICOM pixels. Known masking/transition examples are additional purposeful selections, not part of the sampling denominator."
    )
    report.p(
        "*Brightness is median frame mean on 0-255 grayscale; nonblack is the fraction above 6 after analysis resizing. Laplacian variance measures local high-frequency content. Source framing and compression affect all three. The apparent lower MIMIC brightness is not evidence of inferior clinical quality. No population failure rate is inferred from this sample.",
        small=True,
    )

    report.new_page("4.8 Dynamic masking example: accepted normal fan")
    report.figure("09_mask_normal", maxheight=350)
    normal = mask[mask.example == "Normal fan"].iloc[0]
    report.p(
        f"Native MP4: <b>{normal.status}</b>, output {int(normal.output_width)}x{int(normal.output_height)}, {int(normal.n_frames)} frames. The imaging mask covers {normal.mask_fraction * 100:.1f}% of the original canvas. This is motion-derived masking with one fixed mask per cine, not a mask that changes on every frame.",
        small=True,
    )
    report.p(
        f"<b>Representation sensitivity:</b> the corresponding released JPEG sequence yields {normal.matched_jpeg_status}. A shared source identity does not imply pixel-equivalent input or identical detector behavior. Keep representation provenance and avoid treating MP4/JPEG versions as independent observations.",
        small=True,
    )

    report.new_page("4.9 Dynamic masking example: uncertain dark sector")
    report.figure("10_mask_dark", maxheight=350)
    report.p(
        "The dark input remains <b>LOW_CONFIDENCE</b>. The proposed geometry is shown for diagnosis, but the output retains the original image and frame count. Low motion or weak boundary support can make the visible tissue support smaller than the true imaging sector.",
        small=True,
    )
    report.p(
        "These two examples establish behavior on selected inputs, not overall masking accuracy. No independent sector masks are available; neither clean-looking boundaries nor temporal stability measures anatomical preservation directly.",
        small=True,
    )

    report.new_page("4.10 Native-video frame transitions: candidate example")
    report.figure("11_transition", maxheight=330)
    report.p(
        "Frames are from one original EV9V file, not clips concatenated across patients or studies. The two models show a persistent A4C-to-A5C prediction change around the selected interval. The images support reviewing a changing appearance; the exact anatomical boundary remains unverified.",
        small=True,
    )
    report.p(
        "There is no verified multi-view label track for this video. The shaded region is a model-selected review window, not a ground-truth change interval. The EDA does not infer a transition frequency from this purposefully selected discovery case. MIMIC study playlists and synthetic joins are not substituted for a missing continuous acquisition.",
        small=True,
    )

    report.new_page("4.11 Uncertain interval: possible probe transition")
    report.figure("12_uncertain", maxheight=330)
    report.p(
        "The displayed interval was selected using high smoothed EchoPrime entropy. B0 and EchoPrime fluctuate between A4C and A5C, but probability fluctuations cannot identify the cause. Probe motion, cyclic anatomy, temporary poor visibility and model instability remain competing explanations.",
        small=True,
    )
    report.p(
        "This is an <b>uncertain-segment example</b>, not a confirmed probe-motion event. Original frame indices and timestamps are retained in segment_inventory.csv. Both illustrated intervals are pending independent human review; assistant inspection is not expert ground truth.",
        small=True,
    )

    report.new_page("Findings, limitations and reproducibility")
    report.table(
        ["Established by this EDA", "Practical implication"],
        [
            [
                "Three datasets differ in format, canvas, timing and label provenance.",
                "Report source-specific distributions; do not pool model predictions with expert labels.",
            ],
            [
                "EV9V has a longer tail; MIMIC mainly supplies short stored loops.",
                "Stratify temporal analyses by duration and distinguish file boundaries from natural transitions.",
            ],
            [
                "MIMIC labels and expert transition intervals are unavailable.",
                "Predicted class distributions and transition candidates are exploratory.",
            ],
            [
                "EchoNet has complete FileList measurements but six unmatched tracing cases.",
                "Preserve a join-status flag; exclude only for tracing-dependent endpoints.",
            ],
            [
                "MP4 and JPEG versions can trigger different masking decisions.",
                "Track representation, use native inputs for preprocessing comparisons, and verify lossless assumptions.",
            ],
        ],
        [390, 380],
    )
    report.p(
        "<b>Reproduce this report:</b> use the packaged frozen inventory with scripts/eda_analyze.py and scripts/eda_report.py. The package includes source-table hashes, sample selection, frame-level QC, segment candidates, figure captions, PNG/PDF figures and scripts. For a new snapshot, run scripts/eda_inventory.py in a new output directory, then the analysis script. The report narrative is specific to this snapshot and requires review before reuse with changed data."
    )
    report.p(
        "<b>Limitations:</b> headers were checked for all records; pixel decoding was sampled, not exhaustive. No pixel-content duplicate search across all clinical files was attempted. MIMIC is partially downloaded; recording keys are not patient identifiers; image proxies depend on preprocessing; no independent anatomical masks or expert temporal labels were added."
    )
    report.p(
        "<b>Source documentation:</b> EV9V: huggingface.co/datasets/bgx666/EV9V (Bo Gou et al.; CC BY 4.0). MIMIC-IV-Echo v1.0.1: physionet.org/content/mimic-iv-echo/1.0.1/ (credentialed local data). EchoNet-Dynamic: echonet.github.io/dynamic/ (Stanford research agreement). ECHOVIEW: physionet.org/content/echoview/0.1/ documents machine-generated labels; those labels were not imported here. New clinical-image figures, row-level clinical tables and the complete package remain local.",
        small=True,
    )
    report.finish()
    metrics = {
        "datasets": summary.reset_index().to_dict("records"),
        "bmode_duration": durations.reset_index().to_dict("records"),
        "qc_sample": ana["qc_sample_by_source"],
        "expert_temporal_labels": False,
        "echonet_matched_tracings": int((en.traced_frames > 0).sum()),
        "echonet_unmatched_videos": trace_missing,
        "mimic_prediction_count": len(pred),
        "report_pages_designed": report.page,
    }
    (root / "report_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Report rendered:", report.page, "designed pages", flush=True)


if __name__ == "__main__":
    main()
