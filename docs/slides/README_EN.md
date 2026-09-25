# Echocardiography preprocessing: data, methods and results

`progress_2026-09-25_EN.tex` is an 11-slide English Beamer presentation, in 16:9 format. It describes the processed data, the current methodology, the evaluation and the results. Development stages and release chronology are not used to organize the presentation.

## Contents

1. Scope and processed volume: 42 entries, 7,318 frames, seven input sources.
2. Data sources, formats, counts and coverage limits.
3. Unified input representation, metadata provenance, routing and export.
4. Temporal motion, connected intensity support and layout exclusions.
5. Geometry families, fit reliability, temporal stability and fallback behavior.
6. Real-case evaluation, synthetic ground truth and full export checks.
7. Processing outcomes for all 42 entries.
8. Stability and evidence retention for the five accepted fits.
9. A4C example with an accepted fan fit.
10. Dark PASA example retained because boundary evidence is insufficient.
11. Visual quality results and remaining limitations.

Report date: 2026-09-25. Results through 2026-09-18, from experiment code `c2289a9`. This presentation summarizes existing experiments; no preprocessing experiments were rerun. Audit entries include derived inputs and are not independent patients. Sample-mask IoU measures temporal repeatability, not anatomical segmentation accuracy.

## Evidence

The source data are the repository's `configs/audit_manifest.csv`, `docs/audit_results_stage2.csv`, `docs/audit_export_verification_stage2.csv`, `docs/audit_visual_review_stage2.csv`, `docs/stage2_verification.json`, and recorded synthetic tests in `tests/test_stage2.py`. The methodology is documented in `docs/STAGE1.md` and `docs/STAGE2.md`. The existing Chinese report, `docs/PROGRESS_REPORT_2026-09-25_ZH.md`, provides a detailed narrative of the same results.

## Compile

From the folder containing the `.tex` file:

```bash
tectonic --keep-logs progress_2026-09-25_EN.tex
```

Alternatively, run XeLaTeX twice:

```bash
xelatex progress_2026-09-25_EN.tex
xelatex progress_2026-09-25_EN.tex
```

The English source uses TeX Gyre Heros from the TeX distribution; no external Chinese font files are needed. Required packages are Beamer, fontspec, TikZ, booktabs, tabularx and colortbl. The delivered PDF was compiled with Tectonic 0.17.0. For Overleaf, upload the package and select XeLaTeX with `progress_2026-09-25_EN.tex` as the main file.

## Local example figures

| Package asset | Existing local source | Content |
|---|---|---|
| `assets/normal_fan.png` | `outputs/stage2_subset_release/cases/ev9v_d47d8e7414433eb2.png` | Accepted A4C fan fit |
| `assets/dark_sector.png` | `outputs/stage2_subset_release/cases/ev9v_a9b7d8c2e96c23f1.png` | Dark PASA input retained |

Underlying images: [EV9V, Bo Gou et al.](https://huggingface.co/datasets/bgx666/EV9V), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Project modifications include motion evidence, exclusion masks, geometry outlines, processed outputs and panel annotations. These examples do not establish clinical validity. The image license is separate from the repository's software license.

Under the repository publication policy, these assets, PDFs, PPTX files and ZIP packages remain in the ignored `outputs/slides_2026-09-25_EN/` directory. Only source text, documentation and export code are committed. Public source also compiles without assets, using placeholders on the two example slides.

## PowerPoint and delivery

The PDF retains text and vector diagrams. The PowerPoint is a full-slide-image presentation rendered at 1920 pixels wide, with English text in the slide notes. **Edit the `.tex` source to change text or diagrams; the PPTX does not contain separately editable text and diagram objects.**

After installing `pymupdf` and `python-pptx`, export a compiled PDF with:

```bash
python export_deck.py Echo_Preprocessing_Methods_Results_EN.pdf --outdir delivery --language en
```

Local deliverables:

- `Echo_Preprocessing_Methods_Results_EN.pdf`
- `Echo_Preprocessing_Methods_Results_EN.pptx`
- `Echo_Preprocessing_Methods_Results_EN_LaTeX.zip`: source, example assets, instructions, export script and presentation files.

Validation covers all 11 rendered slides, aspect ratio, TeX overflow, key audit figures, English content and notes, slide-image integrity, compilation without local assets, and ZIP integrity. No preprocessing code was changed.
