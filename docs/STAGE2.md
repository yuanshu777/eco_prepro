# Stage 2: sector geometry refinement

The proposed scope is appropriate for the Stage 1 results: motion usually locates the imaging
region, but a content hull cannot establish a complete fan boundary, and UI can contaminate it.
This release adds evidence-scored geometry in `echoprep/sector/` only. Stage 1 readers, `Cine`,
metadata/FPS provenance, standardization/export, audit scripts, manifest, source configuration,
and existing tests remain byte-identical to public baseline `b494f9f`.
`stage2_frozen_contract.json` records those hashes; the subset runner and regression test enforce them.
New tests and this report accompany the sector changes. No learned model, classification experiment,
static-screen solver, or multi-panel solver was introduced.

## What changes

1. Run the original detector to preserve its status/dispatch decisions. `PASSTHROUGH`, `STATIC`,
   `MULTI_REGION`, `FAILED`, and low-motion results return before geometry. A fragmented Unity
   still remains `MULTI_REGION`; this known Stage 1 false positive is deliberately not reclassified.
2. Build an allowed area from explicit caller exclusions. One verified 320×240 EV9V layout adds
   UI-only header/text, gray-bar and outer depth-scale exclusions. It requires the known source
   hint mask **and** matching header/bar image evidence; it is never rescaled to arbitrary images.
   The existing bottom ECG exclusion is reused. Other layouts use caller zones or the full canvas.
   Profiles supply no fan outline, apex, radius or opening angle. An exclusion that intersects
   more than 1.5% of the original eroded support causes retention for review.
3. Repeat motion-guided candidate search, connected support growth and morphology inside the
   allowed area. Motion selects location; connected intensity support includes near-static tissue.
4. Fit standard fans, flat/virtual-apex fans (near plane or inner arc), rotated wedges and rectangles.
   Deterministic robust flank lines propose an apex and direction across 12 coordinate rotations;
   angular extent samples constrain the outer radius. Every proposed mask is clipped to allowed
   pixels before scoring. Rectangles require high support overlap and cannot simply approximate
   the bounding box of a fan.
5. Reject unsupported fits using residual, boundary consensus, two-flank span/consensus, outer-arc
   support, retained tissue/motion, area expansion and cut-depth checks. The original baseline's
   eroded support is checked separately. Speckle holes are not treated as sector boundaries.
   Outward completion of bounded dark gaps is possible when observed flanks and arc support it;
   invisible boundaries are not assumed to be known. Only masks change; no image content is generated.
6. Require temporal consistency before accepting geometry: fit both halves of the sampled evidence,
   and, for longer cines, resample both parity streams of the original cine. All checks must produce
   valid geometry, pairwise mask IoU must be at least 0.98, and overlap with the proposed mask at
   least 0.97. One accepted mask is then applied to the whole cine. These are engineering thresholds,
   not confidence probabilities or evidence of clinical accuracy.
7. If geometry is unreliable, reuse the original trusted hull clipped to the allowed area. A weak
   baseline is never promoted by hull fallback. `LOW_CONFIDENCE` still retains the original frames
   through the unchanged standardizer. Original quality guards run on accepted fits; captured motion
   is measured in the **final** mask, and the seed-component share is retained as a separate statistic.

`detect_sector(..., refine_geometry=False)` reproduces Stage 1. `verify_stability=False` is a
diagnostic switch; ordinary processing leaves it enabled. Accepted masks, candidate rejection
reasons, fit parameters, residuals, clipping, layout, mask hashes and stability checks are recorded
under `sector_stats.stage2_*`. The existing top-level `processing_method` and package version are
unchanged because the Stage 1 export schema is frozen; use `stage2_method` and the run's source
fingerprint to identify the actual sector path. `hull_xy` is an enclosing diagnostic hull, not an
exact encoding of clipped/nonconvex fitted masks; reproduce the mask with the recorded source and
parameters. The Stage 2 four-panel figures plot the actual proposed/fallback mask.

## Fixed subset and limits of real coverage

`echoprep/sector/audit_subset.json` freezes ten existing IDs: seven moving EV9V entries and three
retention controls. The JPEG entry derives from the normal-fan source and is **not** an independent
patient. No EchoNet/CAMUS pass-through case was used for tuning.

| Stable ID | Role |
|---|---|
| `ev9v_d47d8e7414433eb2` | Normal/wide fan, ECG |
| `ev9v_7de26f89888ddbbc` | Narrower visible support |
| `ev9v_a9b7d8c2e96c23f1` | Dark sector |
| `ev9v_cd3379c7d3cb01d0` | Wide fan |
| `ev9v_64426727a52ea9f4` | Narrower visible field |
| `ev9v_c4c5ec54d47a2742` | Fan-edge ticks |
| `ev9v_fb3e1179f73d21f2` | JPEG sequence / ECG artifacts |
| `unity_7cda63d7242f48fa` | Static dark-screen retention control |
| `unity_bdefb61b0cc5feb6` | Static tinted-screen retention control |
| `samples_dicom_eb89e1fa39011987` | Rotated color cine plus reference panel; retention control |

The narrower/wider EV9V descriptions concern visible support; they do not establish distinct probe
geometries. The selected Unity image is tinted B-mode, not verified color Doppler. The actual color
DICOM has multiple regions, so it is not a single-region geometry validation case. There is no
eligible real moving flat-top, rotated single-region, or rectangular-field echo cine in this frozen
audit. Those paths have synthetic known-mask tests, including a missing dark segment and exclusion
holes. Broader real-world validation remains open.

## Reproduce and review

```bash
.venv/bin/python -m echoprep.sector.review --out outputs/my_stage2_subset --export mp4
# Optional extra diagnostic: --stability compares two complete interleaved runs.
# Inspect the subset first; then run the unchanged full audit:
.venv/bin/python scripts/run_audit.py --out outputs/my_stage2_full
.venv/bin/python scripts/compare_audits.py outputs/audit_stage1_release/results.csv \
    outputs/my_stage2_full/results.csv --out outputs/my_stage2_full/versus_stage1.csv
.venv/bin/python -m pytest
.venv/bin/ruff check echoprep scripts tests
```

Use new output directories. The subset runner verifies both frozen infrastructure and source-input
hashes, and emits `before | motion evidence | fitted/fallback shape | after` for each ID. The human
`review.csv` remains blank; assistant review is a separate table with PASS / MINOR_ISSUE / FAIL.
These labels describe clean-output readiness. A FAIL on an intentionally retained raw screen does
not mean a runtime failure or a violation of the retention policy. Assistant inspection is sampled
engineering review, not independent human sign-off.

Local figures, videos and report ZIPs remain under ignored `outputs/`. Restricted Unity-derived
images must not be uploaded. Public artifacts contain code, documentation and tabular results only.

## Release results — 2026-09-18

The final ten-case subset is `outputs/stage2_subset_release/`. All ten four-panel figures were
inspected. Two cases accept geometry; three retain trusted hulls; two remain `LOW_CONFIDENCE`;
the three static/multi-region controls retain their Stage 1 behavior. Both accepted fits exceed
0.994 mask IoU across the internal temporal checks. The wide PMASA proposal is rejected by the
stability gate (minimum mask IoU 0.954); the original trusted hull is used. The JPEG sequence is
more sensitive to sampling phase (0.802), so its original raw frames remain retained. The dark
PASA case lacks sufficient arc/flank support and remains unresolved. No thresholds were adjusted
after starting the full audit.

The full frozen run is `outputs/audit_stage2_release/`: **42 cases, zero runtime failures, zero
status changes from Stage 1**. All source frames were processed/exported; no frame cap was used.
Summed audit processing time was approximately 161 seconds on this server.

| Status | Stage 1 | Stage 2 |
|---|---:|---:|
| SUCCESS | 9 | 9 |
| PASSTHROUGH | 20 | 20 |
| LOW_CONFIDENCE | 2 | 2 |
| STATIC | 9 | 9 |
| MULTI_REGION | 2 | 2 |
| FAILED | 0 | 0 |

Of eleven moving raw EV9V entries, **five accept standard-fan geometry**, four use trusted hull
fallback, and two retain the raw input. Accepted cases include three outside the tuning subset:
PLHLA, PMPALA and the 3,243-frame cine. Their minimum temporal-check IoUs range from 0.9927 to
0.9958. This is mask repeatability under sampled evidence, not annotated segmentation accuracy.
The remaining 31 pass-through/static/multi-region cases bypass geometry entirely.

Synthetic tests recover standard, narrow/wide, flat-top, annular near-field, rotated and rectangular
shapes, and reconstruct a supported missing dark segment. Tests also cover clipped exclusions,
unregistered layouts, conflicting exclusions, abstention, temporal inconsistency, input immutability,
and unchanged dispatch. **46 tests pass**, Ruff passes, and `git diff --check` is clean.

Release verification directly loaded the detector from committed baseline `b494f9f` and compared
it with `refine_geometry=False`: masks, hulls, boxes, flags, statuses and statistics agree exactly
on **42/42** real inputs. All 30 frozen non-sector infrastructure files match their hashes. All 33
pass-through/retained outputs preserve every input pixel in memory. All 42 MP4s decode completely
with the expected dimensions, timing and frame counts, including all 3,243 long-cine frames.
MP4 remains lossy; pixel-exact preservation refers to in-memory preprocessing, not encoding.
Subset and full-run source fingerprints match the released source.

All full-audit overview pages and first/middle/last comparisons decoded from the actual exported
videos were visually inspected. Clean-output readiness remains **10 PASS, 19 MINOR_ISSUE, 13 FAIL**;
the 13 FAIL outputs are correctly retained unresolved raw screens. The subset has five MINOR_ISSUE
and five FAIL ratings on that same scale. Both human review CSVs remain blank. See
`stage2_subset_visual_review.csv`, `audit_visual_review_stage2.csv`,
`audit_export_verification_stage2.csv`, and `stage2_verification.json` for the recorded evidence.

The visual result is a conservative first geometry release, not completion of all difficult cases.
Supported outlines become regular fan masks, with no obvious new tissue loss in sampled reviewed
frames. Small fan-edge ticks, occasional thin side-marker fragments and ECG tips touching the
imaging boundary remain. Dark-sector recovery
is demonstrated on known synthetic shapes, but **not established for the real dark PASA case**.
There are no manual ground-truth masks, no independent human ratings yet, and no complete-cine
clinical review. Before expanding to Stage 3, the useful next check is human review of these same
stable IDs, especially retained dark regions and the distinction between edge ticks and anatomy.
