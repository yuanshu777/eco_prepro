# Stage 1: frozen audit and preprocessing infrastructure

Scope: preprocessing only. No learned model, new geometry fitting, or downstream classification
experiment. The v7 motion detector remains the baseline; this stage adds preservation policies,
quality reporting, reproducible identities, acquisition reporting, and review artifacts.

## Reproduce on this server

```bash
cd /home/william/echo-preprocess
.venv/bin/python scripts/fetch_samples.py --all
.venv/bin/python scripts/build_audit_set.py
.venv/bin/python scripts/run_audit.py --out outputs/my_new_audit
.venv/bin/python scripts/compare_audits.py outputs/audit_v7/results.csv \
    outputs/my_new_audit/results.csv --out outputs/my_new_audit/versus_v7.csv
.venv/bin/python -m pytest
.venv/bin/ruff check echoprep scripts tests
```

Use a fresh output directory for each run. Existing results and review decisions are never silently
overwritten. The audit uses all frames unless `--max-frames` is explicitly supplied; truncation is
recorded in `run.json`. Sampling for region detection is separate: at most 64 evenly spaced frames,
including endpoints. Sampling now occurs before grayscale conversion to limit long-cine memory use.

## Frozen identities and case coverage

`configs/audit_manifest.csv` freezes 42 cases: the original 39 plus one derived MHD fixture, one
3,243-frame EV9V cine, and one existing EV9V JPEG sequence. `case_type` documents each case's purpose.
`configs/audit_id_map.csv` maps all 39 legacy IDs to their new stable IDs.

IDs are `<dataset>_<first 16 hex digits of SHA-256>`; the full digest is also stored and verified.
They depend on content, not row order or absolute location. Directory hashes include frame names,
order and contents. MHD hashes include both the header and its external pixel payload. Changed
content fails validation rather than silently reusing an old ID. Store additions in a new manifest
version while preserving existing identities. `freeze_audit.py` is the one-time migration tool and
refuses to overwrite an existing frozen manifest; `build_audit_set.py` only materializes it.
Dataset paths resolve through `configs/datasets.yaml`; no personal absolute root is required in the
frozen manifest. The MHD fixture can be regenerated from its frozen NIfTI parent during materialization.
It is reader coverage, not an independent patient or a new acquisition distribution.

| Preprocessing situation | Coverage / limitation |
|---|---|
| Nine grayscale views, raw Philips-style screen, ECG, depth ticks | Nine original EV9V cines |
| Already masked/cropped inputs | Dynamic 3, Pediatric 4, LVH 6; early source-policy pass-through |
| Large images and container/dataset FPS disagreement | Six LVH cines; both FPS values retained |
| Short cine, NIfTI, varying image quality | Six CAMUS sequences |
| MHD/RAW reader | One derived CAMUS fixture, same pixels and timing as its parent |
| Very long cine | One existing 3,243-frame EV9V clip; full export, no frame truncation |
| Image directory | Existing JPEG sequence matching an original EV9V case |
| Static raw layouts, dark sector, sepia-tinted stills | Six Unity PNGs; restricted local QC |
| DICOM codec paths / noncardiac stills | Four public pydicom test files, including noncardiac rectangular-field power-Doppler stills |
| Historical color cine, static reference plus moving main panel | One 11-frame 1994 DICOM; not modern vendor coverage |
| Low-motion real cine | Unresolved: no candidate in the existing audit or a bounded screen of 64 further EV9V clips (16 sampled frames, temporal-std p99 < 3 criterion). This screening does not prove absence in the full dataset. Synthetic regression test covers the fallback. |
| Modern real color Doppler cine | Unresolved |
| Raw GE screen video | Unresolved; GE still layouts do not satisfy this row |
| Rectangular-field echo cine | Unresolved; noncardiac rectangular-field stills exercise decoding only |
| Broader raw-vendor diversity | Unresolved; no large datasets downloaded to fill gaps |

## Processing policy

Statuses are `SUCCESS`, `PASSTHROUGH`, `LOW_CONFIDENCE`, `STATIC`, `MULTI_REGION`, and `FAILED`.

- `PASSTHROUGH` is decided before region growth, edge-band removal, hull construction or cropping.
  EchoNet uses explicit audited source policy. Unknown inputs use stricter image evidence: dominant
  near-full imaging support, negligible disconnected lit content, and motion. Pass-through leaves
  the entire original pixel array and canvas unchanged in memory, including rectangular inputs.
- `SUCCESS` applies the existing v7 crop/mask/pad operation.
- `LOW_CONFIDENCE`, `STATIC`, `MULTI_REGION`, and detected `FAILED` retain original frames and canvas
  with `action=retained_for_review`. The proposed mask remains visible in QC figures. A decoder error
  cannot export frames: it produces a FAILED sidecar/result, and other cases continue. Audit exit
  status is nonzero if any case is FAILED.
- Two substantial disconnected support regions trigger `MULTI_REGION`, including a static reference
  panel beside a moving image. Thin UI bars do not qualify solely by being connected regions.
  This remains a heuristic: disconnected anatomy can trigger review, and other layouts can be missed.

Guards record area fraction, solidity, captured-motion share, overlap with known exclusion zones,
and color fractions inside/outside. Thresholds are review triggers, not calibrated confidence.
For successful candidates: area < 0.15, solidity < 0.88, selected core motion share < 0.75, or known
zone contact > 0.005 changes the status to LOW_CONFIDENCE. Color outside > 0.01 is informational:
it may correctly represent a removed ECG or color bar. EV9V exclusion zones are conservative QC
regions only; no template masking or fan fitting has been added. Low-motion candidates are flagged
by the existing motion tests. Status checks are not proof that the full sector was recovered.

## Timing, export and provenance

Every sidecar carries `fps_value`, `fps_source`, `fps_container`, `fps_dataset_table`, `fps_reader`,
native size/frame count, original reader metadata, proposed sector statistics, applied crop/padding,
action, warnings, content identity, source access policy, and run fingerprints. Dataset-table FPS
has precedence while the container value remains intact; the FPS-table file digest is recorded.
CAMUS FPS is attributed to its sibling configuration, not to a nonexistent video container.

Unknown native FPS remains null. A video encoder may use a 30 FPS display default; the sidecar marks
`fps_assumed=true`. This is relevant to stills and frame folders without explicit timing. Lossless
NPZ keeps unknown FPS as NaN. No biological timing is inferred from a filename.

MP4 remains lossy H.264 (CRF 12). Pixel-exact pass-through describes the in-memory preprocessing,
not a lossless MP4 round trip. Codec-required right/bottom padding to even dimensions is recorded
separately under `export.pad_tblr`, with the actual encoded dimensions and frame count.
The processing crop/padding fields describe the pre-encoding frames.

## Acquisition and restrictions

`fetch_samples.py` uses local files first, with a default limit of three samples (maximum ten).
Otherwise it reports `DOWNLOADED`, `AUTH_REQUIRED`, `ARCHIVE_ONLY`, `UNAVAILABLE`, or `SKIPPED`.
Only configured public individual files are downloaded, with a byte cap, checksum when configured,
DICOM-header validation, and atomic replacement. One source's failure does not abort others.
`--local-only` disables network acquisition. This Stage 1 tool deliberately has no full-archive mode.
The sample inventory is separate from the fixed audit: downloading a file never changes the audit.

Restricted services have no authenticated per-file adapters in Stage 1. The tool checks configured
credential environment-variable presence without printing values; credentials do not establish
access approval. Missing credentials yield AUTH_REQUIRED; present credentials without an implemented
adapter yield SKIPPED. Authorized local samples are always reusable. No access agreements are
accepted automatically.

The registry records source, access method, license, and redistribution restrictions for every
configured source. Unverified terms are explicitly marked unverified. In particular, Unity-derived
videos and figures are never marked redistributable. All run products stay under ignored `outputs/`.
The old mixed-data overview and non-approved LVH/DICOM examples were moved from `docs/figures/` to
local `outputs/legacy_doc_figures/`. Historical images remain only on the local `legacy/local-before-github` branch. The published
`main` branch starts from a cleaned snapshot and excludes those historical commits. Never push or
merge the legacy branch into the public repository.
The remaining tracked EV9V example figures have the dataset's attribution requirement.

Primary source records checked for this stage:
- [EV9V dataset card](https://huggingface.co/datasets/bgx666/EV9V): CC-BY-4.0.
- [Stanford EchoNet-Dynamic agreement](https://echonet.github.io/dynamic/): research access and redistribution restrictions.
- [CAMUS dataset page](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html): dataset details and citation requirement; redistribution approval not inferred.
- [pydicom-data license](https://github.com/pydicom/pydicom-data/blob/abc42b90985fb6cf385aa4af766d2c9c94a257a4/LICENSE): MIT; individual public download pinned to that revision and a content checksum.
- [MIMIC-IV-Echo](https://physionet.org/content/mimic-iv-echo/1.0.1/): credentialed access and DUA.
- Unity's NoDerivatives restriction is carried forward as explicit project policy from the handoff;
  the exact underlying dataset license remains unverified, so redistribution stays disabled.

## Review artifacts

Each run produces MP4/JSON pairs, `results.csv`, an overall before/after grid, paginated grids,
and per-case first/middle/last before/after panels plus diagnostic maps. `audit_review.csv` starts
blank, accepts PASS / MINOR_ISSUE / FAIL, and has reason/reviewer fields. Blank means unreviewed.
Automated SUCCESS is not a visual PASS. Assistant inspection is recorded separately from human
review so the two are not confused.

`compare_audits.py` matches by stable identity, includes added/removed cases, normalizes legacy
statuses, and reports status, dimensions, frame-count, FPS and sector-statistic changes. Status
changes flag cases to inspect; they do not automatically imply a regression.

## Verified results (2026-09-18)

Final run: `outputs/audit_stage1_release/`, all 42 cases, no frame cap, approximately 117 seconds
summed processing time on this server. All 42 MP4s decode fully with the expected frame counts,
encoded dimensions, and recorded playback FPS. The 3,243-frame cine exports all 3,243 frames.
Source, configuration and manifest fingerprints match the final working files.

| Processing status | Count |
|---|---:|
| SUCCESS | 9 |
| PASSTHROUGH | 20 |
| LOW_CONFIDENCE | 2 |
| STATIC | 9 |
| MULTI_REGION | 2 |
| FAILED | 0 |

Validation: **27 tests pass**, Ruff passes, and `git diff --check` is clean. Tests cover preservation
before any region growth, moving and static/moving dual panels, blank/static/low-motion inputs,
overlay warnings, FPS overrides through encoding, unknown FPS, codec padding, MHD pixels/timing,
content identity, acquisition statuses and byte limits, restricted-data policy, and stable-ID
comparison. An actual public DICOM download passes its pinned checksum and header check. MHD
regeneration in a temporary directory reproduces the frozen digest, pixel orientation and FPS.

Visual inspection covers every case in the before/after grids plus first/middle/last frames decoded
from the exported MP4s. It is a sampled engineering inspection, not full video playback or clinical
validation. `assistant_visual_review.csv` records the assistant's findings. `audit_review.csv` remains
blank for independent human review. The clean-output readiness ratings are **10 PASS, 19 MINOR_ISSUE,
13 FAIL**. The 13 FAIL ratings describe safely retained, unresolved raw outputs; they are not runtime
failures or silent anatomy deletion. Small source artifacts also remain in some pass-through inputs.

Specific unresolved findings:

- Dark EV9V PASA (`ev9v_a9b7d8c2e96c23f1`, formerly `ev9v_002`): low solidity triggers review;
  the full original is retained instead of cropping to the visible tissue blob.
- JPEG frame folder (`ev9v_fb3e1179f73d21f2`): overlay contact and low solidity trigger review;
  no crop is applied. Timing is unknown in this reader, with assumed display FPS explicitly marked.
- Historical DICOM (`samples_dicom_eb89e1fa39011987`): MULTI_REGION correctly catches a static
  reference next to the moving color image. Both panels remain; cleanup is deferred.
- Dark Unity still (`unity_7cda63d7242f48fa`, formerly `unity_031`): disconnected anatomical
  support produces a **false-positive MULTI_REGION** warning. The original is retained safely.
  This exposes the limit of component-based panel counting; it is documented, not tuned away.
- Other static raw screens retain external UI. Static-region cleanup remains unresolved.
- Successful EV9V crops can retain small ticks/markers near the hull boundary. A future detector
  refinement may address them; Stage 1 adds no new geometric fitting.
- Several already-processed EchoNet samples still contain small inherited ECG fragments or
  colored/edge markers. PASSTHROUGH preserves them intentionally; it does not certify UI-free data.
- The selected Unity color stills are sepia-tinted B-mode, **not confirmed color Doppler**. Manifest
  annotations were corrected after inspection. Noncardiac power-Doppler DICOM stills and the old
  color cine do not fill the modern echo color-cine coverage gap.

Comparison with v7: **39 matched cases, 3 additions, no removed cases, 22 normalized status changes**.
The existing successful raw-cine crops are retained; the principal changes are real pass-through and
preservation of uncertain cases. No new obvious anatomy-loss regression was found in the inspected
frames. This is not evidence of generalization to unseen vendors, nor a claim that every mask is
correct. The exported comparison is `outputs/audit_stage1_release/versus_v7.csv`.

Local artifacts include `export_verification.csv`, `temporal_review/`, acquisition/download reports,
and the bounded low-motion screen. Small tabular results are mirrored in `docs/`; restricted pixels
are not. Human review and the explicitly deferred detector/data gaps remain available for Stage 2.

