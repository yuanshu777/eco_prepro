# Research notes: standardized echo preprocessing (2026-09-17)

**Stage 1 update (2026-09-18):** See [STAGE1.md](STAGE1.md) for the frozen audit, acquisition layer,
true early pass-through, review statuses, provenance and current results. The v7 account below is
historical. Geometry fitting and downstream classification experiments were not performed in Stage 1.

Scope of this pass: set up the project, inventory what data and code already exist, read the key
references, build a first (classical, Mayo-style) imaging-region detector, run it on a 39-file audit
set drawn from every dataset on the server, and record what worked, what failed and what to do next.

## 1. What exists on the server

- Machine: one NVIDIA L40S (46 GB), 8 vCPU, 31 GB RAM, 860 GB free. Shared with `qwen-agent-training`
  and `echo-view-routing`; keep GPU memory fractions modest when training.
- `~/echo-view-routing` (301 GB) already holds every dataset we can use today (see `docs/AUDIT.md` for
  the table). This project does not copy anything: `data/<name>` are symlinks into that store.
- This project: `~/echo-preprocess`, own `.venv` (uv, Python 3.12, torch 2.14 cu130, OpenCV 5,
  pydicom 3, SimpleITK, nibabel, PyAV, scikit-image), `pytest` green, git initialised. GitHub CLI is
  installed at `~/.local/bin/gh` but not yet authenticated (`scripts/github_bootstrap.sh`).

## 2. Data: what each source really looks like

Inspection of first/middle/last frames (`outputs/audit_v0/cases/*.png`, `overview.png`):

| Source | What the pipeline must remove | Verdict for the preprocessing branch |
|---|---|---|
| **EV9V** (320x240, 30 fps, MP4) | Full Philips-style screen: parameter text top-left, TIS/MI top-right, grey bar and depth ticks right, green ECG/timeline strip along the bottom, logo, calipers. Single hospital, single layout, depth 15-16 cm. | The only *video* source on disk with real overlays, so it is where the detector is actually tested. Because the layout is constant, a dataset-level template will beat per-video detection here. |
| **EchoNet-Dynamic / Pediatric** (112x112, MJPEG) | Nothing: Stanford already masked and cropped; a few residual dots. Pediatric frames carry a blue/pink tint (colour fraction 0.12-0.39), so "grayscale" cannot be assumed. | Already-standardized control. Must be passed through, not re-cropped (v0 over-cropped one file to 103x103; fixed with the `precropped` policy). |
| **EchoNet-LVH** (1024x768 / 800x600) | Nothing: sector on black, masked, no text. Some fans have a truncated (flat) top, apex above the image. | Tests resolution-scaled kernels; fine. **AVI header fps (50/63) contradicts Stanford's CSV (19.97/15.91)**: native fps must come from authoritative metadata, not the container. Now handled with `fps_table` in `configs/datasets.yaml`. |
| **CAMUS** (NIfTI, ~500-750 px, 48-57 fps from cfg) | Nothing: tall fan on black. Half cycle only (14-26 frames). | Exercises the non-video reader; detector is exact here (solidity 0.99). Orientation of the NIfTI axes verified visually (apex up). |
| **Unity Imaging** (PNG stills, 400x300 to 1024x768) | Raw GE/Philips screens: colour Doppler box, colour bar, ECG, HR, depth scale. | No temporal axis, so only the intensity-only fallback applies; it fails on dark sectors (one still locked on to a small bright blob). Useful as a *layout catalogue*, not as detector test data. NoDerivatives licence. |
| **MIMIC-IV-Echo** | Real clinical DICOM exports, burned-in text confirmed by PhysioNet's OCR-based de-identification. | Not on disk; credentialing pending. This is the source that will decide whether a classical detector is enough. |
| **Public DICOM samples** | rubomedical 0020: 1994 TEE colour-Doppler cine, palette colour, RLE, 11 frames, *two* imaging regions on screen; pydicom `US1_*`: single-frame lymph-node images (JPEG 2000 and uncompressed). | Exercise the DICOM reader paths (palette LUT, RLE, J2K, `FrameTime`). No modern echo cine is publicly downloadable without registration; MIMIC or HMC-QU (Kaggle) are the realistic options. |
| **EchoXFlow** (zarr) | Beamspace (pre-scan-conversion) data + ECG. | Not a screen image at all; out of scope for sector detection. |

Two facts that change the plan:

1. **EchoNet is not a test of the crop.** Dynamic, Pediatric and LVH are already sector-only. They can
   serve as "already standardized" external test sets, but the raw-vs-standardized comparison can only
   be made on EV9V now and on MIMIC-IV-Echo later.
2. **Frame rate metadata is unreliable in containers.** Keep `fps_container`, `fps_source` and the
   authoritative value side by side in every sidecar, as the pipeline now does.

## 3. Reference code, assessed

| Reference | What it does | Reusable? |
|---|---|---|
| Naser et al. 2024 (Mayo) | Temporal pixel change, morphology, bounding box of the largest moving region; ECG kept if it overlaps the sector; first second discarded, 10 middle frames, 256x256 padded. | Recipe only (no code released). Implemented and generalized in `echoprep/sector/motion.py`. |
| EchoNet `ConvertDICOMToAVI.ipynb` | Fixed diagonal-band geometric mask for the Stanford layout, crop dark rows, square, 10 % margin, 112x112 MJPEG, fps from (0018,0040). | Layout-specific; only the DICOM->AVI plumbing is a useful reference. |
| EchoPrime `utils.mask_outside_ultrasound` (vendored under `echo-view-routing/third_party/EchoPrime`) | "Ever non-zero" map eroded 10x, AND first-vs-last frame difference, dilated 10x, flood fill, convex hull. | Closest public implementation of the Mayo idea. Weak on 112 px inputs (fixed 3x3 x 10 erosion) and uses only two frames for motion. Ours: temporal std over up to 64 frames, resolution-scaled kernels, vertical-line opening to cut ECG bands, support-region union, quality flags. |
| EchoJEPA (arXiv:2602.02603) | Idealized sector = apex, half-angle, max radius. | The parametrization to fit (RANSAC-style) on top of our hull; next step. |
| DICOM `SequenceOfUltrasoundRegions` (0018,6011) | Vendor-provided region boxes, region type (tissue / colour flow / spectral / ECG wave), physical pixel size. | Free prior for DICOM sources; already parsed by the DICOM reader into `cine.meta["us_regions"]`. Absent in the 1994 sample; expect it in modern Philips/GE exports. |
| **Echo-Toolkit** (NN sector prediction + RANSAC fit) | As described by the mentor. | **Could not be located** by name on GitHub (API search), PyPI, arXiv API, Semantic Scholar, or web search on 2026-09-17. Need the exact link from the mentor. Closest public alternative: an nnU-Net cone segmenter is mentioned in the multi-domain segmentation literature but not released. |

## 4. Prototype on the audit set (final run: `outputs/audit_v7`, copied to `docs/audit_results_v7.csv`)

Pipeline: decode -> `Cine` -> temporal std and lit-fraction maps on <= 64 evenly spaced frames ->
"core" support (pixels lit in >= 50 % of frames) and "extended" support (lit in >= 2 %) -> choose the
connected core region that contains the most motion (mild central preference) -> grow it into the
extended support so dim far field and corners are kept -> fill holes -> edge-band cut (ECG trace /
timeline glued to the bottom or top) -> convex hull -> bbox -> crop, mask outside the hull (skipped for
inputs that already look pre-cropped), square black padding -> `.mp4` (H.264 crf 12) + JSON sidecar
-> per-case figure. Native resolution and fps are untouched.

Result: 39/39 files processed without error, mean 1.8 s per file on CPU (0.4 s at 112 px, 8 s at
1024x768). Status: 29 `ok` (every real cine) and 10 `static_only` (stills, expected). No `low_motion`
on any real cine, including 14-frame CAMUS half cycles. Figures: `docs/figures/`.

What works now:
- All nine EV9V views: text, grey bar, depth ticks, logo and the ECG strip are excluded; the hull hugs
  the fan (bbox 207-271 x 182-199 px of 320x240; the fan is ~40 % of the screen).
- All LVH fans, including flat-topped ones whose apex lies above the image (solidity 0.88-0.99).
- CAMUS: hull equals the fan (solidity 1.00); EchoNet Dynamic/Pediatric: flagged `precropped` and
  passed through unmasked, so the already-standardized sets are not damaged.
- Colour Doppler in the 1994 DICOM sample: colour box inside, colour bar outside.

What it took (seven detector iterations on the same 39 files; each was run and inspected):

| Version | Idea | Outcome |
|---|---|---|
| v0 | motion hull + lit support, morphological closing | ECG strip merged into the hull through the closing; dim fan edges lost at 112 px |
| v1 | vertical-line opening + bounded reconstruction | reconstruction grew back into the strip; one band pixel is enough to drag a convex hull |
| v2-v3 | region = lit *core* with most motion, grown into a low-lit *extent* | fixed LVH and paediatric regressions; strip still entered via the extent where it touches the fan |
| v4-v5 | drop "flashing" pixels (high std, rarely lit) | removed real anatomy: a wall moving through dark blood has the same statistics as a sweeping trace; rejected |
| v6-v7 | cut a wide thin row block at the region's edge from the row-width profile, only if mostly briefly lit and the input is not pre-cropped | ECG strip removed on EV9V, no anatomy cut, no regressions; **final** |

Known limitations of the classical detector (all visible in `docs/figures/`):
- **Dark sectors** (`ev9v_002`): the hull follows visible tissue; the invisible fan edge cannot be
  recovered by any per-video intensity or motion rule. Template or geometric fit needed (section 5).
- **Residual overlay dots** near the fan edge (depth ticks, caliper marker) survive because the hull is
  a polygon around lit pixels, not the true fan. A fitted fan mask would remove them.
- **Stills**: intensity-only; a dark sector on a raw screen fails (`unity_031`).
- **Two imaging regions on one screen** (`samples_dicom_038`): the larger, more moving one is kept.
- The band rule is a shape heuristic (bottom/top edge, >= 30 % of the width, <= 20 % of the height,
  mostly briefly-lit pixels). Vertical side panels are not covered by it, but they are normally
  separate connected components and are therefore never selected.

## 5. Recommendations

1. **Keep the classical detector as the baseline, but stop treating detection as per-video.** Group
   videos by (source, resolution, layout signature = hash of the static "lit" map); estimate one
   template mask per group from many videos; use the per-video result only to verify and to catch
   layout changes. EV9V has one layout; MIMIC will have a few dozen.
2. **Add a geometric fan model** (apex, opening angle, inner/outer radius) fitted to the hull edges,
   RANSAC-style; it fixes dark-sector cases, gives a clean mask instead of a polygon, and yields
   interpretable audit fields (angle, depth). This is what Echo-Toolkit reportedly does after its NN.
3. **Learned fallback only if needed.** Decide after MIMIC: if template + geometric fit reach > 95 %
   visually acceptable crops on a 200-file MIMIC audit, no network is required.
4. **Canonical store**: native resolution, native (authoritative) fps, RGB, square black-padded canvas,
   mask applied for raw screens and skipped for pre-cropped sources; sidecar JSON with crop box,
   padding, fps provenance, detector status/flags/stats, hull polygon. Consumers resize (224 / 256).
5. **Audit set additions**: HMC-QU (322 A4C/A2C videos, Philips + GE, 25 fps, raw screens; Kaggle login
   needed) is the cheapest way to get a second raw-screen *video* source before MIMIC arrives.
6. *(Out of scope for this branch as of 2026-09-17; belongs to `echo-view-routing`.)* The research question can be tested on EV9V: train the frame/video classifier on raw
   EV9V vs. sector-only EV9V, evaluate on CAMUS (A4C/A2C, sector-only) and EchoNet (A4C, PLAX, PSAX,
   sector-only). Because the external sets are already sector-only, any gain from standardized training
   isolates the effect of removing machine overlays from the training distribution. The `echo-view-routing`
   feature-extraction code can consume the standardized MP4s directly.

## 6. Questions for the mentor

- Exact link to Echo-Toolkit (name did not resolve anywhere public).
- MIMIC-IV-Echo credentialing status; a 200-study stratified subset is enough for the audit.
- Mask outside the sector (EchoNet/EchoPrime convention) or crop only (Mayo bounding box)? Default here
  is mask for raw screens; the sidecar allows switching.
- Is the ECG trace ever wanted downstream (Mayo kept it when it overlapped the sector)?
