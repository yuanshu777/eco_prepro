# echo-preprocess

Standardized preprocessing for echocardiography cines from heterogeneous sources (EV9V, EchoNet,
CAMUS, MIMIC-IV-Echo, future clinical DICOM). Every source is decoded into one canonical
representation, the ultrasound imaging region (sector) is detected, and a clean cropped cine is
exported together with a JSON record of what was done. Model-specific resizing and frame-rate
changes are deliberately **not** part of this pipeline.

```text
raw file (MP4 / AVI / DICOM / NIfTI / MHD / frame folder)
   -> reader            echoprep.readers          -> Cine (T,H,W,3 uint8 RGB, native fps, metadata)
   -> sector detection  echoprep.sector.motion    -> mask + bbox + hull + status + stats
   -> standardize       echoprep.standardize      -> crop, mask outside the region, pad to square
   -> export            echoprep.standardize      -> .mp4 (near-lossless) / .avi / .npz + .json sidecar
```

## Scope (decided 2026-09-17)

This branch builds a robust, lightweight, source-aware echo standardization pipeline. Its success
criterion: given an echo file from any source, it reads the format, finds the real imaging region,
removes surrounding screen information, keeps the correct metadata, and writes one standardized
video, or flags the case clearly instead of silently producing a wrong result.

Out of scope here: training view classifiers, raw-vs-cropped accuracy comparisons, learned
segmentation models, large-scale sector-mask annotation, and any downstream benchmark. Validation is a
small heterogeneous audit set with automated status checks and manual visual QC.

Companion project: [`echo-view-routing`](https://github.com/WilliamQiuzy/echo-view-routing) (view
classification / temporal routing). This repository owns the *data standardization* branch of the
project; `data/` here only holds symlinks into that project's shared raw store on the server.

## Repository

The project repository is [yuanshu777/eco_prepro](https://github.com/yuanshu777/eco_prepro).
Completed, checked code and documentation are committed and pushed to `main`. Large outputs,
report ZIPs and restricted dataset-derived figures remain local. See [AGENTS.md](AGENTS.md)
for the standing workflow. `scripts/github_bootstrap.sh` handles first-time GitHub authentication
and pushes the existing reviewed branch to this repository.

## Stage 1

The current audit is frozen at **42 cases** with content-based IDs. Start with
[`docs/STAGE1.md`](docs/STAGE1.md) for the workflow, case matrix, statuses, FPS provenance,
acquisition restrictions and validation results. The preprocessing branch is version 0.2.0.

```bash
.venv/bin/python scripts/fetch_samples.py --all
.venv/bin/python scripts/build_audit_set.py
.venv/bin/python scripts/run_audit.py --out outputs/my_new_audit
```

Review `overview.png`, `review_pages/`, `cases/`, and `audit_review.csv` in the run directory.
Uncertain detections retain the input for review. Already-standardized cines have an early
PASSTHROUGH decision. MP4 exports are lossy; use NPZ for lossless pixel storage.

## Layout

| Path | Purpose |
|---|---|
| `echoprep/cine.py` | `Cine` dataclass: the canonical in-memory representation |
| `echoprep/readers/` | one small reader per input family: `video` (PyAV/OpenCV), `dicom` (pydicom 3), `nifti`/`sitk` (CAMUS, MHD), `images` (frame folders, stills) |
| `echoprep/sector/motion.py` | Mayo-style motion-based imaging-region detector (temporal std -> morphology -> largest central region -> convex hull), plus quality flags and rough fan geometry |
| `echoprep/standardize.py` | crop / mask / pad, export with JSON sidecar |
| `echoprep/audit/` | per-file metadata probe and inspection figures |
| `scripts/build_audit_set.py` | verify/materialize the frozen audit manifest (no resampling) |
| `scripts/run_audit.py` | run the current pipeline on the audit set, write `outputs/<run>/results.csv`, per-case figures and an overview grid |
| `scripts/probe_file.py` | inspect one file of any format, optionally with detection and a figure |
| `configs/datasets.yaml` | dataset roots, readers, notes |
| `docs/` | research notes, audit findings, references |
| `tests/` | synthetic fan + overlays; reader round-trips (mp4/avi/npz/DICOM/NIfTI) |

## Quickstart (server)

```bash
cd ~/echo-preprocess
source .venv/bin/activate                     # created with uv, Python 3.12, torch cu130
pytest -q                                     # synthetic tests, CPU, ~10 s
python scripts/build_audit_set.py             # verify 42 fixed cases
python scripts/run_audit.py --out outputs/my_new_audit
python scripts/probe_file.py data/ev9v/Videos/<id>.mp4 --detect --fig /tmp/case.png
```

Python API:

```python
from echoprep import read_cine, detect_sector, standardize_cine, export_cine
cine = read_cine("study/0001.dcm")                 # any supported format
sector = detect_sector(cine.frames)                # .mask .bbox .hull .status .stats
clean, record = standardize_cine(cine, sector)     # native resolution, native fps, square canvas
export_cine(clean, "out/0001.mp4", record)         # + out/0001.json
```

## Design rules

- **One canonical representation.** Readers hide the file format; nothing downstream knows whether
  the source was DICOM or AVI. Grayscale is stored as three identical channels so colour Doppler goes
  through the same path without losing colour.
- **Motion locates the sector, it does not define it.** The detector uses temporal variation only to
  find the imaging region; the exported region is the full convex sector, including low-motion parts.
- **Do not standardize too aggressively.** Native resolution and fps are kept; 224x224 or 256x256
  resizing is the consumer's job. Every output has a sidecar with crop coordinates, padding, native
  size/fps, detector status and statistics, so processing can be audited and reproduced from the retained source.
- **Audit before algorithms.** `docs/AUDIT.md` records how each source actually looks before any
  method is tuned.

## Status

See `docs/RESEARCH_NOTES.md` for the research pass (references, code found, dataset audit, prototype
results and iteration history) and the open questions; `docs/AUDIT.md` for the audit protocol and the
per-source layout table. Current local figures are under `outputs/audit_stage1_release/`; see
`docs/STAGE1.md` for current validation. Historical reports below predate Stage 1.
