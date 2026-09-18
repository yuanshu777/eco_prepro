# Preprocessing audit set

**Historical v7 protocol.** The current frozen 42-case manifest, statuses and workflow are documented
in [STAGE1.md](STAGE1.md). The old seeded builder and v7 flags below describe the initial prototype.

Purpose: look at how each source actually looks *before* tuning any algorithm. The audit set is
small (39 files), seeded, and stratified so that it covers the layouts we know about: full clinical
screens with overlays, pre-masked sectors, tall fans on black, colour Doppler, tiny 112 px videos,
large 1024 px videos, non-video formats (DICOM, NIfTI, PNG stills).

Build / run:

```bash
python scripts/build_audit_set.py --seed 0          # -> data/audit_set/audit_manifest.csv (+ symlinks)
python scripts/run_audit.py --out outputs/audit_v0  # -> results.csv, cases/*.png, overview.png, standardized/*.mp4+json
```

## What is on the server today (2026-09-17)

| Source | Files on disk | Format seen by the reader | Native size | fps | Layout (from inspection) | Role in the audit |
|---|---|---|---|---|---|---|
| EV9V | 5,138 MP4 + 910k JPEG frames | MPEG-4 part 2, yuv420p | 320x240 | 30 | **Full Philips-style screen**: text block top-left (probe, Hz, depth, gain), TIS/MI top-right, grey bar right, depth ticks right of the sector, green timeline/ECG strip bottom, "bpm" bottom-right. Sector centred, apex near the top. | Main development set; the overlays are real, so the detector *is* exercised here. |
| EchoNet-Dynamic | 10,030 AVI | MJPEG, yuvj420p | 112x112 | 50 | Already masked and cropped by Stanford; sector fills the frame. | "Already standardized" control; nothing to remove. |
| EchoNet-LVH | ~12k AVI (4 batches) | MJPEG | 1024x768 and 800x600 | 50 (metadata says 19.97 in the CSV; check) | Sector on black, masked, no text; PLAX only. | Large-resolution, pre-masked case; tests scale-dependent kernels. |
| EchoNet-Pediatric | 7,810 AVI | MJPEG | 112x112 | 50 | As Dynamic, but paediatric, A4C + PSAX; some frames have a pink tint (colour fraction > 0). | Tiny-frame + colour-tint case. |
| CAMUS | 500 patients x (2CH, 4CH) NIfTI | float32 (X, Y, T), 0.308 mm px | ~700x600 (varies) | 48.4 (from cfg) | Tall fan on black, no overlays, half cardiac cycle (ED to ES). | Non-video format + no-overlay fan; orientation check for the NIfTI reader. |
| Unity Imaging | 7,522 PNG stills | RGB PNG | e.g. 636x422 | none | **Raw GE/Philips screens**: colour Doppler box + colour bar, ECG trace, depth ticks, HR text. | Static-layout audit only (no temporal axis); NoDerivatives licence, evaluation only. |
| MIMIC-IV-Echo | not downloaded (PhysioNet credentialing pending) | DICOM multi-frame | ? | ? | Real clinical exports; text detected by OCR during de-identification, so burned-in text is present. | The main raw-clinical DICOM source once access is granted. |
| samples_dicom | 5 public DICOMs | RLE palette-colour multi-frame (rubomedical 0020, 600x430x11 frames, 15.9 fps) and 4 single-frame pydicom test images (J2K / uncompressed) | | | General ultrasound, not cardiac. | Exercises the DICOM reader paths (palette colour, RLE, JPEG 2000, FrameTime) before MIMIC arrives. |
| EchoXFlow | 20-exam sample (13 GB) | zarr beamspace streams + ECG | n/a | n/a | Pre-scan-conversion data; would need scan conversion to look like a sector. | Not used for the sector detector. |

## Per-file record

`results.csv` has one row per file with: format, file size, native height/width, frame count, fps,
duration, colour fraction (share of pixels with RGB spread > 24), reader metadata (codec, DICOM
photometric/transfer syntax/manufacturer, NIfTI zooms), detector status and statistics (area
fraction, solidity, bbox, whether the bbox touches each border, motion inside/outside, rough fan
geometry: apex, opening angle, widths), output size, runtime, and any error.

Per-file figure (`cases/<id>.png`): first / middle / last frame, temporal-std map with the hull,
mask + bbox overlay on the middle frame, and the standardized middle frame.

## Things to look for when inspecting

- Does the hull hug the fan, or does it leak into the grey bar / text / ECG strip?
- Is the far field (bottom of the fan) included even where nothing moves?
- Pre-masked datasets: bbox should touch the borders and the mask should be (almost) the whole frame.
- Colour Doppler: the colour box must stay inside the mask; the colour bar must stay outside.
- Status flags: `low_motion` on real cines is a bug or a truly static clip; `static_only` is expected
  for stills; `small_region` usually means the detector locked on to an overlay.
