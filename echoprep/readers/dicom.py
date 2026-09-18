"""DICOM ultrasound (single- or multi-frame) via pydicom 3.

Handles MONOCHROME1/2, RGB, YBR_* (converted to RGB by pydicom), PALETTE COLOR (LUT applied),
8- and 16-bit samples, and extracts frame rate + `SequenceOfUltrasoundRegions` hints when present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import apply_color_lut, pixel_array

from echoprep.cine import Cine


def _to_uint8(a: np.ndarray, bits_stored: int | None) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    hi = float(2 ** bits_stored - 1) if bits_stored and bits_stored > 8 else float(max(a.max(), 1.0))
    return np.clip(a / hi * 255.0 + 0.5, 0, 255).astype(np.uint8)


def frame_rate(ds: pydicom.Dataset) -> float | None:
    ft = ds.get("FrameTime")  # ms between frames
    if ft:
        try:
            ft = float(ft)
            if ft > 0:
                return 1000.0 / ft
        except (TypeError, ValueError):
            pass
    for tag in ("CineRate", "RecommendedDisplayFrameRate"):
        v = ds.get(tag)
        if v:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    ftv = ds.get("FrameTimeVector")
    if ftv:
        try:
            v = np.asarray([float(x) for x in ftv][1:])
            if v.size and v.mean() > 0:
                return 1000.0 / float(v.mean())
        except (TypeError, ValueError):
            pass
    return None


def ultrasound_regions(ds: pydicom.Dataset) -> list[dict]:
    out = []
    for r in ds.get("SequenceOfUltrasoundRegions", []) or []:
        out.append({
            "spatial_format": r.get("RegionSpatialFormat"),  # 1 = 2D (tissue or flow), 3 = spectral, 4 = wave (ECG)
            "data_type": r.get("RegionDataType"),            # 1 = tissue, 2 = colour flow, 3 = PW spectral, ...
            "x0": r.get("RegionLocationMinX0"), "y0": r.get("RegionLocationMinY0"),
            "x1": r.get("RegionLocationMaxX1"), "y1": r.get("RegionLocationMaxY1"),
            "phys_dx": r.get("PhysicalDeltaX"), "phys_dy": r.get("PhysicalDeltaY"),
            "phys_units_x": r.get("PhysicalUnitsXDirection"), "phys_units_y": r.get("PhysicalUnitsYDirection"),
        })
    return out


def read_dicom(path: str | Path, max_frames: int | None = None) -> Cine:
    path = Path(path)
    ds = pydicom.dcmread(str(path))
    pi = str(ds.get("PhotometricInterpretation", "MONOCHROME2"))
    arr = pixel_array(ds, as_rgb=True)  # YBR* -> RGB handled by pydicom
    bits = ds.get("BitsStored")
    if pi == "PALETTE COLOR":
        arr = apply_color_lut(arr, ds)  # -> (..., 3) in the LUT's bit depth
        arr = _to_uint8(arr, ds.get("RedPaletteColorLookupTableDescriptor", [None, None, 8])[2])
    else:
        arr = _to_uint8(arr, bits)
    n_frames = int(ds.get("NumberOfFrames", 1) or 1)
    samples = int(ds.get("SamplesPerPixel", 1))
    if samples == 1 and pi != "PALETTE COLOR":
        if pi == "MONOCHROME1":
            arr = 255 - arr
        frames = arr if arr.ndim == 3 else arr[None]          # (T,H,W)
    else:
        frames = arr if arr.ndim == 4 else arr[None]          # (T,H,W,3)
    if n_frames > 1 and frames.shape[0] != n_frames:
        # pixel_array already reshaped by pydicom; keep what we got but record the mismatch
        pass
    if max_frames:
        frames = frames[:max_frames]
    meta = {
        "manufacturer": str(ds.get("Manufacturer", "")),
        "model": str(ds.get("ManufacturerModelName", "")),
        "sop_class": getattr(ds.get("SOPClassUID"), "name", str(ds.get("SOPClassUID", ""))),
        "transfer_syntax": ds.file_meta.TransferSyntaxUID.name if hasattr(ds, "file_meta") and "TransferSyntaxUID" in ds.file_meta else None,
        "photometric": pi,
        "bits_stored": bits,
        "samples_per_pixel": samples,
        "declared_frames": n_frames,
        "ultrasound_color_data_present": ds.get("UltrasoundColorDataPresent"),
        "image_type": "\\".join(ds.get("ImageType", [])) if ds.get("ImageType") else None,
        "burned_in_annotation": ds.get("BurnedInAnnotation"),
        "n_us_regions": len(ds.get("SequenceOfUltrasoundRegions", []) or []),
    }
    cine = Cine(frames, frame_rate(ds), str(path), reader="dicom", meta=meta)
    cine.meta["us_regions"] = ultrasound_regions(ds)  # non-scalar: kept out of summary() automatically
    return cine
