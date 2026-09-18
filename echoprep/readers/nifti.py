"""CAMUS-style NIfTI sequences (nibabel) and MHD/MHA/NRRD sequences (SimpleITK)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from echoprep.cine import Cine


def _to_uint8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= 255.0 and lo >= 0.0:
        return np.clip(a + 0.5, 0, 255).astype(np.uint8)
    return np.clip((a - lo) / max(hi - lo, 1e-6) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def camus_cfg(path: Path) -> dict:
    """Read Info_2CH.cfg / Info_4CH.cfg next to a CAMUS file, if present."""
    m = re.search(r"_(2CH|4CH)_", path.name)
    if not m:
        return {}
    cfg = path.parent / f"Info_{m.group(1)}.cfg"
    if not cfg.exists():
        return {}
    out = {}
    for line in cfg.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def read_nifti(path: str | Path, max_frames: int | None = None, fps: float | None = None) -> Cine:
    import nibabel as nib

    path = Path(path)
    img = nib.load(str(path))
    a = np.asarray(img.dataobj)
    zooms = tuple(float(z) for z in img.header.get_zooms())
    # CAMUS stores (X, Y, T) with X along image columns; transpose each frame to (rows, cols).
    if a.ndim == 2:
        frames = a.T[None]
    elif a.ndim == 3:
        frames = np.transpose(a, (2, 1, 0))  # (T, rows, cols)
    else:
        raise ValueError(f"unexpected NIfTI shape {a.shape}")
    frames = _to_uint8(frames)
    if max_frames:
        frames = frames[:max_frames]
    cfg = camus_cfg(path)
    if fps is None and "FrameRate" in cfg:
        try:
            fps = float(cfg["FrameRate"])
        except ValueError:
            fps = None
    meta = {"zooms": str(zooms), "src_dtype": str(a.dtype), "src_shape": str(a.shape),
            "pixel_mm_x": zooms[0] if len(zooms) > 0 else None,
            "pixel_mm_y": zooms[1] if len(zooms) > 1 else None,
            **{f"cfg_{k}": v for k, v in cfg.items()}}
    return Cine(frames, fps, str(path), reader="nifti", meta=meta)


def read_sitk(path: str | Path, max_frames: int | None = None, fps: float | None = None) -> Cine:
    import SimpleITK as sitk

    path = Path(path)
    img = sitk.ReadImage(str(path))
    a = sitk.GetArrayFromImage(img)  # (T, rows, cols) for a 3-D sequence
    if a.ndim == 2:
        a = a[None]
    frames = _to_uint8(a)
    if max_frames:
        frames = frames[:max_frames]
    cfg = camus_cfg(path)
    if fps is None and "FrameRate" in cfg:
        fps = float(cfg["FrameRate"])
    meta = {**{f"cfg_{k}": v for k, v in cfg.items()}, "spacing": str(img.GetSpacing()), "src_dtype": str(a.dtype), "src_shape": str(a.shape)}
    return Cine(frames, fps, str(path), reader="sitk", meta=meta)
