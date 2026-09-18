"""Crop to the detected imaging region, optionally mask outside it, pad to square, export + sidecar."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from echoprep.cine import Cine
from echoprep.sector.motion import SectorResult


def standardize_cine(cine: Cine, sector: SectorResult, mask_policy: str = "auto", margin_px: int = 0,
                     square: bool = True) -> tuple[Cine, dict]:
    """Return (standardized cine, processing record). Native resolution and fps are preserved; only the
    canvas changes. Model-specific resizing is deliberately left to the consumer.

    mask_policy: "always" zero everything outside the detected region; "never" crop only;
                 "auto" (default) mask unless the input already looks pre-cropped (EchoNet-style),
                 in which case pixels and canvas are preserved exactly in memory.
    Uncertain detections retain the original frames for review under every mask policy."""
    if margin_px < 0:
        raise ValueError("margin_px must be nonnegative")
    if mask_policy not in {"auto", "always", "never"}:
        raise ValueError(f"bad mask_policy {mask_policy!r}")
    preserve = sector.status in {"PASSTHROUGH", "STATIC", "MULTI_REGION", "LOW_CONFIDENCE", "FAILED"}
    apply_mask = not preserve and (mask_policy == "always" or (mask_policy == "auto" and not sector.precropped))
    H, W = cine.height, cine.width
    x0, y0, x1, y1 = (0, 0, W, H) if preserve else sector.bbox
    if x1 - x0 < 2 or y1 - y0 < 2:
        x0, y0, x1, y1 = 0, 0, W, H
    x0, y0 = max(0, x0 - margin_px), max(0, y0 - margin_px)
    x1, y1 = min(W, x1 + margin_px), min(H, y1 + margin_px)
    fr = cine.frames[:, y0:y1, x0:x1]
    if apply_mask and sector.mask.any():
        m = sector.mask[y0:y1, x0:x1]
        fr = fr * m[None, :, :, None].astype(np.uint8)
    pad = (0, 0, 0, 0)
    if square and not preserve:
        h, w = fr.shape[1], fr.shape[2]
        s = max(h, w)
        top, left = (s - h) // 2, (s - w) // 2
        pad = (top, s - h - top, left, s - w - left)
        if any(pad):
            fr = np.pad(fr, ((0, 0), (pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), constant_values=0)
    record = {
        "schema_version": 2,
        "status": sector.status,
        "warnings": list(sector.flags),
        "processing_method": "passthrough" if sector.status == "PASSTHROUGH" else "motion_v7_with_stage1_guards",
        "action": "passthrough" if sector.status == "PASSTHROUGH" else ("retained_for_review" if preserve else "crop_mask_pad"),
        "fps_value": cine.fps,
        "fps_source": cine.meta.get("fps_source", "unknown"),
        "fps_container": cine.meta.get("fps_container"),
        "fps_dataset_table": cine.meta.get("fps_dataset_table"),
        "fps_reader": cine.meta.get("fps_reader", cine.fps),
        "source_metadata": dict(cine.meta),
        "source_path": cine.path,
        "dataset": cine.dataset,
        "reader": cine.reader,
        "native": {"height": H, "width": W, "n_frames": cine.n_frames, "fps": cine.fps},
        "crop_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "pad_tblr": [int(p) for p in pad],
        "mask_policy": mask_policy,
        "mask_applied": bool(apply_mask and sector.mask.any()),
        "sector_flags": list(sector.flags),
        "output": {"height": int(fr.shape[1]), "width": int(fr.shape[2]), "n_frames": int(fr.shape[0]), "fps": cine.fps},
        "sector_status": sector.status,
        "sector_stats": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in sector.stats.items()},
        "hull_xy": sector.hull.tolist(),
        "color_fraction": cine.color_fraction(),
        "echoprep_version": "0.2.0",
    }
    out = Cine(np.ascontiguousarray(fr), cine.fps, cine.path, cine.dataset, cine.reader, dict(cine.meta))
    return out, record


def export_cine(cine: Cine, out_path: str | Path, record: dict | None = None, crf: int = 12,
                default_fps: float = 30.0) -> Path:
    """Write .mp4 (H.264, near-lossless crf), .avi (MJPEG, EchoNet convention) or .npz (lossless).
    A JSON sidecar with the processing record is written next to the file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = cine.fps or default_fps
    fmt = out_path.suffix.lower()
    export_pad = [0, 0, 0, 0]
    encoded_h, encoded_w = cine.height, cine.width
    if fmt == ".npz":
        np.savez_compressed(out_path, frames=cine.frames, fps=np.float64(cine.fps if cine.fps is not None else np.nan))
    elif fmt in {".mp4", ".avi", ".mkv"}:
        import av

        fr = cine.frames
        h, w = fr.shape[1], fr.shape[2]
        if (h % 2 or w % 2):  # yuv420p needs even dimensions
            export_pad = [0, h % 2, 0, w % 2]
            fr = np.pad(fr, ((0, 0), (0, h % 2), (0, w % 2), (0, 0)))
            h, w = fr.shape[1], fr.shape[2]
        encoded_h, encoded_w = h, w
        rate = Fraction(fps).limit_denominator(1000)
        with av.open(str(out_path), "w") as container:
            if fmt == ".avi":
                stream = container.add_stream("mjpeg", rate=rate)
                stream.pix_fmt = "yuvj420p"
            else:
                stream = container.add_stream("libx264", rate=rate)
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": str(crf), "preset": "medium"}
            stream.width, stream.height = w, h
            for f in fr:
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    else:
        raise ValueError(f"unsupported output format {fmt}")
    if record is not None:
        rec = dict(record)
        rec["export"] = {"path": str(out_path), "format": fmt, "fps_written": float(fps),
                         "fps_assumed": cine.fps is None and fmt != ".npz",
                         "pad_tblr": export_pad, "height": encoded_h, "width": encoded_w,
                         "n_frames": cine.n_frames, "lossy": fmt != ".npz"}
        if fmt == ".npz" and cine.fps is None:
            rec["export"]["fps_written"] = None
        out_path.with_suffix(".json").write_text(json.dumps(rec, indent=2))
    return out_path
