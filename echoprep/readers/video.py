"""MP4/AVI/... via PyAV (ffmpeg). Falls back to OpenCV if PyAV cannot open the file."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from echoprep.cine import Cine


def _fps_from_stream(stream) -> float | None:
    for attr in ("average_rate", "guessed_rate", "base_rate"):
        r = getattr(stream, attr, None)
        if r:
            return float(Fraction(r))
    return None


def read_video(path: str | Path, max_frames: int | None = None, stride: int = 1) -> Cine:
    path = Path(path)
    try:
        import av
    except ImportError:  # pragma: no cover
        av = None
    if av is not None:
        try:
            return _read_pyav(path, max_frames, stride)
        except Exception as e:  # noqa: BLE001 - fall through to OpenCV
            err = e
    else:
        err = None
    cine = _read_cv2(path, max_frames, stride)
    if err is not None:
        cine.meta["pyav_error"] = repr(err)
    return cine


def _read_pyav(path: Path, max_frames, stride) -> Cine:
    import av

    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = _fps_from_stream(stream)
        cc = stream.codec_context
        meta = {
            "codec": cc.name,
            "pix_fmt": getattr(cc, "pix_fmt", None),
            "declared_frames": stream.frames or None,
            "container": container.format.name,
            "src_width": cc.width,
            "src_height": cc.height,
        }
        for i, fr in enumerate(container.decode(stream)):
            if stride > 1 and i % stride:
                continue
            frames.append(fr.to_ndarray(format="rgb24"))
            if max_frames and len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    if stride > 1 and fps:
        meta["native_fps"] = fps
        fps = fps / stride
    return Cine(np.stack(frames), fps, str(path), reader="video", meta=meta)


def _read_cv2(path: Path, max_frames, stride) -> Cine:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    frames, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if stride == 1 or i % stride == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    meta = {"codec": "cv2", "declared_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    return Cine(np.stack(frames), fps, str(path), reader="video", meta=meta)
