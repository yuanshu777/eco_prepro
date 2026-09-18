"""A directory of frames (e.g. EV9V Images/{id}/*.jpg) or a single still image."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from echoprep.cine import Cine
from echoprep.readers import IMAGE_EXTS


def read_images(path: str | Path, max_frames: int | None = None, fps: float | None = None) -> Cine:
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise ValueError(f"no image frames under {path}")
    else:
        files = [path]
    if max_frames:
        files = files[:max_frames]
    frames = []
    for f in files:
        with Image.open(f) as im:
            frames.append(np.asarray(im.convert("RGB")))
    shapes = {fr.shape for fr in frames}
    if len(shapes) != 1:
        raise ValueError(f"inconsistent frame sizes in {path}: {shapes}")
    meta = {"n_files": len(files), "first_file": files[0].name}
    return Cine(np.stack(frames), fps, str(path), reader="images", meta=meta)
