"""Per-file metadata for the preprocessing audit set."""
from __future__ import annotations

from pathlib import Path

from echoprep.cine import Cine
from echoprep.readers import guess_reader, read_cine


def file_format(path: str | Path) -> str:
    p = Path(path)
    if p.is_dir():
        return "image_dir"
    n = p.name.lower()
    if n.endswith(".nii.gz"):
        return "nii.gz"
    return p.suffix.lower().lstrip(".") or "dicom(no-ext)"


def probe_file(path: str | Path, reader: str | None = None, dataset: str = "",
               max_frames: int | None = None) -> tuple[Cine, dict]:
    p = Path(path)
    reader = reader or guess_reader(p)
    cine = read_cine(p, reader=reader, max_frames=max_frames, dataset=dataset)
    size = sum(f.stat().st_size for f in p.iterdir()) if p.is_dir() else p.stat().st_size
    info = {"format": file_format(p), "file_bytes": int(size), **cine.summary()}
    return cine, info
