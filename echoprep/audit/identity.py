"""Content identities for frozen cases, including compound MHD inputs and frame folders."""

from __future__ import annotations

import hashlib
from pathlib import Path

from echoprep.readers import IMAGE_EXTS


def content_sha256(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()

    def feed(f: Path):
        with f.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)

    if p.is_dir():
        # Frame order/names are part of a sequence's identity; absolute location is not.
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise ValueError(f"empty frame folder: {p}")
        for f in files:
            h.update(f.name.encode() + b"\0" + str(f.stat().st_size).encode() + b"\0")
            feed(f)
    else:
        feed(p)
        if p.suffix.lower() == ".mhd":
            lines = p.read_text().splitlines()
            raw = next(l.split("=", 1)[1].strip() for l in lines if l.startswith("ElementDataFile"))
            if raw == "LOCAL" or raw.startswith("LIST") or "%" in raw:
                raise ValueError("frozen audit supports single external MHD payloads only")
            h.update(b"\0MHD_PAYLOAD\0")
            feed(p.parent / raw)
    return h.hexdigest()


def stable_case_id(dataset: str, digest: str) -> str:
    return f"{dataset}_{digest[:16]}"


def resolve_case_path(row: dict, datasets: dict, project_root: Path) -> Path:
    path = Path(row["path"])
    if path.is_absolute():  # old audit manifests remain readable for version comparison
        return path
    base = (
        project_root
        if row.get("path_base") == "project"
        else project_root / datasets[row["dataset"]]["root"]
    )
    return base / path
