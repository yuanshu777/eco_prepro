"""Dataset-specific readers. Each one is tiny: decode -> Cine. Everything after that is shared."""
from __future__ import annotations

from pathlib import Path

from echoprep.cine import Cine

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def guess_reader(path: str | Path) -> str:
    p = Path(path)
    if p.is_dir():
        return "images"
    suf = p.suffix.lower()
    name = p.name.lower()
    if suf in VIDEO_EXTS:
        return "video"
    if suf == ".dcm" or suf == "" or name.endswith(".dicom"):
        return "dicom"
    if name.endswith((".nii", ".nii.gz")):
        return "nifti"
    if suf in {".mhd", ".mha", ".nrrd"}:
        return "sitk"
    if suf in IMAGE_EXTS:
        return "images"
    return "dicom"  # extension-less clinical exports are usually DICOM


def read_cine(path: str | Path, reader: str | None = None, max_frames: int | None = None,
              dataset: str = "", **kw) -> Cine:
    """Decode any supported input into a Cine. `reader` in {video, dicom, nifti, sitk, images}."""
    reader = reader or guess_reader(path)
    if reader == "video":
        from echoprep.readers.video import read_video
        cine = read_video(path, max_frames=max_frames, **kw)
    elif reader == "dicom":
        from echoprep.readers.dicom import read_dicom
        cine = read_dicom(path, max_frames=max_frames, **kw)
    elif reader == "nifti":
        from echoprep.readers.nifti import read_nifti
        cine = read_nifti(path, max_frames=max_frames, **kw)
    elif reader == "sitk":
        from echoprep.readers.nifti import read_sitk
        cine = read_sitk(path, max_frames=max_frames, **kw)
    elif reader == "images":
        from echoprep.readers.images import read_images
        cine = read_images(path, max_frames=max_frames, **kw)
    else:
        raise ValueError(f"unknown reader {reader!r}")
    cine.meta.setdefault("fps_reader", cine.fps)
    cine.meta.setdefault("fps_container", cine.fps if reader == "video" else None)
    cine.meta.setdefault("fps_dataset_table", None)
    if "fps_source" not in cine.meta:
        source = "unknown"
        if cine.fps is not None:
            source = "container" if reader == "video" else ("dicom_metadata" if reader == "dicom" else "reader_metadata")
            if "fps" in kw:
                source = "caller"
            elif "cfg_FrameRate" in cine.meta:
                source = "dataset_config"
        cine.meta["fps_source"] = source
    cine.dataset = dataset or cine.dataset
    return cine
