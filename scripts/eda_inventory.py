"""Frozen local inventory for EV9V, MIMIC-IV-Echo and EchoNet-Dynamic.

Reads headers and frame-directory names only. Individual records remain in outputs/.
Run from the repository root with its existing virtual environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pydicom


def stable_id(dataset, value):
    return (
        dataset.lower().split("-")[0] + "_" + hashlib.sha256(str(value).encode()).hexdigest()[:16]
    )


def number(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def timing(ds, n):
    """Preserve frame intervals, acquisition span and playback duration separately.

    DICOM FrameTimeVector starts with zero. Its sum is the first-to-last span;
    the displayed last frame is given one mean interval for playback duration.
    """
    ft = number(ds.get("FrameTime"))
    cine_rate = number(ds.get("CineRate"))
    display = number(ds.get("RecommendedDisplayFrameRate"))
    out = {
        "frame_time_ms": ft,
        "cine_rate": cine_rate,
        "display_fps": display,
        "fps": None,
        "timing_source": "missing",
        "duration_s": None,
        "time_span_s": None,
        "timing_conflict": False,
        "variable_intervals": False,
        "vector_invalid": False,
    }
    ftv = ds.get("FrameTimeVector")
    if ftv is not None:
        try:
            v = np.asarray(ftv, dtype=float).reshape(-1)
            valid = (
                len(v) == n and n > 1 and v[0] == 0 and np.all(np.isfinite(v)) and np.all(v[1:] > 0)
            )
            if valid:
                mean = float(v[1:].mean())
                out.update(
                    fps=1000 / mean,
                    timing_source="FrameTimeVector",
                    time_span_s=float(v.sum() / 1000),
                    duration_s=float((v.sum() + mean) / 1000),
                    variable_intervals=bool(np.ptp(v[1:]) > 0.01),
                )
            else:
                out["vector_invalid"] = True
        except (TypeError, ValueError):
            out["vector_invalid"] = True
    if out["fps"] is None and ft is not None and ft > 0:
        out.update(
            fps=1000 / ft,
            timing_source="FrameTime",
            duration_s=n * ft / 1000,
            time_span_s=max(0, n - 1) * ft / 1000,
        )
    if out["fps"] is None:
        for source, rate in [("CineRate", cine_rate), ("RecommendedDisplayFrameRate", display)]:
            if rate is not None and rate > 0:
                out.update(
                    fps=rate,
                    timing_source=source,
                    duration_s=n / rate,
                    time_span_s=max(0, n - 1) / rate,
                )
                break
    alternatives = [
        x for x in [1000 / ft if ft and ft > 0 else None, cine_rate, display] if x and x > 0
    ]
    if out["fps"] and alternatives:
        out["timing_conflict"] = any(abs(x / out["fps"] - 1) > 0.05 for x in alternatives)
    if n <= 1:
        out.update(duration_s=None, time_span_s=None)
    return out


def video_header(row):
    row = dict(row)
    p = Path(row["path"])
    row["file_exists"] = p.is_file()
    row["header_error"] = ""
    row["pixel_decode_checked"] = False
    if not p.is_file():
        row["header_error"] = "file_missing"
        return row
    row.update(file_bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
    try:
        with av.open(str(p)) as c:
            st = c.streams.video[0]
            row.update(
                width=st.width,
                height=st.height,
                container_frames=st.frames or None,
                container_fps=float(st.average_rate) if st.average_rate else None,
                codec=st.codec_context.name,
                container_duration_s=float(st.duration * st.time_base) if st.duration else None,
            )
        row["timing_conflict"] = bool(
            row.get("container_fps") and abs(row["container_fps"] / row["fps"] - 1) > 0.05
        )
        row["frame_count_conflict"] = bool(
            row.get("container_frames") and row["container_frames"] != row["n_frames"]
        )
    except Exception as exc:  # noqa: BLE001 - isolate and record per-file decode failures
        row["header_error"] = type(exc).__name__
    if row["dataset"] == "EV9V":
        folder = Path(row["frames_dir"])
        names = [p.name for p in folder.glob("*.jpg")]
        nums = sorted(int(Path(x).stem) for x in names if Path(x).stem.isdigit())
        row["jpeg_count"] = len(names)
        row["jpeg_contiguous"] = nums == list(range(row["n_frames"]))
        row["jpeg_count_conflict"] = len(names) != row["n_frames"]
    return row


def dicom_header(row):
    row = dict(row)
    p = Path(row["path"])
    row.update(
        file_exists=True,
        file_bytes=p.stat().st_size,
        mtime_ns=p.stat().st_mtime_ns,
        header_error="",
        pixel_decode_checked=False,
        format="DICOM",
        dataset="MIMIC-IV-Echo",
        original_view=None,
        view_source="unlabelled",
        family_view=None,
        split="not assigned",
        patient_id_known=True,
    )
    try:
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        n0 = number(ds.get("NumberOfFrames"))
        n = int(n0) if n0 is not None and n0 > 0 else 1
        regs = list(ds.get("SequenceOfUltrasoundRegions", []))
        pairs = [
            (number(g.get("RegionSpatialFormat")), number(g.get("RegionDataType"))) for g in regs
        ]
        bmode = bool(pairs) and all(x == (1.0, 1.0) for x in pairs)
        mode = (
            "2D tissue only" if bmode else ("Other/mixed region codes" if pairs else "Unknown mode")
        )
        spacing = [
            g
            for g in regs
            if number(g.get("PhysicalUnitsXDirection")) == 3
            and number(g.get("PhysicalUnitsYDirection")) == 3
            and number(g.get("PhysicalDeltaX")) not in (None, 0)
            and number(g.get("PhysicalDeltaY")) not in (None, 0)
        ]
        row.update(
            n_frames=n,
            frame_count_tag_present=n0 is not None,
            width=number(ds.get("Columns")),
            height=number(ds.get("Rows")),
            image_mode=mode,
            dynamic=n > 1,
            n_regions=len(regs),
            region_pairs=json.dumps(pairs),
            bmode_cine=bmode and n > 1,
            bits_stored=number(ds.get("BitsStored")),
            samples_per_pixel=number(ds.get("SamplesPerPixel")),
            photometric=str(ds.get("PhotometricInterpretation", "")),
            manufacturer=str(ds.get("Manufacturer", "")),
            model=str(ds.get("ManufacturerModelName", "")),
            heart_rate=number(ds.get("HeartRate")),
            waveform_present=bool(ds.get("WaveformSequence")),
            rwave_vector_present=bool(ds.get("RWaveTimeVector")),
            spatial_calibration_cm=bool(spacing),
            pixel_spacing_tag_present=bool(ds.get("PixelSpacing")),
            sop_uid=str(ds.get("SOPInstanceUID", "")),
            view_tag_present=bool(ds.get("ViewPosition")) or bool(ds.get("ViewCodeSequence")),
            burned_annotation_tag=str(ds.get("BurnedInAnnotation", "")),
            codec=str(ds.file_meta.get("TransferSyntaxUID", "")),
        )
        row.update(timing(ds, n))
    except Exception as exc:  # noqa: BLE001 - isolate and record per-file decode failures
        row["header_error"] = type(exc).__name__
    return row


def pooled_map(fn, rows, workers, label):
    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(fn, rows), 1):
            out.append(result)
            if i % 2000 == 0:
                print(label, i, "/", len(rows), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--routing-root", type=Path, default=Path("/home/william/echo-view-routing"))
    ap.add_argument(
        "--mimic-roots",
        type=Path,
        nargs="+",
        default=[Path("/mnt/mimic/mimic-iv-echo/1.0.1"), Path("/home/william/mimic-iv-echo/1.0.1")],
    )
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "tables").mkdir()
    (out / "figures").mkdir()
    root = args.routing_root
    inputs = [
        root / "data/manifests/ev9v_native.csv",
        root / "data/echonet/dynamic/EchoNet-Dynamic/FileList.csv",
        root / "data/echonet/dynamic/EchoNet-Dynamic/VolumeTracings.csv",
        args.mimic_roots[0] / "echo-record-list.csv",
    ]
    provenance = {
        "started_utc": datetime.now(UTC).isoformat(),
        "routing_root": str(root),
        "seed": 20261005,
        "input_tables": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    }
    ev = pd.read_csv(inputs[0])
    en = pd.read_csv(inputs[1])
    tr = pd.read_csv(inputs[2])
    rec = pd.read_csv(inputs[3], dtype=str)
    found = {}
    physical, partial = 0, 0
    for base in args.mimic_roots:
        for dp, _, files in os.walk(base / "files"):
            for name in files:
                if name.endswith(".dcm"):
                    p = Path(dp) / name
                    found.setdefault(str(p.relative_to(base)), str(p))
                    physical += 1
                elif name.endswith(".part"):
                    partial += 1
    rec["downloaded"] = rec.dicom_filepath.isin(found)
    studies = (
        rec.groupby(["subject_id", "study_id"])
        .agg(expected=("downloaded", "size"), downloaded=("downloaded", "sum"))
        .reset_index()
    )
    studies["complete"] = studies.expected == studies.downloaded
    studies["started"] = studies.downloaded > 0
    studies.to_csv(out / "tables/mimic_study_downloads.csv", index=False)
    rec.to_csv(out / "tables/mimic_official_inventory_snapshot.csv", index=False)
    complete = set(studies.loc[studies.complete, "study_id"])
    lookup = rec.set_index("dicom_filepath").to_dict("index")
    mr = []
    for rel, path in sorted(found.items()):
        info = lookup.get(rel, {})
        mr.append(
            {
                "case_id": stable_id("mimic", rel),
                "path": path,
                "relative_path": rel,
                "subject_id": info.get("subject_id"),
                "study_id": info.get("study_id"),
                "acquisition_datetime": info.get("acquisition_datetime"),
                "study_complete": info.get("study_id") in complete,
                "listed_in_inventory": rel in lookup,
            }
        )
    er = []
    for r in ev.to_dict("records"):
        frames_dir = Path(r["frames_dir"])
        video_path = Path(r["video_path"])
        if not video_path.is_absolute():
            video_path = root / video_path
        if not frames_dir.is_absolute():
            frames_dir = root / frames_dir
        er.append(
            {
                "case_id": stable_id("ev9v", r["video_id"]),
                "dataset": "EV9V",
                "format": "MP4",
                "path": str(video_path),
                "frames_dir": str(frames_dir),
                "original_id": r["video_id"],
                "session": r["video_id"].split("_320x240")[0],
                "patient_id_known": False,
                "original_view": r["raw_label"],
                "family_view": r["family5_label"],
                "view_source": "dataset clip label",
                "split": r["split"],
                "n_frames": r["n_frames"],
                "fps": r["fps_playback"],
                "duration_s": r["n_frames"] / r["fps_playback"],
                "time_span_s": (r["n_frames"] - 1) / r["fps_playback"],
                "timing_source": "dataset playback FPS",
                "dynamic": r["n_frames"] > 1,
                "image_mode": "B-mode by dataset scope",
                "spatial_calibration_cm": False,
            }
        )
    trace_count = tr.groupby("FileName").Frame.nunique()
    nr = []
    for r in en.to_dict("records"):
        name = str(r["FileName"]).removesuffix(".avi")
        fps, n = r["FPS"], r["NumberOfFrames"]
        nr.append(
            {
                "case_id": stable_id("echonet", name),
                "dataset": "EchoNet-Dynamic",
                "format": "AVI",
                "path": str(inputs[1].parent / "Videos" / f"{name}.avi"),
                "original_id": name,
                "patient_id_known": False,
                "original_view": "A4C",
                "family_view": "A4C",
                "view_source": "dataset inclusion criterion",
                "split": r["Split"].lower(),
                "n_frames": n,
                "fps": fps,
                "duration_s": n / fps,
                "time_span_s": (n - 1) / fps,
                "timing_source": "FileList FPS",
                "dynamic": n > 1,
                "image_mode": "B-mode by dataset scope",
                "declared_width": r["FrameWidth"],
                "declared_height": r["FrameHeight"],
                "ef": r["EF"],
                "esv": r["ESV"],
                "edv": r["EDV"],
                "traced_frames": int(trace_count.get(f"{name}.avi", 0)),
                "spatial_calibration_cm": False,
            }
        )
    allrows = pooled_map(video_header, er + nr, args.workers, "video headers")
    allrows += pooled_map(dicom_header, mr, args.workers, "DICOM headers")
    df = pd.DataFrame(allrows)
    df.to_csv(out / "tables/video_inventory.csv", index=False)
    facts = dict(
        provenance,
        official_mimic_records=len(rec),
        official_mimic_studies=len(studies),
        official_mimic_patients=rec.subject_id.nunique(),
        local_mimic_files=len(found),
        local_mimic_physical_files=physical,
        local_mimic_part_files=partial,
        local_mimic_started_studies=int(studies.started.sum()),
        local_mimic_complete_studies=int(studies.complete.sum()),
        local_mimic_patients_complete=int(studies.loc[studies.complete, "subject_id"].nunique()),
        local_mimic_unlisted=sum(rel not in lookup for rel in found),
        ev9v_manifest_shape=list(ev.shape),
        echonet_filelist_shape=list(en.shape),
        echonet_tracings_shape=list(tr.shape),
        official_mimic_inventory_shape=[len(rec), len(rec.columns) - 1],
        table_rows=len(df),
        table_columns=len(df.columns),
        rows_by_dataset=dict(Counter(df.dataset)),
        ev9v_original_label_counts=dict(Counter(ev.raw_label)),
        ev9v_unmapped=int(ev.family5_label.isna().sum()),
        finished_utc=datetime.now(UTC).isoformat(),
    )
    (out / "inventory_summary.json").write_text(json.dumps(facts, indent=2))
    print(
        "COMPLETE",
        {
            k: facts[k]
            for k in [
                "rows_by_dataset",
                "local_mimic_files",
                "local_mimic_complete_studies",
                "table_columns",
            ]
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
