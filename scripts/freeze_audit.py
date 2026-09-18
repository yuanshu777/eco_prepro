#!/usr/bin/env python
"""One-time migration of the existing audit. Refuses to overwrite a frozen manifest."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import SimpleITK as sitk
import yaml

from echoprep.audit.identity import content_sha256, stable_case_id

ROOT = Path(__file__).resolve().parents[1]


def freeze():
    target = ROOT / "configs/audit_manifest.csv"
    if target.exists():
        print(f"Frozen manifest already exists; unchanged: {target}")
        return
    cfg = yaml.safe_load((ROOT / "configs/datasets.yaml").read_text())["datasets"]
    with (ROOT / "data/audit_set/audit_manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["legacy_id"] = row.pop("id")
        tags = {
            "ev9v": "grayscale;raw_screen;fan;ECG;depth_ticks;machine_text",
            "echonet_lvh": "grayscale;sector_only;large_resolution;fps_mismatch;flat_top_fan",
            "echonet_dynamic": "grayscale;sector_only;small_resolution",
            "echonet_pediatric": "sector_only;small_resolution;pediatric;color_tint",
            "camus": "grayscale;fan;nifti;short_cine",
            "unity": "static;raw_screen;vendor_layout;manual_review",
            "samples_dicom": "dicom;static;noncardiac;codec_test;rectangular_field_still;power_doppler_noncardiac",
        }
        row.update(case_type=tags[row["dataset"]], derived_from="", derivation="")
        if row["legacy_id"] == "ev9v_002":
            row["case_type"] += ";dark_sector"
        if row["legacy_id"] in {"unity_028", "unity_029", "unity_030"}:
            row["case_type"] += ";color_tint_still"
        if row["legacy_id"] == "unity_031":
            row["case_type"] += ";dark_sector"
        if row["legacy_id"] == "samples_dicom_038":
            row["case_type"] = "dicom;historical_color_cine;multi_region;RLE;palette_color"
        row["content_sha256"] = content_sha256(row["path"])
        row["id"] = stable_case_id(row["dataset"], row["content_sha256"])
    # Re-encode an existing sequence solely to exercise the MHD reader; not new clinical coverage.
    parent = next(x for x in rows if x["legacy_id"] == "camus_022")
    derived = ROOT / "data/audit_derived/camus"
    derived.mkdir(parents=True, exist_ok=True)
    mhd = derived / Path(parent["path"]).name.replace(".nii.gz", ".mhd")
    sitk.WriteImage(sitk.ReadImage(parent["path"]), str(mhd))
    view = "2CH"
    shutil.copyfile(Path(parent["path"]).parent / f"Info_{view}.cfg", derived / f"Info_{view}.cfg")
    rows.append(
        {
            "dataset": "camus",
            "path": str(mhd),
            "reader": "sitk",
            "label": "2CH MHD reader fixture",
            "legacy_id": "",
            "case_type": "mhd;derived_format_fixture;short_cine",
            "derived_from": parent["id"],
            "derivation": "SimpleITK NIfTI to MHD; same pixels, orientation and sibling FPS config",
        }
    )
    long_path = ROOT / cfg["ev9v"]["root"] / "Videos/2022-05-03_09-59-32_000_320x240_6.mp4"
    if long_path.exists():
        rows.append(
            {
                "dataset": "ev9v",
                "path": str(long_path),
                "reader": "video",
                "label": "3243-frame long cine",
                "legacy_id": "",
                "case_type": "very_long_cine;raw_screen;fan;ECG",
                "derived_from": "",
                "derivation": "",
            }
        )
    # Exercise the previously unaudited image-folder reader using an existing matching EV9V case.
    for parent in rows[:9]:
        folder = ROOT / cfg["ev9v"]["root"] / "Images" / Path(parent["path"]).stem
        if folder.is_dir():
            rows.append(
                {
                    "dataset": "ev9v",
                    "path": str(folder),
                    "reader": "images",
                    "label": "existing JPEG frame sequence",
                    "legacy_id": "",
                    "case_type": "image_folder;raw_screen;fps_unknown",
                    "derived_from": parent["id"],
                    "derivation": "Existing extracted JPEGs; no FPS inferred from file names",
                }
            )
            break
    fields = [
        "id",
        "legacy_id",
        "dataset",
        "label",
        "reader",
        "path_base",
        "path",
        "content_sha256",
        "case_type",
        "derived_from",
        "derivation",
        "source_url",
        "access_method",
        "license",
        "redistributable",
        "redistribution_restrictions",
    ]
    for row in rows:
        path = Path(row["path"])
        digest = row.setdefault("content_sha256", content_sha256(path))
        row.setdefault("id", stable_case_id(row["dataset"], digest))
        d = cfg[row["dataset"]]
        try:
            row["path"] = str(path.resolve().relative_to((ROOT / d["root"]).resolve()))
            row["path_base"] = "dataset"
        except ValueError:
            row["path"] = str(path.relative_to(ROOT))
            row["path_base"] = "project"
        for key in fields[-5:]:
            row[key] = d[key]
        row["source_url"] = d["source_url"]
    if len({x["id"] for x in rows}) != len(rows):
        raise ValueError("Duplicate content identities; review before freezing")
    with target.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with (ROOT / "configs/audit_id_map.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["legacy_id", "id", "content_sha256"])
        w.writeheader()
        w.writerows({k: row[k] for k in w.fieldnames} for row in rows if row["legacy_id"])
    print(f"Frozen {len(rows)} cases -> {target}")


if __name__ == "__main__":
    freeze()
