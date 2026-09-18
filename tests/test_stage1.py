"""Regression checks for anatomy preservation, provenance and reproducible audit infrastructure."""

from __future__ import annotations

import io
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from echoprep.acquisition import acquire_samples
from echoprep.audit.identity import content_sha256, stable_case_id
from echoprep.cine import Cine
from echoprep.readers import read_cine
from echoprep.readers.fps_tables import apply_fps_provenance
from echoprep.sector import motion
from echoprep.standardize import export_cine, standardize_cine


def test_passthrough_is_early_and_pixel_exact(monkeypatch, synthetic_cine):
    frames, _ = synthetic_cine
    frames = frames[:, 30:191, 60:255].copy()  # deliberately rectangular and odd-sized
    monkeypatch.setattr(
        motion, "_propagate", lambda *args: pytest.fail("pass-through entered region growth")
    )
    sec = motion.detect_sector(frames, already_standardized=True)
    assert sec.status == "PASSTHROUGH"
    cine = Cine(frames, 19.97, "test.avi")
    out, rec = standardize_cine(cine, sec)
    assert np.array_equal(out.frames, frames)
    assert rec["crop_xyxy"] == [0, 0, 195, 161]
    assert rec["pad_tblr"] == [0, 0, 0, 0]
    assert not rec["mask_applied"]


def test_raw_screen_not_mistaken_for_passthrough(synthetic_cine):
    frames, _ = synthetic_cine
    sec = motion.detect_sector(frames)
    assert sec.status == "SUCCESS"
    assert not sec.precropped


def test_two_moving_panels_flagged_and_preserved():
    rng = np.random.default_rng(3)
    frames = np.zeros((12, 100, 180, 3), np.uint8)
    for x in (10, 100):
        frames[:, 20:80, x : x + 60] = rng.integers(20, 200, (12, 60, 60, 1), dtype=np.uint8)
    sec = motion.detect_sector(frames)
    assert sec.status == "MULTI_REGION"
    assert sec.stats["candidate_regions"] == 2
    out, rec = standardize_cine(Cine(frames, 30, "two"), sec)
    assert np.array_equal(out.frames, frames)
    assert rec["action"] == "retained_for_review"


@pytest.mark.parametrize(
    "kind,status", [("blank", "FAILED"), ("still", "STATIC"), ("low", "LOW_CONFIDENCE")]
)
def test_uncertain_cases_retain_original(kind, status):
    frames = np.zeros((1 if kind == "still" else 8, 100, 140, 3), np.uint8)
    if kind != "blank":
        frames[:, 10:90, 20:120] = 100
    sec = motion.detect_sector(frames)
    assert sec.status == status
    out, rec = standardize_cine(Cine(frames, None, "uncertain"), sec)
    assert np.array_equal(out.frames, frames)
    assert rec["action"] == "retained_for_review"


def test_overlay_and_color_guards(synthetic_cine):
    frames, _ = synthetic_cine
    exclusion = np.ones(frames.shape[1:3], bool)
    sec = motion.detect_sector(frames, exclusion_mask=exclusion)
    assert sec.status == "LOW_CONFIDENCE"
    assert "overlay_contact" in sec.flags
    assert sec.stats["overlay_contact_frac"] == 1
    assert sec.stats["color_fraction_outside"] > sec.stats["color_fraction_inside"]


def test_fps_override_survives_export_and_codec_padding(tmp_path):
    (tmp_path / "fps.csv").write_text("name,rate\ntest,19.97\n")
    frames = np.full((4, 31, 45, 3), 80, np.uint8)
    cine = Cine(frames, 50, str(tmp_path / "test.avi"), reader="video")
    apply_fps_provenance(
        cine, {"fps_table": {"csv": "fps.csv", "key": "name", "fps": "rate"}}, tmp_path
    )
    sec = motion.detect_sector(frames)
    out, rec = standardize_cine(cine, sec)
    p = export_cine(out, tmp_path / "out.mp4", rec)
    record = json.loads(p.with_suffix(".json").read_text())
    assert record["fps_container"] == 50
    assert record["fps_dataset_table"] == record["fps_value"] == 19.97
    assert record["fps_source"] == "dataset_table"
    assert record["export"]["pad_tblr"] == [0, 1, 0, 1]
    back = read_cine(p)
    assert back.frames.shape == (4, 32, 46, 3)
    assert back.fps == pytest.approx(19.97, abs=0.001)


def test_unknown_fps_not_fabricated_in_sidecar(tmp_path):
    cine = Cine(np.zeros((1, 20, 20, 3), np.uint8), None, "still")
    sec = motion.detect_sector(cine.frames)
    out, rec = standardize_cine(cine, sec)
    p = export_cine(out, tmp_path / "still.mp4", rec)
    record = json.loads(p.with_suffix(".json").read_text())
    assert record["fps_value"] is None
    assert record["export"]["fps_assumed"] is True
    p = export_cine(out, tmp_path / "still.npz", rec)
    assert np.isnan(np.load(p)["fps"])
    assert json.loads(p.with_suffix(".json").read_text())["export"]["fps_written"] is None


def test_mhd_roundtrip_includes_timing_and_compound_identity(tmp_path, synthetic_cine):
    import SimpleITK as sitk

    frames, _ = synthetic_cine
    p = tmp_path / "patient_2CH_half_sequence.mhd"
    sitk.WriteImage(sitk.GetImageFromArray(frames[..., 0]), str(p))
    (tmp_path / "Info_2CH.cfg").write_text("FrameRate: 48.4\n")
    back = read_cine(p)
    assert np.array_equal(back.frames[..., 0], frames[..., 0])
    assert back.fps == 48.4 and back.meta["fps_source"] == "dataset_config"
    original = content_sha256(p)
    raw = p.with_suffix(".raw")
    with raw.open("r+b") as f:
        f.write(b"\xff")
    assert original != content_sha256(p)


def test_identity_survives_move_and_manifest_insertion(tmp_path):
    a = tmp_path / "a.mp4"
    a.write_bytes(b"video bytes")
    b = tmp_path / "renamed.mp4"
    shutil.copy(a, b)
    first = stable_case_id("example", content_sha256(a))
    assert stable_case_id("example", content_sha256(b)) == first
    ids = [stable_case_id("example", content_sha256(p)) for p in (b, a)]
    assert ids == [first, first]
    b.write_bytes(b"changed")
    assert stable_case_id("example", content_sha256(b)) != first


@pytest.mark.parametrize(
    "access,status",
    [("archive", "ARCHIVE_ONLY"), ("restricted", "AUTH_REQUIRED"), ("unknown", "UNAVAILABLE")],
)
def test_unavailable_sources_are_explicit(tmp_path, access, status):
    r = acquire_samples("x", {"root": "missing", "access_method": access}, tmp_path)
    assert r["status"] == status


def test_acquisition_local_first_and_disabled_network(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"local")
    r = acquire_samples(
        "x",
        {"root": ".", "videos_glob": "*.bin"},
        tmp_path,
        opener=lambda *a, **k: pytest.fail("network used for local input"),
    )
    assert r["status"] == "FOUND_LOCAL"
    assert (
        acquire_samples("x", {"root": "missing"}, tmp_path, download=False)["status"] == "SKIPPED"
    )


def test_bounded_download_and_failure_cleanup(tmp_path):
    spec = {
        "root": "public",
        "access_method": "public_samples",
        "sample_urls": [
            {"url": "https://example.invalid/sample", "filename": "a.bin", "max_bytes": 10}
        ],
    }
    r = acquire_samples("x", spec, tmp_path, opener=lambda *a, **k: io.BytesIO(b"sample"))
    assert r["status"] == "DOWNLOADED"
    (tmp_path / "public/a.bin").unlink()
    r = acquire_samples("x", spec, tmp_path, opener=lambda *a, **k: io.BytesIO(b"x" * 20))
    assert r["status"] == "UNAVAILABLE"
    assert not list((tmp_path / "public").iterdir())


def test_unity_restriction_and_manifest_coverage():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/datasets.yaml").read_text())["datasets"]
    assert cfg["unity"]["redistributable"] is False
    df = pd.read_csv(root / "configs/audit_manifest.csv")
    assert not df.id.duplicated().any()
    assert len(df[df.legacy_id.notna()]) == 39
    assert not df[df.dataset == "unity"].redistributable.any()
    assert {"sitk", "images", "dicom", "video", "nifti"} <= set(df.reader)


def test_comparison_matches_by_identity_not_row_order():
    from scripts.compare_audits import compare

    before = pd.DataFrame({"id": ["old-a", "old-b"], "status": ["ok", "ok"], "out_h": [10, 20]})
    after = pd.DataFrame(
        {
            "id": ["b", "a", "c"],
            "status": ["SUCCESS", "PASSTHROUGH", "STATIC"],
            "out_h": [20, 12, 30],
        }
    )
    result = compare(before, after, {"old-a": "a", "old-b": "b"}).set_index("id")
    assert result.loc["a", "out_h_delta"] == 2
    assert result.loc["b", "out_h_delta"] == 0
    assert result.loc["c", "presence"] == "added"
    assert result.loc["a", "status_changed"]


def test_static_reference_panel_alongside_moving_panel():
    rng = np.random.default_rng(7)
    frames = np.zeros((12, 120, 200, 3), np.uint8)
    frames[:, 10:95, 90:180] = rng.integers(20, 200, (12, 85, 90, 1), dtype=np.uint8)
    frames[:, 60:100, 15:50] = 90  # smaller, completely static reference image
    frames[:, 5:90, 190:197] = 200  # narrow UI color/gray scale is not a panel
    sec = motion.detect_sector(frames)
    assert sec.status == "MULTI_REGION"
    assert sec.stats["candidate_regions"] == 2
