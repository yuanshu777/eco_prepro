"""Known-shape geometry and preservation regressions; no real data required."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from echoprep.sector.geometry import fit_geometry
from echoprep.sector.layouts import allowed_area, ev9v_hint_mask
from echoprep.sector.motion import detect_sector


def fan(apex=(180, 20), axis=0, half=42, radius=240, near=0, inner=0):
    y, x = np.mgrid[:300, :360]
    dx, dy = x - apex[0], y - apex[1]
    r = np.hypot(dx, dy)
    a = (np.degrees(np.arctan2(dx, dy)) - axis + 180) % 360 - 180
    forward = dx * np.sin(np.radians(axis)) + dy * np.cos(np.radians(axis))
    return (abs(a) <= half) & (r <= radius) & (r >= inner) & (forward >= near)


def iou(a, b):
    return (a & b).sum() / max(int((a | b).sum()), 1)


@pytest.mark.parametrize(
    "kwargs,family",
    [
        ({}, "standard_fan"),
        ({"half": 22, "radius": 210}, "standard_fan"),
        ({"half": 60, "radius": 190}, "standard_fan"),
        ({"apex": (180, -60), "half": 30, "radius": 275, "near": 100}, "flat_top_fan"),
        ({"apex": (180, -20), "half": 35, "inner": 70}, "flat_top_fan"),
        ({"apex": (180, 50), "axis": 35, "half": 30, "radius": 185}, "rotated_wedge"),
        ({"apex": (180, 100), "axis": -60, "half": 25, "radius": 160}, "rotated_wedge"),
    ],
)
def test_geometry_families(kwargs, family):
    truth = fan(**kwargs)
    fit, records = fit_geometry(truth)
    assert fit is not None, records
    assert fit.family == family
    assert iou(fit.mask, truth) > 0.96
    assert (fit.mask & truth).sum() / truth.sum() > 0.99


def test_recovers_supported_dark_gap():
    truth = fan()
    y, x = np.mgrid[:300, :360]
    r = np.hypot(x - 180, y - 20)
    a = np.degrees(np.arctan2(x - 180, y - 20))
    evidence = truth & ~((abs(a) < 18) & (r > 210))
    fit, _ = fit_geometry(evidence)
    assert fit is not None
    assert iou(fit.mask, truth) > 0.97
    assert (fit.mask & (truth & ~evidence)).sum() / (truth & ~evidence).sum() > 0.98


def test_rectangle_and_allowed_hole_are_preserved():
    evidence = np.zeros((240, 320), bool)
    evidence[40:190, 60:260] = True
    allowed = np.ones_like(evidence)
    allowed[75:90, 90:110] = False
    fit, _ = fit_geometry(evidence, allowed)
    assert fit is not None and fit.family == "rectangle"
    assert not (fit.mask & ~allowed).any()
    assert iou(fit.mask, evidence & allowed) > 0.99


def test_fan_cannot_expand_into_excluded_ui():
    truth = fan()
    allowed = np.ones_like(truth)
    allowed[250:] = False
    allowed[170:180, 140:155] = False
    fit, _ = fit_geometry(truth, allowed)
    assert fit is not None
    assert not (fit.mask & ~allowed).any()


def test_irregular_evidence_is_rejected():
    y, x = np.mgrid[:300, :360]
    evidence = ((x - 180) / 100) ** 2 + ((y - 145) / 75) ** 2 < 1
    fit, records = fit_geometry(evidence)
    assert fit is None
    assert records and all(r["reasons"] for r in records)


def test_layout_requires_both_source_hint_and_registration():
    frames = np.zeros((2, 240, 320, 3), np.uint8)
    hint = ev9v_hint_mask((240, 320))
    allowed, name = allowed_area(frames, hint)
    assert name == "caller_zones" and np.array_equal(allowed, ~hint)
    frames[:, :12] = (160, 150, 100)
    frames[:, 45:80, 297:305] = 150
    allowed, name = allowed_area(frames, None)
    assert name == "unknown_layout" and allowed.all()
    allowed, name = allowed_area(frames, hint)
    assert name == "ev9v_philips_320_registered"
    assert not allowed[:25].any() and not allowed[:, -1][25:215].any()
    assert np.array_equal(hint, ev9v_hint_mask((240, 320)))


@pytest.mark.parametrize("mode", ["static", "passthrough", "multi"])
def test_retained_paths_do_not_call_geometry(monkeypatch, synthetic_cine, mode):
    frames, _ = synthetic_cine
    kwargs = {}
    if mode == "static":
        frames = frames[:1]
        kwargs["static"] = True
    elif mode == "passthrough":
        kwargs["already_standardized"] = True
    else:
        frames = np.zeros((12, 240, 320, 3), np.uint8)
        rng = np.random.default_rng(7)
        frames[:, 50:190, 20:135] = rng.integers(30, 220, (12, 140, 115, 3), dtype=np.uint8)
        frames[:, 50:190, 185:300] = rng.integers(30, 220, (12, 140, 115, 3), dtype=np.uint8)
    baseline = detect_sector(frames, refine_geometry=False, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Geometry must not run on retained input")

    monkeypatch.setattr("echoprep.sector.geometry.fit_geometry", forbidden)
    result = detect_sector(frames, **kwargs)
    assert (
        result.status
        == {"static": "STATIC", "passthrough": "PASSTHROUGH", "multi": "MULTI_REGION"}[mode]
    )
    assert result.status == baseline.status
    assert result.bbox == baseline.bbox and np.array_equal(result.mask, baseline.mask)
    assert result.flags == baseline.flags


def test_conflicting_exclusion_retains_cine(synthetic_cine):
    frames, fan_mask = synthetic_cine
    excluded = np.zeros(fan_mask.shape, bool)
    excluded[130:170, 100:220] = True
    result = detect_sector(frames, exclusion_mask=excluded)
    assert result.status == "LOW_CONFIDENCE"
    assert "unsafe_exclusion_or_candidate" in result.flags


def test_temporal_repeatability_and_input_immutability(synthetic_cine):
    frames, _ = synthetic_cine
    original = frames.copy()
    a = detect_sector(frames[::2])
    b = detect_sector(frames[1::2])
    assert a.status == b.status == "SUCCESS"
    assert iou(a.mask, b.mask) > 0.98
    assert np.array_equal(frames, original)
    assert not detect_sector(frames, keep_maps=False).maps


def test_temporally_inconsistent_fit_falls_back(synthetic_cine):
    frames, _ = synthetic_cine
    frames = frames.copy()
    y, x = np.mgrid[:240, :320]
    frames[1::2, np.hypot(x - 160, y - 18) > 165] = 0
    result = detect_sector(frames)
    assert result.stats["stage2_method"] == "fallback_hull"
    assert "geometry_temporal_instability" in result.flags


def test_stage1_infrastructure_is_byte_identical():
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((root / "docs/stage2_frozen_contract.json").read_text())
    for relative, digest in frozen["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
