"""Protect EDA denominators and timing semantics at their important boundaries."""

import pytest
from pydicom.dataset import Dataset

from scripts.eda_inventory import timing


def test_variable_intervals_preserve_span_and_playback_duration():
    ds = Dataset()
    ds.FrameTimeVector = [0, 20, 40, 60]
    ds.FrameTime = 30
    result = timing(ds, 4)
    assert result["timing_source"] == "FrameTimeVector"
    assert result["time_span_s"] == pytest.approx(0.120)
    assert result["duration_s"] == pytest.approx(0.160)
    assert result["fps"] == pytest.approx(25)
    assert result["variable_intervals"]
    assert result["timing_conflict"]


def test_bad_vector_is_recorded_before_falling_back():
    ds = Dataset()
    ds.FrameTimeVector = [0, 0, 20]
    ds.FrameTime = 40
    result = timing(ds, 3)
    assert result["vector_invalid"]
    assert result["timing_source"] == "FrameTime"
    assert result["duration_s"] == pytest.approx(0.120)


def test_unknown_dynamic_timing_does_not_invent_30_fps():
    result = timing(Dataset(), 100)
    assert result["fps"] is None
    assert result["duration_s"] is None
    assert result["timing_source"] == "missing"


def test_static_image_does_not_become_a_zero_second_video():
    ds = Dataset()
    ds.CineRate = 30
    result = timing(ds, 1)
    assert result["duration_s"] is None
    assert result["time_span_s"] is None


def test_display_rate_is_labelled_as_display_not_acquisition():
    ds = Dataset()
    ds.RecommendedDisplayFrameRate = 25
    result = timing(ds, 50)
    assert result["timing_source"] == "RecommendedDisplayFrameRate"
    assert result["duration_s"] == 2
