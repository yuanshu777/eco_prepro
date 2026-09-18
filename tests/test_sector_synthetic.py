import numpy as np

from echoprep.cine import Cine
from echoprep.sector.motion import detect_sector
from echoprep.standardize import standardize_cine


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_detects_fan_and_ignores_overlays(synthetic_cine):
    frames, fan = synthetic_cine
    sec = detect_sector(frames)
    assert sec.status == "SUCCESS", sec.stats
    assert iou(sec.mask, fan) > 0.9, iou(sec.mask, fan)
    x0, y0, x1, y1 = sec.bbox
    ys, xs = np.where(fan)
    assert abs(x0 - xs.min()) <= 4 and abs(x1 - 1 - xs.max()) <= 4
    assert abs(y0 - ys.min()) <= 4 and abs(y1 - 1 - ys.max()) <= 4
    assert not sec.mask[8:40, 6:60].any()          # text block excluded
    assert not sec.mask[30:210, 306:311].any()      # grey bar excluded
    assert not sec.mask[226:236, :].any()           # ECG/timeline band excluded
    assert sec.stats["fan_opening_angle_deg"] > 60  # 2 * 42 deg fan, roughly


def test_static_image_falls_back(synthetic_cine):
    frames, fan = synthetic_cine
    sec = detect_sector(frames[:1], static=True)
    assert sec.status == "STATIC"
    assert iou(sec.mask, fan) > 0.85


def test_standardize_is_square_and_masked(synthetic_cine):
    frames, _fan = synthetic_cine
    cine = Cine(frames, 30.0, "synthetic.mp4", dataset="synthetic")
    sec = detect_sector(frames)
    out, rec = standardize_cine(cine, sec)
    assert out.height == out.width
    assert out.n_frames == cine.n_frames and out.fps == 30.0
    assert rec["crop_xyxy"] == list(sec.bbox)
    assert rec["mask_applied"]
