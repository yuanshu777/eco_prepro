"""Conservative UI-only exclusion profiles. Profiles never supply a fan shape or apex."""

from __future__ import annotations

import numpy as np


def ev9v_hint_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, bool)
    for x0, y0, x1, y1 in [(0, 0, 0.2, 0.16), (0.95, 0.1, 1, 0.88), (0, 0.95, 1, 1)]:
        mask[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = True
    return mask


def allowed_area(frames: np.ndarray, exclusion_mask: np.ndarray | None):
    """Use explicit caller zones; refine the known EV9V hint only after screen registration.

    The fixed 320x240 profile is not scaled to arbitrary scans. A matching caller hint plus a
    distinctive top header and gray bar are required. Slanted ticks near/inside the fan are not
    hard-excluded: geometry must distinguish them without a box that can delete anatomy.
    """
    shape = frames.shape[1:3]
    excluded = (
        np.zeros(shape, bool) if exclusion_mask is None else np.asarray(exclusion_mask, bool).copy()
    )
    if excluded.shape != shape:
        raise ValueError("exclusion mask must match frame dimensions")
    name = "caller_zones" if excluded.any() else "unknown_layout"
    if shape == (240, 320) and np.array_equal(excluded, ev9v_hint_mask(shape)) and frames.ndim == 4:
        sample = frames[
            np.linspace(0, len(frames) - 1, min(3, len(frames))).round().astype(int)
        ].astype(float)
        header = sample[:, :12]
        gold = (
            (header[..., 0] > 90) & (header[..., 1] > 80) & (header[..., 0] - header[..., 2] > 15)
        )
        gray_bar = sample[:, 45:80, 297:305].mean()
        if gold.mean() > 0.80 and gray_bar > 60:
            name = "ev9v_philips_320_registered"
            # Fixed UI: header/text, the external gray bar and outer depth-scale column.
            # Bottom exclusion is the existing caller-provided ECG zone (not an inferred fan).
            excluded[:25, :] = True
            excluded[:95, :48] = True
            excluded[:40, 250:] = True
            excluded[35:135, 296:] = True
            excluded[25:215, 304:] = True
    return ~excluded, name
