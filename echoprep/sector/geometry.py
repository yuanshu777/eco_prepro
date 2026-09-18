"""Deterministic, evidence-scored sector geometry. No learned model or dataset fan template.

Straight flank consensus proposes a common apex; angular boundary samples constrain radii.
Several orientations and near-field shapes are scored against observed support. Missing edges
are not ground truth: unsupported fits are explicitly rejected rather than extrapolated freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class GeometryFit:
    family: str
    mask: np.ndarray
    parameters: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    reliable: bool = False
    reasons: list[str] = field(default_factory=list)

    def record(self) -> dict:
        return {
            "family": self.family,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "reliable": self.reliable,
            "reasons": self.reasons,
        }


def _line_consensus(v: np.ndarray, u: np.ndarray, tolerance: float):
    if len(v) < 12 or np.ptp(v) < 10:
        return None
    rng = np.random.default_rng(19)
    best, best_score = None, -1.0
    for _ in range(120):
        i, j = rng.choice(len(v), 2, replace=False)
        if abs(v[i] - v[j]) < 0.25 * np.ptp(v):
            continue
        a = (u[j] - u[i]) / (v[j] - v[i])
        b = u[i] - a * v[i]
        inside = np.abs(u - (a * v + b)) / np.hypot(a, 1) <= tolerance
        score = float(inside.sum()) + 0.1 * float(np.ptp(v[inside]))
        if score > best_score:
            best, best_score = inside, score
    if best is None or best.sum() < 8:
        return None
    for _ in range(3):
        a, b = np.polyfit(v[best], u[best], 1)
        best = np.abs(u - (a * v + b)) / np.hypot(a, 1) <= tolerance
        if best.sum() < 8:
            return None
    return float(a), float(b), float(best.mean()), float(np.ptp(v[best]) / np.ptp(v))


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask)


def _score(
    fit: GeometryFit, evidence: np.ndarray, allowed: np.ndarray, motion: np.ndarray | None
) -> GeometryFit:
    proposed = fit.mask.copy()
    fit.mask &= allowed  # invariant for every family, including rectangles
    mask = fit.mask
    count = max(int(evidence.sum()), 1)
    retained = float((mask & evidence).sum() / count)
    expansion = float(mask.sum() / count)
    # Use the exterior outline: holes in speckle support are not sector boundaries.
    # Consensus allows bounded outward recovery across dark gaps, while evidence
    # retention, deep-cut, flank and outer-arc guards remain independent.
    # Boundaries imposed by allowed-area/image clipping do not score as evidence.
    boundary = _boundary(ndi.binary_fill_holes(evidence))
    boundary &= ndi.binary_erosion(allowed, iterations=2, border_value=0)
    distance = (
        ndi.distance_transform_edt(~_boundary(mask))
        if mask.any()
        else np.full(mask.shape, max(mask.shape))
    )
    residuals = distance[boundary]
    residual = float(np.quantile(residuals, 0.5)) if residuals.size else float(max(mask.shape))
    residual_p80 = float(np.quantile(residuals, 0.8)) if residuals.size else float(max(mask.shape))
    scale = float(min(mask.shape))
    boundary_consensus = (
        float((residuals <= max(3.0, 0.025 * scale)).mean()) if residuals.size else 0.0
    )
    core = ndi.binary_erosion(evidence, iterations=max(1, round(scale * 0.006)))
    core_count = max(int(core.sum()), 1)
    core_retained = float((mask & core).sum() / core_count)
    motion_retained = (
        float((mask & motion & evidence).sum() / max(int((motion & evidence).sum()), 1))
        if motion is not None
        else retained
    )
    removed_distance = ndi.distance_transform_edt(~mask)[core & ~mask]
    cut_depth = float(np.quantile(removed_distance, 0.95)) if removed_distance.size else 0.0
    iou = float((mask & evidence).sum() / max(int((mask | evidence).sum()), 1))
    fit.metrics.update(
        evidence_retained=retained,
        core_retained=core_retained,
        motion_retained=motion_retained,
        area_ratio=expansion,
        iou=iou,
        boundary_residual_px=residual,
        boundary_residual_p80_px=residual_p80,
        boundary_consensus=boundary_consensus,
        cut_depth_px=cut_depth,
        allowed_clip_fraction=float((proposed & ~allowed).sum() / max(int(proposed.sum()), 1)),
    )
    reasons = []
    if retained < 0.975 or core_retained < 0.99 or motion_retained < 0.975:
        reasons.append("evidence_clipped")
    if cut_depth > max(2.5, 0.012 * scale):
        reasons.append("deep_tissue_cut")
    if not 0.90 <= expansion <= 1.35:
        reasons.append("implausible_area_change")
    if residual > max(3.0, 0.025 * scale) or boundary_consensus < 0.60:
        reasons.append("large_boundary_residual")
    if fit.family != "rectangle":
        if fit.metrics.get("flank_consensus", 0) < 0.60 or fit.metrics.get("flank_span", 0) < 0.65:
            reasons.append("unsupported_flanks")
        if fit.metrics.get("arc_support", 0) < 0.45:
            reasons.append("unsupported_outer_arc")
    elif iou < 0.96:
        reasons.append("not_rectangular")
    fit.reliable = not reasons
    fit.reasons = reasons
    fit.metrics["score"] = float(residual / scale + 2 * (1 - retained) + 0.08 * abs(expansion - 1))
    return fit


def fit_geometry(
    evidence: np.ndarray, allowed: np.ndarray | None = None, motion: np.ndarray | None = None
) -> tuple[GeometryFit | None, list[dict]]:
    """Return the lowest-residual reliable fit and records of all candidates (including rejected).

    Evidence is a connected support region located using motion, not only moving pixels.
    Flat near planes and inner arcs are both considered for truncated/virtual-apex fans.
    Rotation is inferred from line intersections, not assumed from the image axes.
    """
    evidence = np.asarray(evidence, dtype=bool)
    allowed = np.ones_like(evidence) if allowed is None else np.asarray(allowed, dtype=bool)
    if evidence.shape != allowed.shape:
        raise ValueError("allowed area must match evidence")
    evidence = evidence & allowed
    yy, xx = np.nonzero(evidence)
    if len(xx) < 100:
        return None, []
    points = np.column_stack((xx, yy)).astype(float)
    h, w = evidence.shape
    scale = float(min(h, w))
    grid_y, grid_x = np.mgrid[:h, :w]
    candidates = []
    for orientation in range(-180, 180, 30):
        theta = np.deg2rad(orientation)
        c, s = np.cos(theta), np.sin(theta)
        u = xx * c - yy * s
        v = xx * s + yy * c
        bins = np.rint(v).astype(int)
        offset = int(bins.min())
        bins -= offset
        left = np.full(int(bins.max()) + 1, np.inf)
        right = np.full_like(left, -np.inf)
        np.minimum.at(left, bins, u)
        np.maximum.at(right, bins, u)
        row = np.arange(len(left)) + offset
        valid = np.isfinite(left)
        v0, v1 = float(v.min()), float(v.max())
        span = v1 - v0
        valid &= (row >= v0 + 0.08 * span) & (row <= v0 + 0.58 * span)
        l = _line_consensus(row[valid], left[valid], max(1.5, 0.008 * scale))
        r = _line_consensus(row[valid], right[valid], max(1.5, 0.008 * scale))
        if l is None or r is None or r[0] - l[0] < 0.20:
            continue
        av = (l[1] - r[1]) / (r[0] - l[0])
        au = l[0] * av + l[1]
        ax = au * c + av * s
        ay = -au * s + av * c
        lower, upper = np.arctan(l[0]), np.arctan(r[0])
        half = (upper - lower) / 2
        axis = (upper + lower) / 2 + theta
        if not np.deg2rad(12) <= half <= np.deg2rad(75):
            continue
        if not -1.5 * w < ax < 2.5 * w or not -1.5 * h < ay < 2.5 * h:
            continue
        dx = xx - ax
        dy = yy - ay
        angle = np.arctan2(dx, dy) - axis
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        radius = np.hypot(dx, dy)
        angular_bin = np.floor((angle + half) / (2 * half) * 48).astype(int)
        central = (angular_bin >= 5) & (angular_bin < 43)
        outer = np.full(48, -np.inf)
        inner = np.full(48, np.inf)
        np.maximum.at(outer, angular_bin[central], radius[central])
        np.minimum.at(inner, angular_bin[central], radius[central])
        observed = np.isfinite(outer) & np.isfinite(inner)
        if observed.sum() < 25:
            continue
        outer_radius = float(np.quantile(outer[observed], 0.75))
        arc_support = float(
            (np.abs(outer[observed] - outer_radius) <= max(2.5, 0.015 * scale)).mean()
        )
        if outer_radius < 0.15 * scale or outer_radius > 3 * max(h, w):
            continue
        # A small explicit tolerance recovers the discretized physical boundary; not a fan prior.
        margin = max(1.0, 0.004 * scale)
        half_padded = half + np.arctan2(margin, outer_radius)
        outer_padded = outer_radius + margin
        gx, gy = grid_x - ax, grid_y - ay
        ga = (np.arctan2(gx, gy) - axis + np.pi) % (2 * np.pi) - np.pi
        gr = np.hypot(gx, gy)
        forward = gx * np.sin(axis) + gy * np.cos(axis)
        base = (np.abs(ga) <= half_padded) & (gr <= outer_padded)
        direction = ((np.rad2deg(axis) + 180) % 360) - 180
        family = "rotated_wedge" if abs(direction) > 12 else "standard_fan"
        common = {
            "apex_x": float(ax),
            "apex_y": float(ay),
            "axis_deg": float(direction),
            "half_angle_deg": float(np.rad2deg(half_padded)),
            "outer_radius": float(outer_padded),
        }
        metrics = {
            "flank_consensus": min(l[2], r[2]),
            "flank_span": min(l[3], r[3]),
            "arc_support": arc_support,
        }
        variants = [(family, base, {"near_field": "apex"})]
        near = (
            float(np.quantile((xx - ax) * np.sin(axis) + (yy - ay) * np.cos(axis), 0.005)) - margin
        )
        inner_radius = float(np.quantile(inner[observed], 0.15)) - margin
        if near > 0.08 * outer_radius:
            variants.append(
                (
                    "flat_top_fan",
                    base & (forward >= near),
                    {"near_field": "plane", "near_depth": near},
                )
            )
        if inner_radius > 0.08 * outer_radius:
            variants.append(
                (
                    "flat_top_fan",
                    base & (gr >= inner_radius),
                    {"near_field": "inner_arc", "inner_radius": inner_radius},
                )
            )
        for kind, mask, extra in variants:
            candidate = GeometryFit(kind, mask, {**common, **extra}, dict(metrics))
            candidates.append(_score(candidate, evidence, allowed, motion))
    box = cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32)))
    mask = np.zeros(evidence.shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(box).astype(np.int32)], 1)
    candidates.append(
        _score(
            GeometryFit("rectangle", mask.astype(bool), {"corners_xy": box.tolist()}),
            evidence,
            allowed,
            motion,
        )
    )
    ordered = sorted(candidates, key=lambda f: f.metrics["score"])
    reliable = [f for f in ordered if f.reliable]
    return (reliable[0] if reliable else None), [f.record() for f in ordered]
