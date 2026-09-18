"""Motion evidence followed by conservative ultrasound sector geometry.

The frozen Stage 1 detector remains available as `_detect_motion`; the public `detect_sector`
preserves its dispatch and adds allowed-area candidate search, geometry and temporal fit checks.
The baseline algorithm below is retained for comparison and conservative hull fallback.

Follows the Mayo Clinic recipe (Naser et al. 2024): find pixels that change over time, clean up with
morphology, keep the largest (central) moving region, and crop to it. Refinements:

1. The moving region only *locates* the sector. The returned mask is the convex hull of the union of
   that region and the connected "support" region (pixels lit in a fraction of frames) that overlaps
   it, so near-static parts of the sector (far field, pericardium) are kept.
2. Morphological opening with a kernel scaled to the image size removes speckle, cursors, ticks and
   text strokes; an additional opening with a *vertical line* cuts thin horizontal bands (ECG trace,
   timeline strip) that are otherwise merged into the sector because they sweep and therefore "move".
   Corners lost to the line opening are recovered by a bounded geodesic reconstruction.
3. Quality flags tell the caller when the result should not be trusted (low motion, tiny region,
   static input) and whether the input already looks pre-cropped (EchoNet-style).

Everything is classical image processing (OpenCV/SciPy); no learned model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

DARK_LEVEL = 6  # gray levels; at or below this a pixel counts as "unlit"
PROCESSING_STATUSES = {"SUCCESS", "PASSTHROUGH", "LOW_CONFIDENCE", "STATIC", "MULTI_REGION", "FAILED"}


@dataclass
class SectorResult:
    mask: np.ndarray                      # (H, W) bool - actual imaging mask, possibly nonconvex
    bbox: tuple[int, int, int, int]       # x0, y0, x1, y1 (exclusive)
    hull: np.ndarray                      # (N, 2) int32 polygon, xy
    status: str                           # one of PROCESSING_STATUSES
    stats: dict = field(default_factory=dict)
    maps: dict = field(default_factory=dict)  # std / mean / lit / motion / support (figures, debugging)
    flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in {"SUCCESS", "PASSTHROUGH"}

    @property
    def precropped(self) -> bool:
        return bool(self.stats.get("precropped", False))


def _odd(k: float) -> int:
    k = round(k)
    return max(3, k if k % 2 == 1 else k + 1)


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _vline(h: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (1, h))


def to_gray(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 3:
        return frames
    return np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])


def temporal_maps(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """gray: (T, H, W) uint8 -> (std, mean, lit_fraction), each (H, W) float32."""
    g = gray.astype(np.float32)
    std = g.std(axis=0)
    mean = g.mean(axis=0)
    lit = (gray > DARK_LEVEL).mean(axis=0).astype(np.float32)
    return std, mean, lit


def _open(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, se).astype(bool)


def _propagate(seed: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Full geodesic reconstruction of `seed` inside `base` (8-connected)."""
    seed = seed & base
    if not seed.any():
        return seed
    return ndi.binary_propagation(seed, mask=base, structure=np.ones((3, 3), bool))


def _cut_edge_bands(region: np.ndarray, core: np.ndarray, max_frac: float = 0.2, jump: float = 1.3,
                    min_width_frac: float = 0.3, max_core_frac: float = 0.5,
                    margin_frac: float = 0.08) -> tuple[np.ndarray, list[str]]:
    """Remove a wide, thin block of rows glued to the bottom or top of the region (ECG trace,
    timeline strip). Signature: below the region's widest row the width shrinks along the fan's arc,
    then jumps back up (>= jump x the minimum so far) to >= min_width_frac of the image width within
    the last max_frac of the image height, and the block is mostly *not* core (its pixels are lit only
    briefly as the trace sweeps; bright anatomy such as pericardium is core and is left alone).
    Briefly-lit pixels in the margin above a detected band (trace spikes) are dropped as well.
    Symmetric rule for the top."""
    H, W = region.shape
    w = region.sum(axis=1)
    rows = np.where(w > 0)[0]
    if rows.size < 8:
        return region, []
    y0, y1, ymax = int(rows[0]), int(rows[-1]), int(np.argmax(w))
    limit, wmin, margin = int(max_frac * H), min_width_frac * W, int(margin_frac * H)
    out, cut = region.copy(), []

    def core_frac(sl: slice) -> float:
        blk = region[sl]
        return float(core[sl][blk].mean()) if blk.any() else 1.0

    m = float(w[ymax])
    for y in range(ymax + 1, y1 + 1):
        if w[y] >= jump * m and w[y] >= wmin and (y1 - y + 1) <= limit:
            if core_frac(slice(y, y1 + 1)) < max_core_frac:
                out[y:] = False
                lo = max(y0, y - margin)
                out[lo:y] &= core[lo:y]
                cut.append("band_bottom")
            break
        m = min(m, float(w[y]))
    m = float(w[ymax])
    for y in range(ymax - 1, y0 - 1, -1):
        if w[y] >= jump * m and w[y] >= wmin and (y - y0 + 1) <= limit:
            if core_frac(slice(y0, y + 1)) < max_core_frac:
                out[: y + 1] = False
                hi = min(y1 + 1, y + 1 + margin)
                out[y + 1:hi] &= core[y + 1:hi]
                cut.append("band_top")
            break
        m = min(m, float(w[y]))
    return out, cut


def _pick_component(binary: np.ndarray, center_weight: float = 0.5):
    """Largest connected component with a mild preference for central ones. Returns (mask, area)."""
    n, lab, stats, cents = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None, 0
    H, W = binary.shape
    half_diag = 0.5 * float(np.hypot(H, W))
    best, best_score = None, -1.0
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        cx, cy = cents[i]
        d = float(np.hypot(cx - W / 2, cy - H / 2)) / half_diag   # 0 = centre, 1 = corner
        score = area * (1.0 - center_weight * d)
        if score > best_score:
            best, best_score = i, score
    return lab == best, int(stats[best, cv2.CC_STAT_AREA])


def _hull_mask(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(region, dtype=bool), np.zeros((0, 2), np.int32)
    pts = np.concatenate(contours, axis=0)
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.int32)
    mask = np.zeros(region.shape, np.uint8)
    cv2.fillPoly(mask, [hull.reshape(-1, 1, 2)], 1)
    return mask.astype(bool), hull


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fan_geometry(mask: np.ndarray) -> dict:
    """Rough sector descriptors from the mask: apex estimate from the left/right edges of the upper
    part, opening angle between those edges, and widths. Purely descriptive (for the audit)."""
    x0, y0, x1, y1 = _bbox(mask)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return {}
    rows = np.arange(y0, y1)
    left = np.array([np.argmax(mask[r]) for r in rows], dtype=np.float32)
    right = np.array([mask.shape[1] - 1 - np.argmax(mask[r][::-1]) for r in rows], dtype=np.float32)
    h = y1 - y0
    sel = slice(int(0.10 * h), int(0.60 * h))  # straight flanks, away from apex tip and bottom arc
    out = {"top_width": float(right[0] - left[0] + 1), "bottom_width": float(right[-1] - left[-1] + 1),
           "max_width": float((right - left + 1).max()), "height": float(h)}
    if sel.stop - sel.start < 5:
        return out
    yy = rows[sel].astype(np.float32)
    try:
        lp = np.polyfit(yy, left[sel], 1)   # x = a*y + b
        rp = np.polyfit(yy, right[sel], 1)
    except (np.linalg.LinAlgError, ValueError):
        return out
    out["opening_angle_deg"] = float(np.degrees(np.arctan(rp[0])) - np.degrees(np.arctan(lp[0])))
    if abs(rp[0] - lp[0]) > 1e-6:
        ay = (lp[1] - rp[1]) / (rp[0] - lp[0])
        ax = lp[0] * ay + lp[1]
        out["apex_x"], out["apex_y"] = float(ax), float(ay)
        out["apex_inside_image"] = bool(0 <= ay < mask.shape[0] and 0 <= ax < mask.shape[1])
    return out


def _detect_motion(frames: np.ndarray, n_sample: int = 64, min_std: float = 3.0,
                  min_area_frac: float = 0.04, kernel_frac: float = 0.012, band_frac: float = 0.08,
                  core_lit: float = 0.5, ext_lit: float = 0.02, static: bool = False,
                  keep_maps: bool = True, already_standardized: bool = False,
                  exclusion_mask: np.ndarray | None = None,
                  allowed_mask: np.ndarray | None = None) -> SectorResult:
    """Detect the imaging region in a cine.

    frames:        (T, H, W, 3) uint8 RGB or (T, H, W) uint8.
    n_sample:      frames used for the temporal statistics (evenly spaced over the cine).
    min_std:       minimum temporal std (gray levels) for a pixel to count as moving.
    kernel_frac:   opening kernel = kernel_frac * min(H, W), odd, >= 3.
    band_frac:     vertical-line opening height = band_frac * H, used only when choosing the core
                   component, so thin horizontal strips cannot bridge the sector to other blobs.
    core_lit:      core support = pixels lit (> DARK_LEVEL) in at least this fraction of frames.
    ext_lit:       extended support = pixels lit in at least this (small) fraction of frames.
    static:        force the intensity-only path (single images).

    Selection rule: the imaging region is the connected *core* lit region (lit most of the time) that
    contains the most temporal motion (mild preference for central regions). It is then grown into
    the extended support (lit at least occasionally) so dim far-field and fan corners are kept, while
    ECG traces and timeline strips glued to the bottom (or top) of the sector are removed by shape:
    a wide block of rows at the edge of the region that is wider than the fan's arc above it (see
    `_cut_edge_bands`). Stills fall back to the largest central lit region.
    """
    if len(frames) == 0:
        raise ValueError("empty cine")
    T, H, W = frames.shape[:3]
    idx = np.unique(np.linspace(0, T - 1, min(T, n_sample)).round().astype(int))
    # Sample before RGB conversion: long cines must not allocate a full grayscale copy.
    gray = to_gray(frames[idx])
    k = _odd(kernel_frac * min(H, W))
    hv = _odd(band_frac * H)
    std, mean, lit = temporal_maps(gray)
    stats: dict = {"n_frames_used": len(idx), "kernel": k, "band_kernel": hv,
                   "std_p99": float(np.percentile(std, 99)), "std_max": float(std.max())}
    flags: list[str] = []
    use_motion = len(idx) >= 2 and not static

    # --- motion map -------------------------------------------------------------------------
    thr = None
    motion_cut = np.zeros((H, W), bool)
    if use_motion:
        s8 = np.clip(std / max(float(std.max()), 1e-6) * 255.0, 0, 255).astype(np.uint8)
        otsu, _ = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(min_std, min(otsu / 255.0 * float(std.max()), 0.2 * stats["std_p99"]))
        motion_cut = _open(std > thr, _ellipse(k))             # speckle, cursors, ticks, strokes
        stats.update({"std_threshold": float(thr), "motion_area_frac": float(motion_cut.mean())})

    # --- lit support: core (lit most of the time) and extent (lit at least occasionally) ---------
    if len(idx) > 1:
        core = _open(lit >= core_lit, _ellipse(k))
        ext = _open(lit >= ext_lit, _ellipse(k))
    else:
        core = ext = _open(gray[0] > DARK_LEVEL, _ellipse(k))
    if allowed_mask is not None:
        core &= allowed_mask
        ext &= allowed_mask
        motion_cut &= allowed_mask
    support_cut = _open(core, _vline(hv))   # thin strips cannot bridge components during selection
    support = core

    # Early pass-through, before component growth, band removal, hull or crop.
    # A caller may supply a verified source policy; generic evidence is deliberately stricter.
    bb = _bbox(ext)
    touches = sum((bb[0] <= .03 * W, bb[1] <= .03 * H,
                   bb[2] >= .97 * W, bb[3] >= .97 * H))
    _nc, lc, cs, _ = cv2.connectedComponentsWithStats(ext.astype(np.uint8), connectivity=8)
    areas = cs[1:, cv2.CC_STAT_AREA]
    dominant = int(np.argmax(areas) + 1) if areas.size else 0
    # Count separate extended-support panels, including a static reference next to a cine.
    # Core components can split one fan into several pieces, so they are unsuitable for this QC.
    largest = float(areas.max()) if areas.size else 0.0
    panel = ((areas >= max(.01 * H * W, .10 * largest))
             & (cs[1:, cv2.CC_STAT_WIDTH] >= .08 * W)
             & (cs[1:, cv2.CC_STAT_HEIGHT] >= .10 * H))
    stats["candidate_regions"] = int(panel.sum())
    if panel.sum() >= 2:
        flags.append("multiple_regions")
    coverage = float(areas.max() / max(areas.sum(), 1)) if areas.size else 0.0
    outside_lit = float((lit[(lc != dominant) & (lit > 0)]).sum() / max(lit.sum(), 1))
    generic_pre = panel.sum() == 1 and touches >= 3 and ext.mean() >= .50 and coverage > .995 and outside_lit < .002
    if allowed_mask is None and use_motion and stats["std_p99"] >= min_std and (already_standardized or generic_pre):
        mask = np.ones((H, W), bool)
        hull = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], np.int32)
        stats.update({"precropped": True, "passthrough_basis": "source_policy" if already_standardized else "image_evidence",
                      "area_frac": 1.0, "solidity": 1.0, "motion_share_in_region": 1.0,
                      "support_area_frac": float(ext.mean()), "bbox": [0, 0, W, H]})
        result = SectorResult(mask, (0, 0, W, H), hull, "PASSTHROUGH", stats,
                              {"std": std, "lit": lit} if keep_maps else {}, ["precropped"])
        return quality_guards(result, frames[idx], exclusion_mask)

    # --- choose the region -------------------------------------------------------------------
    n, lab, cstats, cents = cv2.connectedComponentsWithStats(support_cut.astype(np.uint8), connectivity=8)
    region = None
    if n > 1:
        half_diag = 0.5 * float(np.hypot(H, W))
        centrality = np.array([1.0 - 0.5 * float(np.hypot(cx - W / 2, cy - H / 2)) / half_diag for cx, cy in cents])
        motion_in = np.bincount(lab[motion_cut], minlength=n).astype(np.float64) if use_motion else np.zeros(n)
        motion_in[0] = 0.0
        if use_motion and motion_in.sum() > 0:
            j = int(np.argmax(motion_in * centrality))
            stats["motion_share_in_region"] = float(motion_in[j] / motion_in.sum())
            if stats["motion_share_in_region"] < 0.5:
                flags.append("motion_split")
        else:
            area = cstats[:, cv2.CC_STAT_AREA].astype(np.float64)
            area[0] = 0.0
            j = int(np.argmax(area * centrality))
            flags.append("static_only" if not use_motion else "low_motion")
        region = _propagate(lab == j, core)              # whole core component (corners included)
        region = _propagate(region, ext | region)        # grow into dim far field / edges
        region = ndi.binary_fill_holes(region)
        rx0, ry0, rx1, ry1 = _bbox(region)
        looks_precropped = (sum([rx0 <= 0.03 * W, ry0 <= 0.03 * H, rx1 >= 0.97 * W, ry1 >= 0.97 * H]) >= 3
                            and region.mean() >= 0.40)
        if not looks_precropped:                        # pre-cropped inputs carry no overlays
            region, cut = _cut_edge_bands(region, core)
            flags.extend(cut)
            if cut:
                region = ndi.binary_fill_holes(region)
    if region is not None and allowed_mask is not None:
        region &= allowed_mask
    motion_comp = (motion_cut & region) if region is not None else np.zeros((H, W), bool)

    if region is None or not region.any():
        flags.append("no_region")
        empty = np.zeros((H, W), bool)
        stats.update(area_frac=0.0, solidity=0.0, motion_share_in_region=0.0)
        return quality_guards(SectorResult(empty, (0, 0, 0, 0), np.zeros((0, 2), np.int32), "FAILED", stats,
                            {"std": std, "lit": lit} if keep_maps else {}, flags), frames[idx], exclusion_mask)

    mask, hull = _hull_mask(region)
    if allowed_mask is not None:
        mask &= allowed_mask
    bbox = _bbox(mask)
    x0, y0, x1, y1 = bbox
    area_frac = float(mask.mean())
    tol_x, tol_y = 0.03 * W, 0.03 * H
    touches = {"left": x0 <= tol_x, "top": y0 <= tol_y, "right": x1 >= W - tol_x, "bottom": y1 >= H - tol_y}
    stats.update({
        "area_frac": area_frac,
        "solidity": float(region.sum() / max(mask.sum(), 1)),
        "bbox": [x0, y0, x1, y1],
        "bbox_w": x1 - x0, "bbox_h": y1 - y0,
        **{f"touches_{k2}": v for k2, v in touches.items()},
        "precropped": False,  # only the early branch can declare PASSTHROUGH
        "lit_inside": float(lit[mask].mean()),
        "lit_outside": float(lit[~mask].mean()) if (~mask).any() else 0.0,
    })
    if thr is not None:
        stats["motion_inside"] = float((std[mask] > thr).mean())
        stats["motion_outside"] = float((std[~mask] > thr).mean()) if (~mask).any() else 0.0
    stats.update({f"fan_{k2}": v for k2, v in fan_geometry(mask).items()})

    if area_frac < min_area_frac:
        flags.append("small_region")
    if use_motion and stats.get("motion_inside", 0.0) < 0.05 and "low_motion" not in flags:
        flags.append("low_motion")
    if stats["precropped"]:
        flags.append("precropped")
    status = "SUCCESS"
    if "small_region" in flags or "low_motion" in flags:
        status = "LOW_CONFIDENCE"
    if not use_motion:
        status = "STATIC"
    if "multiple_regions" in flags:
        status = "MULTI_REGION"
    maps = {"std": std, "mean": mean, "lit": lit, "support": support, "ext": ext,
            "motion": motion_comp, "candidate": region} if keep_maps else {}
    return quality_guards(SectorResult(mask, bbox, hull, status, stats, maps, flags),
                          frames[idx], exclusion_mask)


def quality_guards(result: SectorResult, frames: np.ndarray,
                   exclusion_mask: np.ndarray | None = None) -> SectorResult:
    """Record inexpensive diagnostics; warnings never discard a file.

    Thresholds are engineering review triggers, not calibrated quality probabilities.
    Exclusion zones are diagnostics only in Stage 1 (no template-based masking).
    """
    mask, stats, flags = result.mask, result.stats, result.flags
    if frames.ndim == 4:
        color = np.ptp(frames.astype(np.int16), axis=-1) > 24
    else:
        color = np.zeros(frames.shape, bool)
    stats["color_fraction_inside"] = float(color[:, mask].mean()) if mask.any() else 0.0
    stats["color_fraction_outside"] = float(color[:, ~mask].mean()) if (~mask).any() else 0.0
    stats["overlay_contact_frac"] = None
    if exclusion_mask is not None:
        if exclusion_mask.shape != mask.shape:
            raise ValueError("exclusion mask shape must match frames")
        stats["overlay_contact_frac"] = float((mask & exclusion_mask).sum() / max(mask.sum(), 1))
        if stats["overlay_contact_frac"] > .005:
            flags.append("overlay_contact")
    if stats.get("area_frac", 0) < .15:
        flags.append("small_area")
    if stats.get("solidity", 1) < .88:
        flags.append("low_solidity")
    if stats.get("motion_share_in_region", 1) < .75:
        flags.append("motion_not_captured")
    if stats["color_fraction_outside"] > .01:
        # Often the expected removal of a color bar/ECG; informational, not a failure alone.
        flags.append("color_outside_region")
    if result.status == "SUCCESS" and any(f in flags for f in
            ("small_area", "low_solidity", "motion_not_captured", "overlay_contact")):
        result.status = "LOW_CONFIDENCE"
    if result.status in {"STATIC", "MULTI_REGION", "LOW_CONFIDENCE", "FAILED"}:
        flags.append("manual_review_required")
    result.flags = list(dict.fromkeys(flags))
    return result


def detect_sector(
    frames: np.ndarray,
    n_sample: int = 64,
    min_std: float = 3.0,
    min_area_frac: float = 0.04,
    kernel_frac: float = 0.012,
    band_frac: float = 0.08,
    core_lit: float = 0.5,
    ext_lit: float = 0.02,
    static: bool = False,
    keep_maps: bool = True,
    already_standardized: bool = False,
    exclusion_mask: np.ndarray | None = None,
    refine_geometry: bool = True,
    verify_stability: bool = True,
) -> SectorResult:
    """Stage 2 sector-only extension; Stage 1 dispatch/status and preservation contracts remain.

    Pass-through, static and multi-panel inputs are returned before any geometry work. Motion
    locates a connected candidate; support boundaries constrain geometry. Fit diagnostics live
    in SectorResult.stats/maps, so readers, canonical Cine and export schemas need no changes.
    Set refine_geometry=False to reproduce the frozen Stage 1 detector for regression comparisons.
    verify_stability=False is for diagnostics: production fits must also agree across
    interleaved samples at two evidence densities. One mask is used for the entire cine.
    """
    from hashlib import sha256

    from echoprep.sector.geometry import fit_geometry
    from echoprep.sector.layouts import allowed_area

    options = {
        "n_sample": n_sample,
        "min_std": min_std,
        "min_area_frac": min_area_frac,
        "kernel_frac": kernel_frac,
        "band_frac": band_frac,
        "core_lit": core_lit,
        "ext_lit": ext_lit,
        "static": static,
        "keep_maps": True,
        "already_standardized": already_standardized,
        "exclusion_mask": exclusion_mask,
    }
    baseline = _detect_motion(frames, **options)
    if not refine_geometry:
        if not keep_maps:
            baseline.maps = {}
        return baseline
    baseline.stats["stage2_baseline_status"] = baseline.status
    baseline.stats["stage2_method"] = "retained_stage1"
    if (
        baseline.status in {"PASSTHROUGH", "STATIC", "MULTI_REGION", "FAILED"}
        or "low_motion" in baseline.flags
    ):
        baseline.stats["stage2_skip_reason"] = baseline.status
        if not keep_maps:
            baseline.maps = {}
        return baseline
    idx = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), n_sample)).round().astype(int))
    sampled = frames[idx]
    allowed, layout = allowed_area(sampled, exclusion_mask)
    # Candidate search and growth happen inside allowed pixels. Do not mutate source RGB frames.
    candidate = _detect_motion(sampled, **options, allowed_mask=allowed)
    base_region = baseline.maps.get("candidate", baseline.mask)
    protected = ndi.binary_erosion(
        base_region & baseline.maps.get("support", base_region), iterations=2
    )
    lost = float((protected & ~allowed).sum() / max(int(protected.sum()), 1))
    baseline.stats.update(
        stage2_layout=layout,
        stage2_excluded_core_fraction=lost,
        stage2_allowed_sha256=sha256(np.packbits(allowed).tobytes()).hexdigest(),
    )
    baseline.maps["allowed"] = allowed
    baseline.maps["baseline_mask"] = baseline.mask.copy()
    if (
        lost > 0.015
        or candidate.status in {"FAILED", "STATIC", "MULTI_REGION"}
        or "low_motion" in candidate.flags
    ):
        baseline.status = "LOW_CONFIDENCE"
        baseline.flags = list(
            dict.fromkeys(
                baseline.flags + ["unsafe_exclusion_or_candidate", "manual_review_required"]
            )
        )
        baseline.stats["stage2_skip_reason"] = (
            "exclusion_conflict" if lost > 0.015 else "ambiguous_candidate"
        )
        if not keep_maps:
            baseline.maps = {}
        return baseline
    evidence = candidate.maps.get("candidate", candidate.mask) & allowed
    moving = candidate.maps["std"] > candidate.stats.get("std_threshold", min_std)
    fitted, records = fit_geometry(evidence, allowed, moving)
    baseline.stats["stage2_candidates"] = records
    baseline.maps.update(candidate=evidence, motion=moving & evidence)

    def fallback_hull():
        # Keep only a previously trustworthy hull; never promote a weak baseline by fallback.
        mask = baseline.mask & allowed
        baseline.stats["stage2_method"] = "fallback_hull"
        baseline.flags = list(dict.fromkeys(baseline.flags + ["geometry_fallback_hull"]))
        if baseline.status == "SUCCESS":
            baseline.mask = mask
            baseline.bbox = _bbox(mask)
            convex, baseline.hull = _hull_mask(mask)
            x0, y0, x1, y1 = baseline.bbox
            baseline.stats.update(
                area_frac=float(mask.mean()),
                solidity=float(mask.sum() / max(int(convex.sum()), 1)),
                bbox=list(baseline.bbox),
                bbox_w=x1 - x0,
                bbox_h=y1 - y0,
                touches_left=x0 <= 0.03 * mask.shape[1],
                touches_right=x1 >= 0.97 * mask.shape[1],
                touches_top=y0 <= 0.03 * mask.shape[0],
                touches_bottom=y1 >= 0.97 * mask.shape[0],
            )
            baseline.stats.update({f"fan_{k}": v for k, v in fan_geometry(mask).items()})
            quality_guards(baseline, sampled, exclusion_mask)
        else:
            baseline.flags = list(
                dict.fromkeys(baseline.flags + ["geometry_unreliable", "manual_review_required"])
            )
        baseline.maps["fitted"] = mask
        if not keep_maps:
            baseline.maps = {}
        return baseline

    if fitted is None:
        return fallback_hull()
    mask = fitted.mask
    baseline.stats.update(
        stage2_proposed_family=fitted.family,
        stage2_proposed_parameters=fitted.parameters,
        stage2_proposed_metrics=fitted.metrics,
    )
    core_loss = float((protected & ~mask).sum() / max(int(protected.sum()), 1))
    baseline.stats["stage2_fit_core_loss"] = core_loss
    if core_loss > 0.01:
        baseline.status = "LOW_CONFIDENCE"
        baseline.flags = list(
            dict.fromkeys(baseline.flags + ["geometry_core_loss", "manual_review_required"])
        )
        baseline.stats["stage2_skip_reason"] = "fit_loses_baseline_support"
        baseline.maps["fitted"] = mask
        if not keep_maps:
            baseline.maps = {}
        return baseline
    if verify_stability:
        checks = []
        parts = [sampled[::2], sampled[1::2]]
        if len(frames) > n_sample:
            # Re-sample both original parity streams as well: splitting the original
            # 64-frame sample alone can miss sensitivity to temporal sampling phase.
            parts.extend([frames[::2], frames[1::2]])
        for part in parts:
            if len(part) < 4:
                continue
            check = detect_sector(part, **{**options, "keep_maps": False}, verify_stability=False)
            checks.append(check)
        stable = len(checks) == len(parts)
        overlap = 0.0
        if stable:
            overlap = min(
                float((a.mask & b.mask).sum() / max(int((a.mask | b.mask).sum()), 1))
                for i, a in enumerate(checks)
                for b in checks[i + 1 :]
            )
            stable = overlap >= 0.98 and all(
                check.status == "SUCCESS"
                and check.stats.get("stage2_method") == "geometry_fit"
                and float((mask & check.mask).sum() / max(int((mask | check.mask).sum()), 1))
                >= 0.97
                for check in checks
            )
        baseline.stats.update(
            stage2_stability_iou=overlap,
            stage2_stability_statuses=[check.status for check in checks],
            stage2_stability_methods=[check.stats.get("stage2_method") for check in checks],
            stage2_stability_passed=stable,
        )
        if not stable:
            baseline.flags = list(dict.fromkeys(baseline.flags + ["geometry_temporal_instability"]))
            baseline.stats["stage2_skip_reason"] = "unstable_geometry"
            baseline.maps["rejected_fit"] = mask
            return fallback_hull()
    convex, hull = _hull_mask(mask)
    resolved = {
        "low_solidity",
        "small_area",
        "small_region",
        "overlay_contact",
        "motion_not_captured",
        "manual_review_required",
    }
    flags = [f for f in candidate.flags if f not in resolved]
    stats = {**baseline.stats, **candidate.stats}
    x0, y0, x1, y1 = _bbox(mask)
    lit = candidate.maps["lit"]
    stats.update(
        stage2_seed_motion_share=candidate.stats.get("motion_share_in_region"),
        motion_share_in_region=float((moving & mask).sum() / max(int((moving & allowed).sum()), 1)),
        stage2_method="geometry_fit",
        stage2_family=fitted.family,
        stage2_parameters=fitted.parameters,
        stage2_metrics=fitted.metrics,
        stage2_mask_sha256=sha256(np.packbits(mask).tobytes()).hexdigest(),
        area_frac=float(mask.mean()),
        solidity=float(mask.sum() / max(int(convex.sum()), 1)),
        bbox=list(_bbox(mask)),
        bbox_w=x1 - x0,
        bbox_h=y1 - y0,
        touches_left=x0 <= 0.03 * mask.shape[1],
        touches_right=x1 >= 0.97 * mask.shape[1],
        touches_top=y0 <= 0.03 * mask.shape[0],
        touches_bottom=y1 >= 0.97 * mask.shape[0],
        lit_inside=float(lit[mask].mean()),
        lit_outside=float(lit[~mask].mean()) if (~mask).any() else 0.0,
        motion_inside=float(moving[mask].mean()),
        motion_outside=float(moving[~mask].mean()) if (~mask).any() else 0.0,
        precropped=False,
    )
    stats.update({f"geometry_{k}": v for k, v in fitted.metrics.items()})
    stats.update({f"fan_{k}": v for k, v in fan_geometry(mask).items()})
    result = SectorResult(mask, _bbox(mask), hull, "SUCCESS", stats, baseline.maps, flags)
    result.maps["fitted"] = mask
    result = quality_guards(result, sampled, exclusion_mask)
    if not keep_maps:
        result.maps = {}
    return result
