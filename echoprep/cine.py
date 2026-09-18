"""Canonical in-memory representation shared by every reader and every downstream step."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Cine:
    """A decoded echo cine.

    frames: (T, H, W, 3) uint8, RGB. Grayscale sources are stored as three identical channels so
            colour Doppler and grayscale go through the same code path.
    fps:    native frame rate if known, else None (never resampled here).
    """

    frames: np.ndarray
    fps: float | None
    path: str
    dataset: str = ""
    reader: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        f = np.asarray(self.frames)
        if f.ndim == 2:
            f = f[None, :, :, None]
        elif f.ndim == 3:
            # (T,H,W) grayscale video, or (H,W,3) single colour frame
            f = f[None] if f.shape[-1] == 3 else f[..., None]
        if f.shape[-1] == 1:
            f = np.repeat(f, 3, axis=-1)
        if f.shape[-1] != 3:
            raise ValueError(f"expected 1 or 3 channels, got shape {f.shape}")
        if f.dtype != np.uint8:
            raise ValueError(f"frames must be uint8, got {f.dtype}")
        self.frames = np.ascontiguousarray(f)

    # --- basic properties -------------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def height(self) -> int:
        return int(self.frames.shape[1])

    @property
    def width(self) -> int:
        return int(self.frames.shape[2])

    @property
    def duration_s(self) -> float | None:
        return None if not self.fps else self.n_frames / float(self.fps)

    def sample_indices(self, n: int) -> np.ndarray:
        """Evenly spaced frame indices (at most n, always includes first and last)."""
        n = max(1, min(n, self.n_frames))
        return np.unique(np.linspace(0, self.n_frames - 1, n).round().astype(int))

    def gray(self, indices: np.ndarray | None = None) -> np.ndarray:
        """(T', H, W) uint8 luma."""
        fr = self.frames if indices is None else self.frames[indices]
        return np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in fr])

    def color_fraction(self, max_frames: int = 16, spread: int = 24) -> float:
        """Fraction of pixels whose RGB channels differ by more than `spread` (colour Doppler,
        coloured overlays). Grayscale cines give ~0."""
        fr = self.frames[self.sample_indices(max_frames)].astype(np.int16)
        s = fr.max(axis=-1) - fr.min(axis=-1)
        return float((s > spread).mean())

    def summary(self) -> dict:
        return {
            "path": self.path,
            "dataset": self.dataset,
            "reader": self.reader,
            "n_frames": self.n_frames,
            "height": self.height,
            "width": self.width,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "color_fraction": round(self.color_fraction(), 5),
            **{f"meta.{k}": v for k, v in self.meta.items() if isinstance(v, (str, int, float, bool)) or v is None},
        }
