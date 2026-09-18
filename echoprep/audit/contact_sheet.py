"""Figures for visual inspection: per-case panel (first/middle/last frame, motion map, mask, output)
and an overview grid of many cases."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from echoprep.cine import Cine
from echoprep.sector.motion import SectorResult


def _overlay(frame: np.ndarray, sector: SectorResult, color=(0, 255, 0)) -> np.ndarray:
    img = frame.copy()
    if sector.hull.shape[0] >= 3:
        cv2.polylines(img, [sector.hull.reshape(-1, 1, 2)], True, color, max(1, img.shape[1] // 300))
    x0, y0, x1, y1 = sector.bbox
    if x1 > x0 and y1 > y0:
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (255, 200, 0), max(1, img.shape[1] // 400))
    return img


def render_case(cine: Cine, sector: SectorResult, out_png: str | Path, std_cine: Cine | None = None,
                title: str = "") -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    T = cine.n_frames
    ids = [0, T // 2, T - 1]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for ax, i, name in zip(axes[0], ids, ["first", "middle", "last"]):
        ax.imshow(cine.frames[i])
        ax.set_title(f"{name} (frame {i})")
    if std_cine is not None:
        for ax, i, name in zip(axes[1], ids, ["first", "middle", "last"]):
            ax.imshow(std_cine.frames[i])
            ax.set_title(f"after: {name} (frame {i})")
    std = sector.maps.get("std")
    ax = axes[2][0]
    if std is not None:
        ax.imshow(std, cmap="magma")
        if sector.hull.shape[0] >= 3:
            h = np.vstack([sector.hull, sector.hull[:1]])
            ax.plot(h[:, 0], h[:, 1], "-", color="lime", lw=1.2)
        ax.set_title("temporal std + hull")
    else:
        ax.axis("off")
    axes[2][1].imshow(_overlay(cine.frames[T // 2], sector))
    axes[2][1].set_title(f"region: {sector.status} (area {sector.stats.get('area_frac', 0):.2f})")
    if std_cine is not None:
        axes[2][2].imshow(std_cine.frames[std_cine.n_frames // 2])
        axes[2][2].set_title(f"standardized {std_cine.width}x{std_cine.height}")
    else:
        axes[2][2].axis("off")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fps = f"{cine.fps:.1f} fps" if cine.fps else "fps unknown"
    fig.suptitle(f"{title}  |  {cine.width}x{cine.height}, {T} frames, {fps}, colour {cine.color_fraction():.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def render_overview(items: list[tuple[str, np.ndarray]], out_png: str | Path, cols: int = 4,
                    tile: int = 300) -> Path:
    """items: (caption, RGB image). Thumbnails on a grid with captions."""
    from PIL import Image, ImageDraw

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    rows = (len(items) + cols - 1) // cols
    pad = 34
    tile_height = tile // 2 + 25
    canvas = Image.new("RGB", (cols * tile, rows * (tile_height + pad)), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    for n, (cap, img) in enumerate(items):
        im = Image.fromarray(img)
        im.thumbnail((tile - 6, tile_height - 6))
        x, y = (n % cols) * tile, (n // cols) * (tile_height + pad)
        canvas.paste(im, (x + 3, y + pad + 3))
        d.text((x + 4, y + 4), cap[:60], fill=(255, 230, 0))
    canvas.save(out_png)
    return out_png


def before_after(before: np.ndarray, after: np.ndarray, size: int = 320) -> np.ndarray:
    """Aspect-preserving before | after, with independent black letterboxes."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (size * 2, size + 22), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, (frame, name) in enumerate(((before, "BEFORE"), (after, "AFTER"))):
        im = Image.fromarray(frame)
        im.thumbnail((size, size))
        canvas.paste(im, (index*size+(size-im.width)//2, 22+(size-im.height)//2))
        draw.text((index*size+6, 4), name, fill="white")
    return np.asarray(canvas)
