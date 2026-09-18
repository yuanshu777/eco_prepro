"""Authoritative frame rates from dataset-level tables (the container header is not always right:
EchoNet-LVH AVIs say 50/63 fps while Stanford's MeasurementsList.csv says ~20/16 fps)."""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache(maxsize=16)
def load_fps_table(csv_path: str, key: str, fps: str) -> dict[str, float]:
    df = pd.read_csv(csv_path, usecols=[key, fps]).dropna()
    df = df.drop_duplicates(subset=[key])
    return {str(k): float(v) for k, v in zip(df[key], df[fps]) if math.isfinite(float(v)) and float(v) > 0}


def fps_from_table(dataset_cfg: dict, root: Path, path: str | Path) -> float | None:
    spec = dataset_cfg.get("fps_table")
    if not spec:
        return None
    table = load_fps_table(str(root / spec["csv"]), spec["key"], spec["fps"])
    return table.get(Path(path).stem)


def apply_fps_provenance(cine, dataset_cfg: dict, root: Path) -> None:
    """Keep the container/reader value even when a dataset table supplies acquisition FPS."""
    value = fps_from_table(dataset_cfg, root, cine.path)
    cine.meta.setdefault("fps_reader", cine.fps)
    cine.meta.setdefault("fps_container", cine.fps if cine.reader == "video" else None)
    cine.meta["fps_dataset_table"] = value
    if value is not None:
        cine.fps = value
        cine.meta["fps_source"] = "dataset_table"
        cine.meta["fps_table"] = dataset_cfg["fps_table"]["csv"]
        cine.meta["fps_table_sha256"] = hashlib.sha256((root / cine.meta["fps_table"]).read_bytes()).hexdigest()
    cine.meta.setdefault("fps_source", "unknown")
