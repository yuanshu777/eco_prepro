#!/usr/bin/env python
"""Outer-join run results by stable case ID, optionally migrating legacy IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATUS_MAP = {
    "ok": "SUCCESS",
    "static_only": "STATIC",
    "low_motion": "LOW_CONFIDENCE",
    "small_region": "LOW_CONFIDENCE",
    "no_region": "FAILED",
}


def compare(before: pd.DataFrame, after: pd.DataFrame, mapping: dict | None = None) -> pd.DataFrame:
    before, after = before.copy(), after.copy()
    for df in (before, after):
        if mapping:
            df["id"] = df["id"].map(lambda x: mapping.get(x, x))
        if df["id"].duplicated().any():
            raise ValueError("Duplicate case IDs in comparison input")
        df["status"] = df["status"].replace(STATUS_MAP)
    joined = before.merge(
        after, on="id", how="outer", suffixes=("_before", "_after"), indicator=True
    )
    joined["presence"] = (
        joined["_merge"]
        .astype(str)
        .map({"both": "both", "left_only": "removed", "right_only": "added"})
    )
    joined["status_changed"] = (joined["presence"] == "both") & (
        joined["status_before"] != joined["status_after"]
    )
    for field in ("out_h", "out_w", "sector.area_frac", "sector.solidity", "fps", "n_frames"):
        if f"{field}_before" in joined and f"{field}_after" in joined:
            joined[f"{field}_delta"] = joined[f"{field}_after"] - joined[f"{field}_before"]
    # A changed status is a review trigger, not proof of a quality regression.
    return joined.drop(columns="_merge")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--id-map", default=str(ROOT / "configs/audit_id_map.csv"))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    mapping = (
        pd.read_csv(a.id_map).set_index("legacy_id")["id"].to_dict()
        if Path(a.id_map).exists()
        else {}
    )
    result = compare(pd.read_csv(a.before), pd.read_csv(a.after), mapping)
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    print(result.presence.value_counts().to_string())
    print(
        f"{result.status_changed.sum()} status changes; inspect QC/review CSVs to judge correctness"
    )


if __name__ == "__main__":
    main()
