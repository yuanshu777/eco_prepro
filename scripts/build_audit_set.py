#!/usr/bin/env python
"""Materialize the frozen manifest at dataset roots; never resample or renumber cases."""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from echoprep.audit.derived import materialize_derived
from echoprep.audit.identity import content_sha256, resolve_case_path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=str(ROOT / "configs/audit_manifest.csv"))
    p.add_argument("--config", default=str(ROOT / "configs/datasets.yaml"))
    p.add_argument("--out", default=str(ROOT / "data/audit_set"))
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())["datasets"]
    rows = pd.read_csv(a.manifest, keep_default_na=False).to_dict("records")
    materialize_derived(rows, cfg, ROOT)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = resolve_case_path(row, cfg, ROOT)
        if content_sha256(path) != row["content_sha256"]:
            raise ValueError(f"Input changed for {row['id']}; create a new manifest version")
        row["path"] = str(path.resolve())
        link = out / row["id"]
        if not link.exists():
            link.symlink_to(path.resolve())
    pd.DataFrame(rows).to_csv(out / "audit_manifest.csv", index=False)
    print(f"{len(rows)} frozen cases verified and materialized under {out}")


if __name__ == "__main__":
    main()
