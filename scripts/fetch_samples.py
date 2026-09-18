#!/usr/bin/env python
"""Find local samples, otherwise fetch only configured small public files. Never fetch full archives."""

import argparse
import json
from pathlib import Path

import yaml

from echoprep.acquisition import acquire_samples

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/datasets.yaml"))
    p.add_argument("--all", action="store_true", help="Inspect every configured source (default)")
    p.add_argument("--only", help="Comma-separated source names")
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--limit", type=int, choices=range(1, 11), default=3)
    p.add_argument("--out", default=str(ROOT / "outputs/acquisition.json"))
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())["datasets"]
    selected = a.only.split(",") if a.only else list(cfg)
    reports = []
    for name in selected:
        report = acquire_samples(name, cfg[name], ROOT, download=not a.local_only, limit=a.limit)
        reports.append(report)
        print(f"{name:20s} {report['status']:14s} {report['reason']}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))
    # This inventory is deliberately separate from the immutable, curated audit manifest.
    print(f"Sample inventory -> {out}")


if __name__ == "__main__":
    main()
