#!/usr/bin/env python
"""Print the canonical summary of one file (any supported format) and, optionally, run the sector
detector on it and save a figure.

  python scripts/probe_file.py PATH [--reader video|dicom|nifti|sitk|images] [--detect] [--fig out.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from echoprep.audit.manifest import probe_file
from echoprep.sector.motion import detect_sector
from echoprep.standardize import standardize_cine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--reader", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    cine, info = probe_file(a.path, reader=a.reader, max_frames=a.max_frames)
    print(json.dumps(info, indent=2, default=str))
    if a.detect or a.fig:
        sec = detect_sector(cine.frames, static=cine.n_frames == 1)
        std, _rec = standardize_cine(cine, sec)
        print(json.dumps({"status": sec.status, "bbox": sec.bbox, **sec.stats}, indent=2, default=str))
        if a.fig:
            from echoprep.audit.contact_sheet import render_case
            render_case(cine, sec, a.fig, std, title=Path(a.path).name)
            print("figure ->", a.fig)


if __name__ == "__main__":
    main()
