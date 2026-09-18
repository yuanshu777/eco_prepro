#!/usr/bin/env python
"""Run a frozen audit and export videos, provenance, before/after figures and review tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from echoprep.audit.contact_sheet import before_after, render_case, render_overview
from echoprep.audit.identity import content_sha256, resolve_case_path
from echoprep.audit.manifest import probe_file
from echoprep.readers.fps_tables import apply_fps_provenance
from echoprep.sector.motion import detect_sector
from echoprep.standardize import export_cine, standardize_cine

ROOT = Path(__file__).resolve().parents[1]


def exclusion_zones(spec: dict, height: int, width: int):
    zones = spec.get("exclusion_zones_xyxy_normalized")
    if not zones:
        return None
    mask = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in zones:
        mask[int(y0 * height) : int(y1 * height), int(x0 * width) : int(x1 * width)] = True
    return mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(ROOT / "configs/audit_manifest.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs/audit_stage1"))
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Diagnostic truncation, recorded in run provenance",
    )
    ap.add_argument("--export", default="mp4", choices=["mp4", "avi", "npz", "none"])
    ap.add_argument("--mask-policy", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--only", help="Comma-separated dataset filter")
    ap.add_argument("--config", default=str(ROOT / "configs/datasets.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())["datasets"]
    out = Path(a.out)
    if (out / "results.csv").exists():
        ap.error(
            "Run output already exists; choose a new --out to preserve comparison/review history"
        )
    for folder in ("cases", "standardized", "review_pages"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.manifest, keep_default_na=False)
    if df.id.duplicated().any():
        ap.error("Manifest contains duplicate case IDs")
    if a.only:
        df = df[df.dataset.isin(a.only.split(","))]
    if df.empty:
        ap.error("No cases selected")
    run = {
        "manifest_sha256": hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "max_frames": a.max_frames,
        "export": a.export,
        "mask_policy": a.mask_policy,
        "cases": len(df),
    }
    # Hash working source so dirty runs still have an exact implementation fingerprint.
    source = hashlib.sha256()
    for p in sorted([*ROOT.glob("echoprep/**/*.py"), *ROOT.glob("scripts/*.py")]):
        source.update(str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes())
    run["source_sha256"] = source.hexdigest()
    (out / "run.json").write_text(json.dumps(run, indent=2))
    rows, thumbs = [], []
    for item in df.to_dict("records"):
        start = time.monotonic()
        row = {
            k: item.get(k, "")
            for k in ("id", "legacy_id", "dataset", "label", "case_type", "content_sha256")
        }
        rec = None
        try:
            path = resolve_case_path(item, cfg, ROOT)
            row["path"] = str(path)
            digest = content_sha256(path)
            if item.get("content_sha256") and digest != item["content_sha256"]:
                raise ValueError(
                    "Frozen input content changed; assign a new identity in a new manifest version"
                )
            cine, info = probe_file(
                path, reader=item["reader"], dataset=item["dataset"], max_frames=a.max_frames
            )
            spec = cfg.get(item["dataset"], {})
            apply_fps_provenance(cine, spec, ROOT / spec.get("root", "."))
            sec = detect_sector(
                cine.frames,
                static=cine.n_frames == 1,
                already_standardized=spec.get("already_standardized", False),
                exclusion_mask=exclusion_zones(spec, cine.height, cine.width),
            )
            std, rec = standardize_cine(cine, sec, mask_policy=a.mask_policy)
            rec.update(
                case_id=item["id"],
                content_sha256=digest,
                case_type=item.get("case_type", ""),
                derived_from=item.get("derived_from", ""),
                derivation=item.get("derivation", ""),
                run=run,
                access={
                    k: spec.get(k)
                    for k in (
                        "source_url",
                        "license",
                        "access_method",
                        "redistributable",
                        "redistribution_restrictions",
                    )
                },
            )
            row.update(info)
            row.update(cine.summary())
            row.update(
                {
                    k: rec[k]
                    for k in (
                        "fps_value",
                        "fps_source",
                        "fps_container",
                        "fps_dataset_table",
                        "action",
                    )
                }
            )
            row.update(
                status=sec.status,
                flags="|".join(sec.flags),
                out_h=std.height,
                out_w=std.width,
                **{
                    f"sector.{k}": v
                    for k, v in sec.stats.items()
                    if not isinstance(v, (list, dict))
                },
            )
            if a.export != "none":
                export_cine(std, out / "standardized" / f"{item['id']}.{a.export}", rec)
            else:
                (out / "standardized" / f"{item['id']}.json").write_text(json.dumps(rec, indent=2))
            render_case(
                cine,
                sec,
                out / "cases" / f"{item['id']}.png",
                std,
                title=f"{item['id']} [{item['label']}]",
            )
            thumbs.append(
                (
                    f"{item['id']} {sec.status}",
                    before_after(cine.frames[cine.n_frames // 2], std.frames[std.n_frames // 2]),
                )
            )
            row["error"] = ""
            del cine, std, sec
        except Exception as exc:  # noqa: BLE001 - report failures and continue other cases
            row.update(
                status="FAILED",
                flags="manual_review_required",
                error=f"{type(exc).__name__}: {exc}",
            )
            failure = rec or {"case_id": item["id"], "run": run}
            failure.update(status="FAILED", warnings=["manual_review_required"], error=row["error"])
            (out / "standardized" / f"{item['id']}.json").write_text(json.dumps(failure, indent=2))
            thumbs.append((f"{item['id']} FAILED", np.zeros((300, 600, 3), np.uint8)))
        row["seconds"] = round(time.monotonic() - start, 2)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
        print(
            f"{item['id']:38s} {row['status']:15s} {row['seconds']:6.2f}s {row['error']}",
            flush=True,
        )
    result = pd.DataFrame(rows)
    render_overview(thumbs, out / "overview.png", cols=3, tile=600)
    for i in range(0, len(thumbs), 6):
        render_overview(
            thumbs[i : i + 6], out / "review_pages" / f"page_{i // 6 + 1:02d}.png", cols=2, tile=700
        )
    review = result[["id", "dataset", "label", "status", "flags"]].copy()
    review["review"] = ""
    review["reason"] = ""
    review["reviewer"] = ""
    review["figure"] = review["id"].map(lambda v: f"cases/{v}.png")
    review.to_csv(out / "audit_review.csv", index=False)
    (out / "review_schema.json").write_text(
        json.dumps(
            {
                "review": ["PASS", "MINOR_ISSUE", "FAIL"],
                "reason": "Free text; e.g. clipped anatomy, residual overlay, wrong region, excessive padding, dark sector, multi-region",
                "note": "Blank means unreviewed; automatic SUCCESS does not imply visual PASS.",
            },
            indent=2,
        )
    )
    print(result.groupby(["dataset", "status"]).size().to_string())
    print(f"Results and visual review -> {out}")
    if (result.status == "FAILED").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
