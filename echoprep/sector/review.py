"""Stage 2 diagnostic runner; reuses the frozen readers/export/audit contracts unchanged.

Run: python -m echoprep.sector.review --out outputs/stage2_subset_v1
The subset is fixed in audit_subset.json. Pass-through data are not used for fitting/tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from echoprep.audit.contact_sheet import before_after, render_overview
from echoprep.audit.identity import content_sha256, resolve_case_path
from echoprep.readers import read_cine
from echoprep.readers.fps_tables import apply_fps_provenance
from echoprep.sector.motion import detect_sector
from echoprep.standardize import export_cine, standardize_cine

ROOT = Path(__file__).resolve().parents[2]


def _zones(spec, h, w):
    zones = spec.get("exclusion_zones_xyxy_normalized")
    if not zones:
        return None
    mask = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in zones:
        mask[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = True
    return mask


def render_geometry(cine, sector, output, target, title):
    frame = cine.frames[cine.n_frames // 2]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.8))
    axs[0].imshow(frame)
    axs[0].set_title("Before")
    motion = sector.maps.get("std", np.zeros(frame.shape[:2]))
    evidence = sector.maps.get("candidate", sector.mask)
    axs[1].imshow(motion, cmap="magma")
    if evidence.any() and not evidence.all():
        axs[1].contour(evidence, levels=[0.5], colors="cyan", linewidths=0.7)
    axs[1].set_title("Motion + candidate support")
    axs[2].imshow(frame)
    allowed = sector.maps.get("allowed", np.ones(frame.shape[:2], bool))
    overlay = np.zeros((*allowed.shape, 4))
    overlay[~allowed] = [1, 0, 0, 0.35]
    axs[2].imshow(overlay)
    proposal = sector.maps.get("fitted", sector.mask)
    if proposal.any() and not proposal.all():
        axs[2].contour(proposal, levels=[0.5], colors="lime", linewidths=1)
    method = sector.stats.get("stage2_method", "retained_stage1")
    family = sector.stats.get(
        "stage2_family", sector.stats.get("stage2_skip_reason", "no accepted fit")
    )
    axs[2].set_title(f"{method}\n{family} (red = excluded UI)")
    axs[3].imshow(output.frames[output.n_frames // 2])
    axs[3].set_title(f"After: {sector.status}")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(target, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--export", choices=["none", "mp4"], default="none")
    parser.add_argument(
        "--stability", action="store_true", help="Compare fits from interleaved temporal samples"
    )
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        parser.error("Use a new run directory")
    (out / "cases").mkdir(parents=True)
    (out / "standardized").mkdir()
    spec = yaml.safe_load((ROOT / "configs/datasets.yaml").read_text())["datasets"]
    manifest = pd.read_csv(ROOT / "configs/audit_manifest.csv", keep_default_na=False).set_index(
        "id"
    )
    subset = json.loads(Path(__file__).with_name("audit_subset.json").read_text())
    freeze = json.loads((ROOT / "docs/stage2_frozen_contract.json").read_text())
    for p, digest in freeze["files"].items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest, (
            f"Frozen Stage 1 file changed: {p}"
        )
    run = {
        "subset": subset,
        "frozen_stage1_verified": True,
        "sector_files": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "echoprep/sector").glob("*"))
            if p.is_file()
        },
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))
    rows = []
    thumbs = []
    for selected in subset:
        row = manifest.loc[selected["id"]].to_dict()
        row["id"] = selected["id"]
        path = resolve_case_path(row, spec, ROOT)
        assert content_sha256(path) == row["content_sha256"]
        cine = read_cine(path, reader=row["reader"], dataset=row["dataset"])
        apply_fps_provenance(cine, spec[row["dataset"]], ROOT / spec[row["dataset"]]["root"])
        kwargs = {
            "static": cine.n_frames == 1,
            "already_standardized": spec[row["dataset"]].get("already_standardized", False),
            "exclusion_mask": _zones(spec[row["dataset"]], cine.height, cine.width),
        }
        sector = detect_sector(cine.frames, **kwargs)
        processed, record = standardize_cine(cine, sector)
        record.update(
            case_id=row["id"], case_role=selected["role"], content_sha256=row["content_sha256"]
        )
        if args.export == "mp4":
            export_cine(processed, out / "standardized" / f"{row['id']}.mp4", record)
        else:
            (out / "standardized" / f"{row['id']}.json").write_text(json.dumps(record, indent=2))
        render_geometry(
            cine,
            sector,
            processed,
            out / "cases" / f"{row['id']}.png",
            f"{row['id']} — {selected['role']}",
        )
        result = {
            "id": row["id"],
            "role": selected["role"],
            "tuning": selected["tuning"],
            "baseline_status": sector.stats.get("stage2_baseline_status"),
            "status": sector.status,
            "method": sector.stats.get("stage2_method"),
            "family": sector.stats.get("stage2_family", ""),
            "flags": "|".join(sector.flags),
            "layout": sector.stats.get("stage2_layout", ""),
            "exclusion_loss": sector.stats.get("stage2_excluded_core_fraction"),
            "skip_reason": sector.stats.get("stage2_skip_reason", ""),
            "stability_passed": sector.stats.get("stage2_stability_passed"),
            "stability_iou": sector.stats.get("stage2_stability_iou"),
            **sector.stats.get("stage2_metrics", {}),
        }
        if args.stability and selected["tuning"]:
            a = detect_sector(cine.frames[::2], **kwargs)
            b = detect_sector(cine.frames[1::2], **kwargs)
            result.update(
                temporal_mask_iou=float(
                    (a.mask & b.mask).sum() / max(int((a.mask | b.mask).sum()), 1)
                ),
                temporal_status_a=a.status,
                temporal_status_b=b.status,
                temporal_family_a=a.stats.get("stage2_family", ""),
                temporal_family_b=b.stats.get("stage2_family", ""),
            )
        result.update(review="", reason="", reviewer="")
        rows.append(result)
        pd.DataFrame(rows).to_csv(out / "review.csv", index=False)
        thumbs.append(
            (
                f"{row['id']} {sector.status}",
                before_after(
                    cine.frames[cine.n_frames // 2], processed.frames[processed.n_frames // 2]
                ),
            )
        )
        print(row["id"], sector.status, result["method"], result["family"], flush=True)
    render_overview(thumbs, out / "overview.png", cols=2, tile=650)


if __name__ == "__main__":
    main()
