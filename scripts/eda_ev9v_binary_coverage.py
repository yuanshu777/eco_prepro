"""Create a binary five-family EV9V heatmap without modifying existing artifacts.

Run from the repository root with python -m scripts.eda_ev9v_binary_coverage.
All original filename groups remain in the denominator, including groups whose
only released label is PMPALA. n_videos includes PMPALA videos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from scripts.eda_ev9v_coverage import assign_groups

VIEWS = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
SORT_PRIORITY = ["PLAX", "A4C", "PSAX", "A5C", "SC4C"]
PSAX_CODES = {"PASA", "PMVLSA", "PPMLSA", "PMASA"}
FAMILY_MAP = {
    "PLHLA": "PLAX",
    "PASA": "PSAX",
    "PMVLSA": "PSAX",
    "PPMLSA": "PSAX",
    "PMASA": "PSAX",
    "A4C": "A4C",
    "A5C": "A5C",
    "SC4C": "SC4C",
}
EXPECTED_COUNTS = {"PLAX": 452, "PSAX": 366, "A4C": 415, "A5C": 159, "SC4C": 158}
NOTE = (
    "Filename-derived groups are not verified patients or examinations. A 'not observed' "
    "cell means that no released EV9V clip with that view label was found within the "
    "filename group; it does not establish that the view was never clinically acquired."
)
TITLE = "EV9V cardiac-view availability across filename-derived groups"
OBSERVED, NOT_OBSERVED = "#146D73", "#F1F3F4"


def build_matrix(ev):
    """Preserve all groups before restricting cell values to the five target families."""
    ev = assign_groups(ev).rename(columns={"timestamp": "filename_group"})
    ev["view_family"] = ev.original_view.map(FAMILY_MAP)
    sizes = ev.groupby("filename_group").size().rename("n_videos")
    eligible = ev[ev.view_family.notna()]
    counts = pd.crosstab(eligible.filename_group, eligible.view_family)
    counts = counts.reindex(index=sizes.index, columns=VIEWS, fill_value=0)
    matrix = counts.gt(0).astype("uint8")
    matrix["n_view_families"] = matrix[VIEWS].sum(axis=1)
    matrix["n_videos"] = sizes
    matrix = (
        matrix.reset_index()
        .sort_values(
            ["n_view_families", *SORT_PRIORITY, "n_videos", "filename_group"],
            ascending=[False] * 7 + [True],
        )
        .reset_index(drop=True)
    )
    matrix.columns.name = None
    matrix["plot_rank"] = np.arange(1, len(matrix) + 1)
    coverage = pd.DataFrame(
        {
            "view": VIEWS,
            "groups_with_view": matrix[VIEWS].sum().to_numpy(),
            "total_groups": len(matrix),
        }
    )
    coverage["coverage_percent"] = coverage.groups_with_view / len(matrix) * 100
    distribution = (
        matrix.n_view_families.value_counts()
        .reindex(range(5, -1, -1), fill_value=0)
        .rename_axis("n_view_families")
        .reset_index(name="groups")
    )
    distribution["percent"] = distribution.groups / len(matrix) * 100
    return ev, matrix, coverage, distribution


def verify(ev, matrix, coverage, distribution):
    """Independently recount union membership and validate the exact requested row order."""
    assert ev.original_id.is_unique and ev.filename_group.notna().all()
    assert matrix.filename_group.is_unique
    assert set(matrix.filename_group) == set(ev.filename_group)
    assert int(matrix.n_videos.sum()) == len(ev)
    assert set(np.unique(matrix[VIEWS].to_numpy())).issubset({0, 1})
    assert (matrix.n_view_families == matrix[VIEWS].sum(axis=1)).all()
    assert int(distribution.groups.sum()) == len(matrix)
    for family in VIEWS:
        raw_codes = {code for code, value in FAMILY_MAP.items() if value == family}
        expected = set(ev.loc[ev.original_view.isin(raw_codes), "filename_group"])
        actual = set(matrix.loc[matrix[family].eq(1), "filename_group"])
        assert actual == expected, family
        assert int(coverage.set_index("view").loc[family, "groups_with_view"]) == len(expected)
    records = matrix.to_dict("records")
    expected_order = sorted(
        records,
        key=lambda r: (
            -r["n_view_families"],
            *(-r[v] for v in SORT_PRIORITY),
            -r["n_videos"],
            r["filename_group"],
        ),
    )
    assert [r["filename_group"] for r in records] == [r["filename_group"] for r in expected_order]
    return {
        "every_original_video_assigned_once": True,
        "all_original_groups_retained": True,
        "psax_union_verified": True,
        "all_cells_binary": True,
        "n_videos_includes_pmpala": True,
        "exact_requested_sort_verified": True,
        "groups_interpreted_as_patients": False,
    }


def draw_heatmap(matrix, coverage, out):
    """Keep five binary vector columns, with no interpolation or continuous color scale."""
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    ):
        fig = plt.figure(figsize=(11.6, 11.6), facecolor="white")
        ax = fig.add_axes([0.20, 0.17, 0.755, 0.66])
        n = len(matrix)
        ax.pcolormesh(
            np.arange(6),
            np.arange(n + 1),
            matrix[VIEWS].to_numpy(),
            cmap=ListedColormap([NOT_OBSERVED, OBSERVED]),
            norm=BoundaryNorm([-0.5, 0.5, 1.5], 2),
            shading="flat",
            edgecolors="none",
            antialiased=False,
            rasterized=False,
        )
        ax.set_xlim(0, 5)
        ax.set_ylim(n, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Cardiac view family", labelpad=12, fontsize=12)
        ax.set_ylabel("Filename-derived groups (sorted)", fontsize=12)
        ax.yaxis.set_label_coords(-0.22, 0.5)
        for spine in ax.spines.values():
            spine.set_color("#C3CBCF")
        for x in range(1, 5):
            ax.axvline(x, color="white", alpha=0.6, linewidth=0.45)
        start = 0
        for k in range(5, -1, -1):
            size = int(matrix.n_view_families.eq(k).sum())
            if not size:
                continue
            if start:
                ax.axhline(start, color="#B2BEC3", linewidth=0.65)
            noun = "view" if k == 1 else "views"
            ax.text(
                -0.025,
                start + size / 2,
                f"{k} {noun} (n={size})",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                color="#45545C",
                clip_on=False,
            )
            start += size
        for i, row in enumerate(coverage.itertuples()):
            ax.text(
                (i + 0.5) / 5,
                1.013,
                f"{row.view}\n{row.groups_with_view}/{row.total_groups}\n{row.coverage_percent:.1f}%",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=11,
                linespacing=1.35,
                color="#18363D",
            )
        fig.suptitle(TITLE, x=0.52, y=0.964, fontsize=15, fontweight="semibold", color="#18363D")
        fig.legend(
            handles=[
                Patch(facecolor=OBSERVED, label="Observed"),
                Patch(facecolor=NOT_OBSERVED, edgecolor="#C3CBCF", label="Not observed"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.56, 0.929),
            ncol=2,
            frameon=False,
            fontsize=10,
            handlelength=1.4,
            columnspacing=2.4,
        )
        fig.text(
            0.20,
            0.108,
            f"{int(matrix.n_videos.sum()):,} videos | {n} filename-derived groups | "
            "PMPALA excluded from five-family cells only",
            fontsize=9,
            color="#45545C",
        )
        fig.text(
            0.065,
            0.076,
            textwrap.fill(NOTE, width=125),
            ha="left",
            va="top",
            fontsize=8.5,
            linespacing=1.45,
            color="#45545C",
        )
        stem = out / "ev9v_binary_view_coverage_heatmap"
        fig.savefig(stem.with_suffix(".png"), dpi=400, facecolor="white")
        fig.savefig(
            stem.with_suffix(".pdf"),
            facecolor="white",
            metadata={"Title": TITLE, "Subject": NOTE, "Author": "eco_prepro"},
        )
        plt.close(fig)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inventory",
        type=Path,
        default=Path("outputs/eda_2026-10-05_v6/tables/video_inventory.csv"),
    )
    ap.add_argument(
        "--native-manifest",
        type=Path,
        default=Path("/home/william/echo-view-routing/data/manifests/ev9v_native.csv"),
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(
            "Output directory already exists; choose a new directory to avoid overwriting files"
        )
    sources = {str(p.resolve()): sha(p) for p in [args.inventory, args.native_manifest]}
    data = pd.read_csv(args.inventory, low_memory=False)
    original = data.loc[data.dataset.eq("EV9V"), ["original_id", "original_view", "path", "split"]]
    native = pd.read_csv(args.native_manifest)
    if native.video_id.duplicated().any():
        ap.error("Duplicate IDs in the native manifest")
    if (
        not original.set_index("original_id")
        .original_view.sort_index()
        .equals(native.set_index("video_id").raw_label.sort_index())
    ):
        ap.error("Frozen inventory filename/label pairs do not match the native manifest")
    if not original.path.map(lambda p: Path(p).stem).eq(original.original_id).all():
        ap.error("A video filename differs from its original_id")
    ev, matrix, coverage, distribution = build_matrix(original)
    checks = verify(ev, matrix, coverage, distribution)
    discrepancies = []
    for label, actual, expected in [("videos", len(ev), 5138), ("groups", len(matrix), 495)]:
        if actual != expected:
            discrepancies.append(f"{label}: observed {actual}, reference {expected}")
    for row in coverage.itertuples():
        if row.groups_with_view != EXPECTED_COUNTS[row.view]:
            discrepancies.append(
                f"{row.view}: observed {row.groups_with_view}, reference {EXPECTED_COUNTS[row.view]}"
            )
    excluded = int(ev.original_view.eq("PMPALA").sum())
    distribution_lookup = distribution.set_index("n_view_families").groups
    lines = [
        TITLE,
        "",
        f"Source inventory: {args.inventory.resolve()}",
        "Filename field: original_id (matches the MP4 path stem)",
        "Official label field: original_view (cross-checked against native raw_label)",
        f"Native manifest: {args.native_manifest.resolve()}",
        f"Total EV9V videos assigned to groups: {len(ev):,}",
        f"Videos contributing to the five families: {len(ev) - excluded:,}",
        f"PMPALA videos excluded from cells only: {excluded:,}",
        f"Total filename-derived groups: {len(matrix):,}",
        "",
        "Column coverage:",
        coverage.to_string(index=False, float_format=lambda x: f"{x:.2f}"),
        "",
        "Observed view-family count distribution:",
        distribution.to_string(index=False, float_format=lambda x: f"{x:.2f}"),
        "",
        f"Groups with all five families: {distribution_lookup.loc[5]}",
        f"Groups with only one family: {distribution_lookup.loc[1]}",
        f"Groups with zero target families: {distribution_lookup.loc[0]}",
        "",
        "Reference discrepancies: " + ("; ".join(discrepancies) or "None"),
        "Data-quality checks: passed; all original records and groups retained.",
        "PSAX is the union of groups with PASA, PMVLSA, PPMLSA or PMASA.",
        "n_videos counts all original videos, including PMPALA.",
        "Rows: descending family count; then PLAX, A4C, PSAX, A5C, SC4C (1 before 0);",
        "then descending n_videos; then filename_group alphabetically. No clustering.",
        "",
        NOTE,
    ]
    report = "\n".join(lines) + "\n"
    # Print the requested QC results before saving the final figure.
    print(report, flush=True)
    args.out.mkdir(parents=True, exist_ok=False)
    matrix.to_csv(args.out / "ev9v_filename_group_view_coverage.csv", index=False)
    coverage.to_csv(args.out / "ev9v_view_coverage_summary.csv", index=False)
    distribution.to_csv(args.out / "ev9v_view_count_distribution.csv", index=False)
    ev[["original_id", "original_view", "filename_group", "view_family"]].to_csv(
        args.out / "ev9v_video_group_membership.csv", index=False
    )
    draw_heatmap(matrix, coverage, args.out)
    unchanged = all(sha(Path(path)) == value for path, value in sources.items())
    assert unchanged, "An input file changed during analysis"
    checks["source_file_hashes_unchanged"] = unchanged
    checks["official_native_manifest_labels_match"] = True
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": sources,
        "filename_field": "original_id",
        "official_label_field": "original_view",
        "grouping_rule": "Leading YYYY-MM-DD_HH-MM-SS only",
        "views": VIEWS,
        "pattern_sort_priority": SORT_PRIORITY,
        "total_videos": len(ev),
        "target_family_videos": len(ev) - excluded,
        "pmpala_videos": excluded,
        "total_groups": len(matrix),
        "coverage": coverage.to_dict("records"),
        "view_count_distribution": distribution.to_dict("records"),
        "checks": checks,
        "reference_discrepancies": discrepancies,
        "n_videos_definition": "All original videos in the group, including PMPALA",
        "png_dpi": 400,
        "pdf_rendering": "Vector cells and embedded TrueType text",
        "interpretation_note": NOTE,
    }
    (args.out / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "analysis_summary.txt").write_text(report)
    for path in sorted(args.out.iterdir()):
        print(path.resolve(), flush=True)


if __name__ == "__main__":
    main()
