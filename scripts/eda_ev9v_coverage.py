"""Describe EV9V filename-group view coverage from an existing frozen EDA table.

No group is interpreted as a patient or a verified examination. No pixels,
classifiers, inferred labels, or synthetic-stream recipes enter this analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter

RAW_VIEWS = ["PLHLA", "PASA", "PMVLSA", "PPMLSA", "PMASA", "A4C", "A5C", "SC4C", "PMPALA"]
MERGED_VIEWS = ["PLAX", "PSAX", "A4C", "A5C", "SC4C", "PMPALA"]
MERGE = dict(
    zip(RAW_VIEWS, ["PLAX", "PSAX", "PSAX", "PSAX", "PSAX", "A4C", "A5C", "SC4C", "PMPALA"])
)
NAMES = {
    "PLHLA": "Parasternal long-axis (PLHLA)",
    "PASA": "Parasternal short-axis (PASA)",
    "PMVLSA": "Parasternal short-axis: mitral valve level (PMVLSA)",
    "PPMLSA": "Parasternal short-axis: papillary muscle level (PPMLSA)",
    "PMASA": "Parasternal short-axis: apical level (PMASA)",
    "A4C": "Apical four-chamber (A4C)",
    "A5C": "Apical five-chamber (A5C)",
    "SC4C": "Subcostal four-chamber (SC4C)",
    "PMPALA": "Pulmonary artery long-axis (PMPALA; separate)",
    "PLAX": "Parasternal long-axis (PLAX)",
    "PSAX": "Parasternal short-axis (PSAX; four codes merged)",
}
HEAT_LABELS = [
    "Parasternal\nlong-axis\nPLHLA",
    "Parasternal\nshort-axis\nPASA",
    "Mitral valve\nshort-axis\nPMVLSA",
    "Papillary muscle\nshort-axis\nPPMLSA",
    "Apical-level\nshort-axis\nPMASA",
    "Apical\nfour-chamber\nA4C",
    "Apical\nfive-chamber\nA5C",
    "Subcostal\nfour-chamber\nSC4C",
    "Pulmonary artery\nlong-axis\nPMPALA",
]
RULES = {"timestamp": "Date-time prefix", "full_prefix": "Prefix before _320x240"}
TEAL, BLUE = "#147d80", "#4769a5"
PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:_|-seg)")


def assign_groups(ev: pd.DataFrame) -> pd.DataFrame:
    """Validate released labels and preserve both literal filename grouping rules."""
    ev = ev.copy()
    if ev.empty or ev.original_id.isna().any() or ev.original_id.duplicated().any():
        raise ValueError("Expected nonempty EV9V data with unique, nonmissing original_id")
    if ev.original_view.isna().any() or not set(ev.original_view).issubset(RAW_VIEWS):
        raise ValueError("Missing or unrecognized official EV9V view labels")
    ids = ev.original_id.astype(str)
    ev["timestamp"] = ids.str.extract(PATTERN, expand=False)
    if ev.timestamp.isna().any() or not ids.str.contains("_320x240", regex=False).all():
        raise ValueError("Unparsed EV9V filenames; review the grouping rule before continuing")
    # The second rule retains -seg1/-seg2 and _000/_001 where they occur before resolution.
    ev["full_prefix"] = ids.str.split("_320x240", regex=False).str[0]
    ev["merged_view"] = ev.original_view.map(MERGE)
    return ev


def summarize_groups(ev: pd.DataFrame, rule: str, merged: bool = False) -> dict:
    """Use one vote per group per view, while retaining all video counts."""
    views = MERGED_VIEWS if merged else RAW_VIEWS
    field = "merged_view" if merged else "original_view"
    counts = pd.crosstab(ev[rule], ev[field]).reindex(columns=views, fill_value=0)
    presence = counts.gt(0)
    groups = pd.DataFrame({"videos": counts.sum(axis=1), "n_views": presence.sum(axis=1)})
    groups["pattern"] = presence.astype(int).astype(str).agg("".join, axis=1)
    groups["split_count"] = ev.groupby(rule).split.nunique().reindex(groups.index)
    groups["split"] = ev.groupby(rule).split.agg(lambda x: "|".join(sorted(set(x))))
    groups = groups.sort_values(
        ["n_views", "pattern", "videos", rule], ascending=[False, False, False, True]
    )
    groups["plot_rank"] = np.arange(1, len(groups) + 1)
    counts = counts.loc[groups.index]
    present = counts.gt(0).sum(axis=0)
    coverage = pd.DataFrame(
        {
            "view": views,
            "groups_with_view": present.values,
            "groups_without_observed_label": len(groups) - present.values,
            "total_groups": len(groups),
            "coverage_percent": present.values / len(groups) * 100,
            "videos": counts.sum(axis=0).values,
        }
    )
    diversity = groups.n_views.value_counts().reindex(range(1, len(views) + 1), fill_value=0)
    diversity = diversity.rename_axis("n_views").reset_index(name="groups")
    diversity["percent"] = diversity.groups / len(groups) * 100
    combinations = groups.pattern.value_counts().rename_axis("pattern").reset_index(name="groups")
    combinations = combinations.sort_values(["groups", "pattern"], ascending=[False, False])
    combinations["views"] = combinations.pattern.map(
        lambda x: " + ".join(v for v, bit in zip(views, x) if bit == "1")
    )
    combinations["percent"] = combinations.groups / len(groups) * 100
    return {
        "counts": counts,
        "groups": groups,
        "coverage": coverage,
        "diversity": diversity,
        "combinations": combinations,
    }


def check_tables(ev, tables):
    """Check denominators, video conservation, hierarchy and union semantics."""
    for (rule, merged), tab in tables.items():
        n = ev[rule].nunique()
        assert tab["counts"].to_numpy().sum() == len(ev)
        assert tab["coverage"].videos.sum() == len(ev)
        assert tab["diversity"].groups.sum() == n
        assert tab["combinations"].groups.sum() == n
        assert (tab["coverage"].groups_with_view <= n).all()
        assert tab["groups"].split_count.eq(1).all(), "A group crosses official splits"
        if merged:
            raw = tables[rule, False]["counts"]
            expected = raw[["PASA", "PMVLSA", "PPMLSA", "PMASA"]].sum(axis=1)
            pd.testing.assert_series_equal(
                tab["counts"].PSAX.sort_index(), expected.sort_index(), check_names=False
            )
    assert ev.groupby("full_prefix").timestamp.nunique().eq(1).all()
    return {
        "video_conservation": True,
        "group_denominators": True,
        "psax_union_verified": True,
        "nested_prefix_rules": True,
        "no_groups_cross_official_splits": True,
    }


def make_figures(tables, root):
    registry = []

    def save(fig, name, caption):
        for suffix in ["png", "pdf"]:
            fig.savefig(
                root / "figures" / f"{name}.{suffix}",
                dpi=200,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(fig)
        registry.append(
            {
                "name": name,
                "caption": caption,
                "evidence": "All frozen native EV9V records; official clip labels",
            }
        )

    primary = tables["timestamp", False]
    detailed = tables["full_prefix", False]
    n, n_detail = len(primary["groups"]), len(detailed["groups"])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(14, 6.4), layout="constrained")
    colors = ["#f4f5f7", "#d5eceb", "#9ad0cc", "#55aba8", TEAL, "#17354c"]
    binned = np.digitize(primary["counts"].to_numpy(), [1, 2, 3, 5, 10])
    im = ax.imshow(
        binned,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(colors),
        norm=BoundaryNorm(np.arange(-0.5, 6.5), 6),
    )
    ax.set_xticks(range(9), HEAT_LABELS, fontsize=9)
    ranks = [1] + list(range(100, n + 1, 100)) + [n]
    ax.set_yticks(np.array(ranks) - 1, ranks)
    ax.set_ylabel("Filename group (sorted rank)")
    ax.set_xlabel("Cardiac view")
    ax.set_title(f"EV9V view availability by date-time prefix | {n} groups; 5,138 videos")
    cb = fig.colorbar(im, ax=ax, ticks=range(6), pad=0.018, fraction=0.03)
    cb.ax.set_yticklabels(["0", "1", "2", "3-4", "5-9", "10+"])
    cb.set_label("Video count")
    save(
        fig,
        "13_ev9v_group_heatmap",
        "Each row is a filename-derived group, not a verified patient or examination. "
        "Rows are sorted by view count, label pattern and video count; rank-to-prefix lookup "
        "is supplied in the group table. Zero means no released clip label observed in that group.",
    )

    fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharey=True, layout="constrained")
    for ax, (rule, tab), color in zip(
        axs, [("timestamp", primary), ("full_prefix", detailed)], [TEAL, BLUE]
    ):
        d = tab["diversity"]
        bars = ax.bar(d.n_views, d.groups, color=color, width=0.72)
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_xticks(range(1, 10))
        ax.set_xlabel("Number of observed cardiac views")
        ax.set_title(f"{RULES[rule]} (n={len(tab['groups'])})")
        ax.set_ylim(
            0, max(primary["diversity"].groups.max(), detailed["diversity"].groups.max()) * 1.18
        )
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.15)
    axs[0].set_ylabel("Filename group count")
    save(
        fig,
        "14_ev9v_group_diversity",
        "Each group contributes once. The count is the number of distinct released labels "
        "represented, from one to nine. The two panels use different grouping rules and "
        "denominators, not different video cohorts. Nine labels do not establish clinical completeness.",
    )

    def coverage_bars(tab, title, name, caption):
        d = tab["coverage"]
        fig, ax = plt.subplots(figsize=(13, 5.0 if len(d) == 9 else 4.2), layout="constrained")
        ax.barh(range(len(d)), d.coverage_percent, color=TEAL, height=0.65)
        ax.set_yticks(range(len(d)), [NAMES[v] for v in d.view], fontsize=10)
        ax.invert_yaxis()
        for i, row in enumerate(d.itertuples()):
            ax.text(
                row.coverage_percent + 1,
                i,
                f"{row.groups_with_view}/{row.total_groups} ({row.coverage_percent:.1f}%)",
                va="center",
                fontsize=9,
            )
        ax.set_xlim(0, 119)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.xaxis.set_major_formatter(PercentFormatter(100))
        ax.set_xlabel("Filename groups containing at least one video with this label (%)")
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.15)
        save(fig, name, caption)

    coverage_bars(
        primary,
        f"Cardiac view coverage | {n} date-time prefix groups",
        "15_ev9v_group_coverage",
        f"Coverage = groups with at least one matching clip / all {n} groups. "
        "Each group can contribute to several views, so percentages need not sum to 100%. "
        "This is group-weighted availability, not video class frequency or a patient missingness rate.",
    )

    top = primary["combinations"].head(12)
    fig, (ax, bars) = plt.subplots(
        1,
        2,
        figsize=(13, 5.2),
        sharey=True,
        gridspec_kw={"width_ratios": [2, 1.3]},
        layout="constrained",
    )
    for y, pattern in enumerate(top.pattern):
        selected = [i for i, bit in enumerate(pattern) if bit == "1"]
        ax.scatter(range(9), [y] * 9, s=32, color="#dce2e8")
        ax.plot(selected, [y] * len(selected), color=TEAL, linewidth=1.4)
        ax.scatter(selected, [y] * len(selected), s=48, color=TEAL, zorder=3)
    ax.set_xticks(range(9), RAW_VIEWS, rotation=40, ha="right")
    ax.set_yticks(range(len(top)), [f"{i + 1}" for i in range(len(top))])
    ax.invert_yaxis()
    ax.set_ylabel("Combination rank")
    ax.set_title("Exact label combination")
    ax.set_xlabel("Cardiac view (colored = observed; gray = not observed)")
    b = bars.barh(range(len(top)), top.groups, color=TEAL, height=0.65)
    bars.bar_label(
        b, labels=[f"{r.groups} ({r.percent:.1f}%)" for r in top.itertuples()], padding=5
    )
    bars.set_xlim(0, top.groups.max() * 1.5)
    bars.set_xlabel("Filename group count")
    bars.tick_params(axis="y", left=False, labelleft=False)
    bars.set_title(f"Top 12 of {len(primary['combinations'])} combinations")
    bars.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(
        fig,
        "16_ev9v_group_combinations",
        f"Patterns are exact combinations: gray markers indicate labels absent from that pattern. "
        f"The 12 shown patterns contain {int(top.groups.sum())}/{n} groups; "
        f"{n - int(top.groups.sum())} groups have other patterns and remain in the complete CSV. "
        "Within-pattern repeated videos do not increase the group count.",
    )

    fig, ax = plt.subplots(figsize=(13, 5.0), layout="constrained")
    a, b = primary["coverage"], detailed["coverage"]
    ax.hlines(
        range(9),
        np.minimum(a.coverage_percent, b.coverage_percent),
        np.maximum(a.coverage_percent, b.coverage_percent),
        color="#a6b6c0",
        linewidth=2,
    )
    ax.scatter(
        a.coverage_percent, range(9), color=TEAL, s=70, label=f"Date-time prefix (n={n})", zorder=3
    )
    ax.scatter(
        b.coverage_percent,
        range(9),
        color=BLUE,
        marker="s",
        s=45,
        label=f"Prefix before _320x240 (n={n_detail})",
        zorder=3,
    )
    ax.set_yticks(range(9), [NAMES[v] for v in RAW_VIEWS], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xlabel("Filename groups containing the view (%)")
    ax.set_title("Coverage sensitivity to the filename grouping rule")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.15)
    save(
        fig,
        "17_ev9v_group_sensitivity",
        "Both rules include the same 5,138 videos. Retaining additional prefix tokens subdivides "
        "some date-time groups. Differences are descriptive percentage-point changes; neither "
        "rule is validated as a patient identifier. No patient-level confidence interval is inferred.",
    )

    coverage_bars(
        tables["timestamp", True],
        f"Grouped cardiac view coverage | {n} date-time prefix groups",
        "18_ev9v_group_merged",
        "PASA, PMVLSA, PPMLSA and PMASA are merged into PSAX; PLHLA is renamed PLAX. "
        "PSAX coverage is the union of groups with any of those four labels, not a sum "
        "of four percentages. PMPALA remains a separate sixth category; all 5,138 videos remain.",
    )
    return registry


def report_sections(summary):
    main = summary["analyses"]["timestamp_raw9"]
    alt = summary["analyses"]["full_prefix_raw9"]
    top = max(main["coverage"], key=lambda r: r["coverage_percent"])
    low = min(main["coverage"], key=lambda r: r["coverage_percent"])
    n = main["groups"]
    delta = max(summary["sensitivity"], key=lambda r: abs(r["difference_pp"]))
    titles = [
        "Filename-group view availability",
        "Number of views per filename group",
        "Cardiac view coverage by filename group",
        "Common cardiac-view combinations",
        "Sensitivity to the filename grouping rule",
        "Coverage with PSAX subtypes merged",
    ]
    figures = [
        "13_ev9v_group_heatmap",
        "14_ev9v_group_diversity",
        "15_ev9v_group_coverage",
        "16_ev9v_group_combinations",
        "17_ev9v_group_sensitivity",
        "18_ev9v_group_merged",
    ]
    texts = [
        [
            (
                f"<b>Unit:</b> {summary['videos']:,} original EV9V videos form {n} date-time prefix groups. "
                "For example, 2022-05-03_09-59-32_000_320x240_6 becomes 2022-05-03_09-59-32. "
                "Clip suffixes, optional segment tokens and extra numeric tokens are excluded from this first rule."
            ),
            (
                "<b>Interpretation:</b> these are literal filename groups, not verified patients, examinations "
                "or complete acquisitions. The official release reports 703 patients but supplies no usable "
                "video-to-patient mapping. Missing cells mean that no clip with that released label was observed; "
                "they do not prove that a view was never acquired."
            ),
        ],
        [
            (
                f"Under the date-time rule, the median group contains {main['median_views']:g} distinct labels "
                f"and {main['median_videos']:g} videos. {main['single_view_groups']} groups contain one label; "
                f"{main['all_categories_groups']}/{n} ({main['all_categories_groups'] / n * 100:.1f}%) contain all nine. "
                f"The more detailed rule gives {alt['groups']} groups and a median of {alt['median_views']:g} labels."
            ),
            (
                "The analysis counts observed released clip labels, not temporally annotated appearances. "
                "A4C and A5C are distinct; A2C is outside the EV9V label set and is not counted as missing. "
                "A group with all nine labels is not necessarily a complete clinical examination."
            ),
        ],
        [
            (
                f"{NAMES[top['view']]} has the highest observed coverage: {top['groups_with_view']}/{n} "
                f"groups ({top['coverage_percent']:.1f}%). {NAMES[low['view']]} has the lowest: "
                f"{low['groups_with_view']}/{n} ({low['coverage_percent']:.1f}%)."
            ),
            (
                "Each group receives one vote for a view, regardless of how many matching clips it contains. "
                "The denominator includes every parsed group; no group is excluded for having few views. "
                "Availability is descriptive and does not measure clinical acquisition failure."
            ),
        ],
        [
            (
                f"There are {main['distinct_combinations']} different exact nine-label combinations. "
                "The left matrix identifies the observed labels in each displayed pattern; the right bars "
                "count groups with exactly that pattern. The complete combination table includes every pattern."
            ),
            (
                "Common combinations can guide representative QC sampling. They do not establish a clinical "
                "protocol, patient phenotype, or temporal order between views."
            ),
        ],
        [
            (
                f"Retaining all text before _320x240 changes the count from {n} to {alt['groups']} groups. "
                f"{summary['timestamp_groups_subdivided']} date-time groups split into multiple detailed groups. "
                "Both definitions keep all original videos and neither has groups crossing the official splits."
            ),
            (
                f"The largest absolute coverage change is {delta['view']}: "
                f"{delta['difference_pp']:+.1f} percentage points (detailed minus date-time rule). "
                "The comparison measures grouping sensitivity, not uncertainty about a validated patient rate. "
                "Agreement with a split boundary alone does not validate patient identity."
            ),
        ],
        [
            (
                "The nine released labels are reduced to six reporting categories by merging four short-axis "
                "codes into PSAX and retaining PMPALA separately. No video is discarded for lacking a five-family mapping."
            ),
            (
                "A group containing several PSAX subtypes contributes once to PSAX coverage. "
                "The raw nine-label analysis remains primary because subtype differences would otherwise be hidden. "
                "CSV outputs include both grouping rules under both label schemes."
            ),
        ],
    ]
    return [
        {"title": f"4.{12 + i} EV9V: {title}", "figure": figure, "paragraphs": paragraphs}
        for i, (title, figure, paragraphs) in enumerate(zip(titles, figures, texts))
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, required=True, help="New EDA revision copied from a frozen snapshot"
    )
    args = ap.parse_args()
    root = args.out
    destination = root / "ev9v_coverage_summary.json"
    if destination.exists():
        ap.error("Coverage results already exist; use a new output directory for another run")
    path = root / "tables/video_inventory.csv"
    from eda_report import REVIEWED_INVENTORY_SHA256, Report

    if hashlib.sha256(path.read_bytes()).hexdigest() != REVIEWED_INVENTORY_SHA256:
        ap.error("This coverage report describes the reviewed 2026-10-05 EDA inventory only")
    d = pd.read_csv(path, low_memory=False)
    ev = assign_groups(
        d.loc[d.dataset.eq("EV9V"), ["case_id", "original_id", "original_view", "split"]]
    )
    tables = {
        (rule, merged): summarize_groups(ev, rule, merged)
        for rule in RULES
        for merged in [False, True]
    }
    validations = check_tables(ev, tables)
    ev.to_csv(root / "tables/ev9v_group_membership.csv", index=False)
    analyses = {}
    for (rule, merged), tab in tables.items():
        tag = f"{rule}_{'merged6' if merged else 'raw9'}"
        for name, frame in tab.items():
            frame.to_csv(
                root / "tables" / f"ev9v_{tag}_{name}.csv", index=name in {"counts", "groups"}
            )
        g = tab["groups"]
        analyses[tag] = {
            "groups": len(g),
            "videos": int(g.videos.sum()),
            "median_views": float(g.n_views.median()),
            "median_videos": float(g.videos.median()),
            "single_view_groups": int(g.n_views.eq(1).sum()),
            "all_categories_groups": int(g.n_views.eq(6 if merged else 9).sum()),
            "distinct_combinations": len(tab["combinations"]),
            "coverage": tab["coverage"].to_dict("records"),
            "diversity": tab["diversity"].to_dict("records"),
        }
    a = tables["timestamp", False]["coverage"].set_index("view")
    b = tables["full_prefix", False]["coverage"].set_index("view")
    comparison = pd.DataFrame(
        {
            "date_time_percent": a.coverage_percent,
            "full_prefix_percent": b.coverage_percent,
            "difference_pp": b.coverage_percent - a.coverage_percent,
        }
    ).reset_index()
    comparison.to_csv(root / "tables/ev9v_grouping_sensitivity.csv", index=False)
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "videos": len(ev),
        "inventory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "grouping_rules": {
            "timestamp": "Leading YYYY-MM-DD_HH-MM-SS only",
            "full_prefix": "Literal substring before _320x240, retaining extra tokens",
        },
        "patient_identity_verified": False,
        "view_source": "Official EV9V clip label",
        "raw_view_order": RAW_VIEWS,
        "merged_view_order": MERGED_VIEWS,
        "view_names": NAMES,
        "merge_mapping": MERGE,
        "analyses": analyses,
        "timestamp_groups_subdivided": int(
            ev.groupby("timestamp").full_prefix.nunique().gt(1).sum()
        ),
        "sensitivity": comparison.to_dict("records"),
        "validation": validations,
        "limitations": [
            "Filename groups are not verified patients or examinations.",
            "No observed clip label does not establish that a view was never acquired.",
            "A2C is outside the released label ontology, not a missing expected category.",
            "Clip labels do not annotate every within-video view or transition.",
        ],
        "official_source": "https://huggingface.co/datasets/bgx666/EV9V/blob/main/README.md",
    }
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    registry = make_figures(tables, root)
    existing = json.loads((root / "figure_manifest.json").read_text())
    names = {r["name"] for r in registry}
    (root / "figure_manifest.json").write_text(
        json.dumps([r for r in existing if r["name"] not in names] + registry, indent=2) + "\n"
    )
    sections = report_sections(summary)
    (root / "ev9v_coverage_sections.json").write_text(json.dumps(sections, indent=2) + "\n")
    report = Report(root)
    report.new_page("EV9V: cardiac view coverage by filename-derived group")
    report.p(
        "<b>Data:</b> 5,138 native EV9V videos from the frozen 2026-10-05 EDA inventory. "
        "Official clip labels are used. Synthetic streams and model predictions are not inputs."
    )
    report.p(
        "<b>Question:</b> which cardiac-view labels occur together in groups formed from the filenames? "
        "This is observed view coverage, not patient-level missingness or clinical completeness."
    )
    report.table(
        ["Grouping rule", "Definition", "Groups"],
        [
            [
                "Primary: date-time prefix",
                "Keep only YYYY-MM-DD_HH-MM-SS",
                analyses["timestamp_raw9"]["groups"],
            ],
            [
                "Sensitivity: detailed prefix",
                "Keep all text before _320x240",
                analyses["full_prefix_raw9"]["groups"],
            ],
        ],
        [185, 445, 140],
    )
    report.p(
        "<b>Coverage:</b> the percentage of groups with at least one clip carrying a given view label. "
        "Every group has equal weight. One group may contain several views, so percentages need not sum to 100%."
    )
    report.p(
        "The official dataset reports 703 patients, but no usable video-to-patient mapping is supplied. "
        "Neither filename rule is assigned a patient or examination interpretation. All nine released labels "
        "are retained; a supplemental analysis merges four PSAX subtypes while keeping PMPALA separate."
    )
    report.p(
        "<b>Source:</b> EV9V by Bo Gou et al., CC BY 4.0; "
        "https://huggingface.co/datasets/bgx666/EV9V/blob/main/README.md",
        small=True,
    )
    for section in sections:
        report.new_page(section["title"])
        report.figure(section["figure"], maxheight=315)
        for text in section["paragraphs"]:
            report.p(text, small=True)
    report.finish(
        "EV9V_View_Coverage_EN",
        "EV9V filename-group view coverage",
        "EV9V | filename-derived groups, not patients | frozen inventory 2026-10-05",
    )
    print(
        json.dumps(
            {
                "videos": len(ev),
                "group_counts": {k: v["groups"] for k, v in analyses.items()},
                "validation": validations,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
