"""Verify retention of zero-target groups and the user-specified sorting priority."""

import pandas as pd

from scripts.eda_ev9v_binary_coverage import VIEWS, build_matrix, verify


def videos(groups):
    rows = []
    for prefix, labels in groups.items():
        for index, label in enumerate(labels):
            rows.append({"original_id": f"{prefix}_000_320x240_{index}", "original_view": label})
    return pd.DataFrame(rows)


def test_pmpala_only_group_remains_zero_and_psax_is_binary_union():
    ev = videos(
        {
            "2022-05-03_09-59-32": ["PASA", "PMASA", "PASA", "PMPALA"],
            "2022-05-04_09-59-32": ["PMPALA", "PMPALA"],
        }
    )
    grouped, matrix, coverage, distribution = build_matrix(ev)
    verify(grouped, matrix, coverage, distribution)
    assert len(matrix) == 2
    assert matrix.iloc[0].PSAX == 1
    assert matrix.iloc[0].n_videos == 4
    assert matrix.iloc[1][VIEWS].sum() == 0
    assert matrix.iloc[1].n_videos == 2
    assert matrix.iloc[1].n_view_families == 0
    assert coverage.set_index("view").loc["PSAX", "coverage_percent"] == 50
    assert distribution.set_index("n_view_families").loc[0, "groups"] == 1


def test_sort_prioritizes_a4c_over_psax_then_total_videos_then_filename():
    prefixes = [f"2022-05-0{i}_09-59-32" for i in range(1, 7)]
    ev = videos(
        {
            prefixes[0]: ["PLHLA", "PASA"] * 5,
            prefixes[1]: ["PLHLA", "A4C"],
            prefixes[2]: ["PLHLA", "A4C", "PMPALA"],
            prefixes[3]: ["PLHLA", "A4C", "PMPALA"],
            prefixes[4]: ["PLHLA", "PASA", "A4C"],
            prefixes[5]: ["PMPALA"] * 12,
        }
    )
    grouped, matrix, coverage, distribution = build_matrix(ev)
    verify(grouped, matrix, coverage, distribution)
    assert list(matrix.columns[:6]) == ["filename_group", *VIEWS]
    assert matrix.filename_group.tolist() == [prefixes[i] for i in [4, 2, 3, 1, 0, 5]]
