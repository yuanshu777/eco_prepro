"""Protect grouping uncertainty, coverage denominators, and the PSAX union."""

import pandas as pd
import pytest

from scripts.eda_ev9v_coverage import assign_groups, check_tables, summarize_groups


def records():
    return pd.DataFrame(
        {
            "original_id": [
                "2022-07-01_10-07-13-seg1_000_320x240_0",
                "2022-07-01_10-07-13-seg1_000_320x240_1",
                "2022-07-01_10-07-13-seg2_001_320x240_0",
                "2022-07-01_10-07-13-seg2_001_320x240_1",
                "2022-07-02_10-07-13_320x240_0",
            ],
            "original_view": ["PASA", "PASA", "PMASA", "PMPALA", "A4C"],
            "split": ["train"] * 4 + ["test"],
        }
    )


def test_segment_tokens_are_retained_only_in_detailed_rule():
    ev = assign_groups(records())
    assert ev.timestamp.nunique() == 2
    assert ev.full_prefix.nunique() == 3
    assert ev.iloc[0].timestamp == "2022-07-01_10-07-13"
    assert ev.iloc[0].full_prefix == "2022-07-01_10-07-13-seg1_000"
    assert "patient_id" not in ev.columns


def test_repeated_clips_do_not_inflate_group_coverage_or_combinations():
    result = summarize_groups(assign_groups(records()), "timestamp")
    coverage = result["coverage"].set_index("view")
    assert coverage.loc["PASA", "videos"] == 2
    assert coverage.loc["PASA", "groups_with_view"] == 1
    assert coverage.loc["PASA", "coverage_percent"] == 50
    assert coverage.loc["PLHLA", "groups_with_view"] == 0
    assert result["combinations"].groups.sum() == 2
    assert set(result["groups"].n_views) == {1, 3}


def test_psax_is_a_union_and_pmpala_is_retained():
    ev = assign_groups(records())
    tables = {
        (rule, merged): summarize_groups(ev, rule, merged)
        for rule in ["timestamp", "full_prefix"]
        for merged in [False, True]
    }
    check_tables(ev, tables)
    coverage = tables["timestamp", True]["coverage"].set_index("view")
    assert coverage.loc["PSAX", "groups_with_view"] == 1
    assert coverage.loc["PSAX", "videos"] == 3
    assert coverage.loc["PMPALA", "videos"] == 1
    assert coverage.videos.sum() == len(ev)


@pytest.mark.parametrize(
    "column,value",
    [
        ("original_id", "unrecognized_file"),
        ("original_view", "A2C"),
        ("original_view", None),
        ("original_id", None),
    ],
)
def test_invalid_input_is_not_silently_dropped(column, value):
    ev = records()
    ev.loc[0, column] = value
    with pytest.raises(ValueError):
        assign_groups(ev)


def test_duplicate_native_id_is_rejected():
    ev = records()
    with pytest.raises(ValueError):
        assign_groups(pd.concat([ev, ev.iloc[[0]]]))
