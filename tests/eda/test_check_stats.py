import io

import pandas
from pandas import testing

from conjurer import eda
from conjurer.logic.eda.check import stat_calculator
from . import csv_data


def test_success_execution():
    df = eda.read_csv(io.StringIO(csv_data.ALL_TYPE_TEST_CSV))
    stat_df = eda.check_stats(df)


def test_string():
    df = pandas.DataFrame({
        "str": pandas.Series([pandas.NA, pandas.NA, "aaa", "bbb"])
    })
    stat_df = eda.check_stats(df, True)
    # pandas 2: object; pandas 3: str
    assert str(stat_df.iloc[0]["dtype"]) in {"object", "str", "string"}
    testing.assert_frame_equal(
        stat_df.drop(columns=["dtype"]).reset_index(drop=True),
        pandas.DataFrame(
            data={
                "column_name": ["str"],
                "min": [pandas.NA],
                "max": [pandas.NA],
                "mean": [pandas.NA],
                "std": [pandas.NA],
                "ratio_na": [0.5],
                "ratio_zero": [pandas.NA],
                "unique_count": [2],
                "is_unique": [False]
            }
        ),
        check_dtype=False,
    )


def test_dict_and_list_columns():
    """Sessions / JSON-like frames often store dict/list in object columns."""
    df = pandas.DataFrame({
        "id": [1, 1, 2],
        "payload": [{"x": 1}, {"x": 1}, {"x": 2}],
        "tags": [["a"], ["a"], ["b"]],
    })
    assert stat_calculator.count_duplicated_rows(df) == 1
    stat_df = eda.check_stats(df, skip_histogram=True)
    payload_row = stat_df[stat_df["column_name"] == "payload"].iloc[0]
    assert payload_row["unique_count"] == 2
    assert not bool(payload_row["is_unique"])


def test_check_stats_dict_columns_with_histogram():
    """Histogram path must not crash on dict/list columns (incl. >num_bins uniques)."""
    df = pandas.DataFrame({
        "id": list(range(60)),
        "payload": [{"i": i} for i in range(60)],
        "tags": [["t", i] for i in range(60)],
        "label": [f"c{i % 3}" for i in range(60)],
    })
    stat_df = eda.check_stats(df)
    assert len(stat_df) == 4
