import io

import pandas
from pandas import testing

from conjurer import eda
from . import csv_data


def test_success_execution():
    df = eda.read_csv(io.StringIO(csv_data.ALL_TYPE_TEST_CSV))
    stat_df = eda.check_stats(df)


def test_string():
    df = pandas.DataFrame({
        "str": pandas.Series([pandas.NA, pandas.NA, "aaa", "bbb"])
    })
    stat_df = eda.check_stats(df, True)
    testing.assert_frame_equal(
        stat_df, pandas.DataFrame(
            data={
                "column_name": ["str"],
                "dtype": ["object"],
                "min": [pandas.NA],
                "max": [pandas.NA],
                "mean": [pandas.NA],
                "std": [pandas.NA],
                "ratio_na": [0.5],
                "ratio_zero": [pandas.NA],
                "unique_count": [2],
                "is_unique": [False]
            }
        ))
