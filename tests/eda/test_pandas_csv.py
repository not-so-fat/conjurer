import io
import unittest
from datetime import datetime

import pandas
from pandas import testing

from conjurer import eda
from conjurer.logic.eda.load import pandas_csv
from . import csv_data


class TestLoadCSV(unittest.TestCase):
    def test_integer(self):
        df = eda.read_csv(io.StringIO(csv_data.INTEGER_TEST_CSV))
        testing.assert_frame_equal(
            df,
            pandas.DataFrame({
                "int1": pandas.Series([1, 2, None, 3], dtype="Int64"),
                "int2": pandas.Series([None, 1, 2, 3], dtype="Int64")
            })
        )

    def test_all_type(self):
        df = eda.read_csv(io.StringIO(csv_data.ALL_TYPE_TEST_CSV))
        testing.assert_frame_equal(
            df,
            pandas.DataFrame({
                "date1": pandas.Series([datetime(2020, 1, 1), datetime(2020, 12, 31),
                                        datetime(2000, 3, 1), datetime(2000, 3, 1)], dtype="datetime64[ns]"),
                "date2": pandas.Series([None, datetime(2020, 1, 1),
                                        datetime(2020, 3, 31), datetime(2020, 7, 30)], dtype="datetime64[ns]"),
                "timestamp1": pandas.Series([datetime(2020, 1, 1, 12, 12, 30), datetime(2020, 1, 1, 0, 0, 59),
                                             datetime(2020, 1, 1, 0, 0, 59), datetime(2020, 1, 1, 0, 0, 59)],
                                            dtype="datetime64[ns]"),
                "int1": pandas.Series([None, 1, 2, 3], dtype="Int64"),
                "float1": pandas.Series([1.1, 2.5, 3.8, -1.3], dtype="float64"),
                "str1": pandas.Series(["A-1113", "B-1515", pandas.NA, "C-1335"]),
                "null": pandas.Series([None, None, None, None], dtype="float64")
            })
        )

    def test_get_timestamp_columns(self):
        date_columns = pandas_csv.get_timestamp_columns(pandas.read_csv(io.StringIO(csv_data.ALL_TYPE_TEST_CSV)))
        self.assertListEqual(date_columns, ["date1", "date2", "timestamp1"])
