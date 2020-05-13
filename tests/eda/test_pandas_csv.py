import io
import unittest
from datetime import datetime

import pandas
from pandas import testing

from conjurer import eda
from conjurer.logic.eda.load import pandas_csv


INTEGER_TEST_CSV = """int1,int2
1,
2,1
,2
3,3
"""

ALL_TYPE_TEST_CSV = """date1,date2,timestamp1,int1,float1,str1,null
2020/1/1,,2020-01-01T12:12:30,,1.1,A-1113,
2020/12/31,2020-01-01,2020-01-01T00:00:59,1,2.5,B-1515,
2000/3/1,2020-03-31,2020-01-01T00:00:59,2,3.8,,
2000/3/1,2020-07-30,2020-01-01T00:00:59,3,-1.3,C-1335,
"""


class TestLoadCSV(unittest.TestCase):
    def test_integer(self):
        df = eda.read_csv(io.StringIO(INTEGER_TEST_CSV))
        testing.assert_frame_equal(
            df,
            pandas.DataFrame({
                "int1": pandas.Series([1, 2, None, 3], dtype="Int64"),
                "int2": pandas.Series([None, 1, 2, 3], dtype="Int64")
            })
        )

    def test_all_type(self):
        df = eda.read_csv(io.StringIO(ALL_TYPE_TEST_CSV))
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
                "str1": pandas.Series(["A-1113", "B-1515", "", "C-1335"]),
                "null": pandas.Series([None, None, None, None], dtype="float64")
            })
        )

    def test_get_timestamp_columns(self):
        date_columns = pandas_csv.get_timestamp_columns(pandas.read_csv(io.StringIO(ALL_TYPE_TEST_CSV)))
        self.assertListEqual(date_columns, ["date1", "date2", "timestamp1"])
