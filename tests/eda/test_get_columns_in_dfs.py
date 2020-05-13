import unittest

import pandas
from pandas import testing

from conjurer import eda


class TestGetColumnsInDf(unittest.TestCase):
    def test_basic(self):
        input_dfs = [
            pandas.DataFrame(columns=["column1", "column2", "column3"]),
            pandas.DataFrame(columns=["column2", "column3"]),
            pandas.DataFrame(columns=["column3"])
        ]
        name_list = ["table1", "table2", "table3"]
        schema_summary = eda.get_columns_in_dfs(input_dfs, name_list)
        testing.assert_frame_equal(
            schema_summary,
            pandas.DataFrame({
                "table_name": ["table1", "table1", "table1", "table2", "table2", "table3"],
                "column_name": ["column1", "column2", "column3", "column2", "column3", "column3"]
            })
        )
