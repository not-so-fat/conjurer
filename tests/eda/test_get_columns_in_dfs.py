import pandas
from pandas import testing

from conjurer import eda


def test_get_columns_in_dfs():
    input_dfs = [
        pandas.DataFrame(
            data={
                "column1": pandas.Series([], dtype=object),
                "column2": pandas.Series([], dtype="Int64"),
                "column3": pandas.Series([], dtype=float)
            },
            columns=["column1", "column2", "column3"]
        ),
        pandas.DataFrame(
            data={
                "column2": pandas.Series([], dtype="Int64"),
                "column3": pandas.Series([], dtype=object)
            },
            columns=["column2", "column3"],
        ),
        pandas.DataFrame(
            data={
                "column3": pandas.Series([], dtype=object)
            },
            columns=["column3"]
        )
    ]
    name_list = ["table1", "table2", "table3"]
    schema_summary = eda.get_columns_in_dfs(input_dfs, name_list)
    testing.assert_frame_equal(
        schema_summary,
        pandas.DataFrame({
            "table_name": ["table1", "table1", "table1", "table2", "table2", "table3"],
            "column_name": ["column1", "column2", "column3", "column2", "column3", "column3"],
            "dtype": ["object", "Int64", "float", "Int64", "object", "object"]
        })
    )
