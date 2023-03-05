import pytest
from datetime import datetime
import numpy
import pandas
from pandas import testing
from conjurer import eda


@pytest.fixture
def data():
    return pandas.DataFrame({
        "store_id": [0, 0, 1, 1, 1, 1],
        "date": [datetime(2023, 1, 1), datetime(2023, 1, 8), datetime(2022, 12, 25), 
                 datetime(2023, 1, 1), datetime(2023, 1, 8), datetime(2023, 1, 15)],
        "event": ["None", "None", "Holiday", "Holiday", "None", "None"],
        "num_visitors": [100, 105, 80, 75, 80, 95]
    })


def test_single_key(data):
    df = data
    result = eda.check_series(df, ["store_id"])
    testing.assert_frame_equal(
        result, pandas.DataFrame({
            "store_id": [0, 1],
            "count": [2, 4],
            "min(date)": [datetime(2023, 1, 1), datetime(2022, 12, 25)],
            "max(date)": [datetime(2023, 1, 8), datetime(2023, 1, 15)],
            "nunique(event)": [1, 2],
            "min(num_visitors)": [100, 75],
            "max(num_visitors)": [105, 95],
            "mean(num_visitors)": [102.5, 82.5],
            "std(num_visitors)": [numpy.std([100, 105], ddof=1), numpy.std([80, 75, 80, 95], ddof=1)]
        }).set_index("store_id")
    )
