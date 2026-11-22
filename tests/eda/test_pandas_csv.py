import io
from datetime import datetime, timedelta, timezone

import pandas
import pytest
from pandas import testing

from conjurer import eda
from conjurer.logic.eda.load import pandas_csv
from . import csv_data


def test_integer():
    df = eda.read_csv(io.StringIO(csv_data.INTEGER_TEST_CSV))
    testing.assert_frame_equal(
        df,
        pandas.DataFrame({
            "int1": pandas.Series([1, 2, None, 3], dtype="Int64"),
            "int2": pandas.Series([None, 1, 2, 3], dtype="Int64")
        })
    )


def test_all_type():
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


def test_get_timestamp_columns():
    date_columns = pandas_csv.get_timestamp_columns(pandas.read_csv(io.StringIO(csv_data.ALL_TYPE_TEST_CSV)))
    assert date_columns == ["date1", "date2", "timestamp1"]


@pytest.mark.parametrize(
    ("suffix", "tz"),
    [
        ("+00:00", timezone.utc),
        ("Z", timezone.utc),
        ("+05:30", timezone(timedelta(hours=5, minutes=30))),
        ("", None),
    ],
)
def test_timestamp_resolution_preserves_timezone(suffix, tz):
    csv = io.StringIO(f"timestamp,id\n2024-01-01 00:00:00{suffix},1\n,2\n")

    result = eda.read_csv(csv)

    dtype = "datetime64[ns]" if tz is None else pandas.DatetimeTZDtype(unit="ns", tz=tz)
    expected = pandas.Series(
        [datetime(2024, 1, 1, tzinfo=tz), None], dtype=dtype, name="timestamp"
    )
    testing.assert_series_equal(result["timestamp"], expected)


@pytest.mark.parametrize("matching_rows", [1, 100])
def test_mixed_timestamp_offsets_stay_as_text(matching_rows):
    values = ["2024-01-01 00:00:00+00:00"] * matching_rows
    values.append("2024-01-01 00:00:00+05:30")
    csv_text = "timestamp\n" + "\n".join(values)
    expected_candidates = ["timestamp"] if matching_rows == 100 else []
    assert pandas_csv.get_timestamp_columns(
        pandas.read_csv(io.StringIO(csv_text))
    ) == expected_candidates

    result = eda.read_csv(io.StringIO(csv_text))

    assert result["timestamp"].tolist() == values


def test_mixed_iso8601_utc_variants_parse_as_utc():
    csv = io.StringIO("timestamp,id\n2024-01-01 00:00:00+00:00,1\n2024-01-02T00:00:00Z,2\n")

    result = eda.read_csv(csv)

    expected = pandas.Series(
        [datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc)],
        dtype=pandas.DatetimeTZDtype(unit="ns", tz=timezone.utc),
        name="timestamp",
    )
    testing.assert_series_equal(result["timestamp"], expected)
