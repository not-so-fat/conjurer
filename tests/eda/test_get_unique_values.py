import pytest

import pandas

from conjurer import eda


@pytest.fixture
def input_df():
    input_df = pandas.DataFrame({
        "integer_1": [1, 2, -3, 0],
        "integer_2": [1, 2, 1, 2],
        "integer_with_null": [1, None, None, None],
        "float_1": [1.5, 2.5, -0.0, 1.3],
        "float_2": [0.1, 0.1, 0.1, 0.3],
        "float_with_null": [None, 0.5, None, -2.0],
        "string_1": ["a", "b", pandas.NA, "d"],
        "string_2": [pandas.NA, "a", "a", "a"],
        "string_with_null": ["aaa", pandas.NA, None, None],
        "all_null": [None, None, None, None]
    })
    for c in input_df.columns:
        dtype = "Int64" if c.startswith("int") else "float" if c.startswith("float") else "object"
        input_df[c] = input_df[c].astype(dtype)
    return input_df


def test_integer_1(input_df):
    _test(input_df, "integer_1", {1, 2, -3, 0})


def test_integer_2(input_df):
    _test(input_df, "integer_2", {1, 2})


def test_integer_with_null(input_df):
    _test(input_df, "integer_with_null", {1})


def test_float_1(input_df):
    _test(input_df, "float_1", {1.5, 2.5, -0.0, 1.3})


def test_float_2(input_df):
    _test(input_df, "float_2", {0.1, 0.3})


def test_float_with_null(input_df):
    _test(input_df, "float_with_null", {0.5, -2.0})


def test_string_1(input_df):
    _test(input_df, "string_1", {"a", "b", "d"})


def test_string_2(input_df):
    _test(input_df, "string_2", {"a"})


def test_string_with_null(input_df):
    _test(input_df, "string_with_null", {"aaa"})


def test_all_null(input_df):
    _test(input_df, "all_null", set([]))


def test_2_composite(input_df):
    _test(input_df, ["integer_1", "float_1"], {(1, 1.5), (2, 2.5), (-3, -0.0), (0, 1.3)})


def test_2_composite_with_null(input_df):
    _test(input_df, ["float_2", "string_with_null"], {(0.1, "aaa")})


def test_2_composite_all_null(input_df):
    _test(input_df, ["float_2", "all_null"], set([]))


def test_3_composite(input_df):
    _test(input_df, ["integer_1", "float_1", "string_with_null"], {(1, 1.5, "aaa")})


def _test(input_df, columns, expected_values):
    unique_values = eda.get_unique_values(input_df, columns)
    assert unique_values == expected_values, \
        "Unique values of {} should be {} (returned {})".format(columns, expected_values, unique_values)
