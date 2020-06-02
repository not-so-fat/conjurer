import logging

import numpy
import pandas
from pandas.api import types


logger = logging.getLogger(__name__)


def read_csv(filepath_or_buffer, **kwargs):
    """
    pandas.read_csv with automatic data type inference for int / timestamp, and resolve string issue
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.errors.DtypeWarning.html
    """
    df = pandas.read_csv(filepath_or_buffer, **kwargs)
    df = convert_integer_columns(df)
    df = convert_timestamp_columns(df)
    df = convert_string_columns(df)
    return df


def convert_integer_columns(df):
    for c in df.columns:
        if numpy.issubdtype(df.dtypes[c], numpy.number):
            integer_series = pandas.Series(df[c], dtype="Int64")
            if numpy.equal(abs(integer_series - df[c]).max(), 0):
                df[c] = integer_series
    return df


def convert_timestamp_columns(df):
    timestamp_columns = get_timestamp_columns(df)
    for c in timestamp_columns:
        df[c] = pandas.to_datetime(df[c])
    return df


def convert_string_columns(df):
    for c in [c for c in df.columns if types.is_object_dtype(df.dtypes[c])]:
        df[c] = df[c].fillna("").astype(str).replace("", pandas.NA)
    return df


def get_timestamp_columns(df):
    date_columns = []
    for c in [c for c in df.columns if types.is_object_dtype(df.dtypes[c])]:
        series = df[~df[c].isnull()][c].head()
        try:
            pandas.to_datetime(series)
        except Exception:
            continue
        else:
            logger.debug("column {} is detected as timestamp: {}".format(c, series))
            date_columns.append(c)
    return date_columns


def _is_integer(series):
    if abs(pandas.Series(series, dtype="Int64") - series).max() == 0:
        return True
