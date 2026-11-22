import logging
import warnings

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
        if not types.is_numeric_dtype(df.dtypes[c]):
            continue
        if df[c].isna().all():
            continue
        try:
            # pandas 3 raises on non-integral floats; older pandas truncated then we rejected
            integer_series = df[c].astype("Int64")
        except (TypeError, ValueError):
            continue
        if (integer_series - df[c]).abs().max() == 0:
            df[c] = integer_series
    return df


def convert_timestamp_columns(df):
    timestamp_columns = get_timestamp_columns(df)
    for c in timestamp_columns:
        parsed = _parse_timestamp_series(df[c])
        if parsed is not None:
            df[c] = parsed
    return df


def convert_string_columns(df):
    for c in [c for c in df.columns if _is_text_like_dtype(df.dtypes[c])]:
        df[c] = df[c].fillna("").astype(str).replace("", pandas.NA)
    return df


def get_timestamp_columns(df):
    timestamp_columns = []
    for c in [c for c in df.columns if _is_text_like_dtype(df.dtypes[c])]:
        series = df[c].dropna().head(100)
        if _parse_timestamp_series(series) is not None:
            logger.debug("column {} is detected as timestamp: {}".format(c, series))
            timestamp_columns.append(c)
    return timestamp_columns


def _parse_timestamp_series(series):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.filterwarnings(
                "ignore", message="In a future version of pandas, parsing datetimes "
                "with mixed time zones", category=FutureWarning
            )
            try:
                parsed = pandas.to_datetime(series)
            except ValueError:
                # The format is inferred from the first value, so ISO 8601 variants
                # in one column (e.g. "...+00:00" and "...T...Z") fail; retry as ISO 8601.
                parsed = pandas.to_datetime(series, format="ISO8601")
        # Pandas 2 can return object dtype for mixed offsets. Keep the original text.
        if types.is_datetime64_any_dtype(parsed.dtype):
            return parsed.dt.as_unit("ns")
    except (TypeError, ValueError, OverflowError):
        # Parsing or ns conversion can fail on values outside the inference sample.
        return None
    return None


def _is_text_like_dtype(dtype):
    # pandas 3 reads CSV text as StringDtype; older pandas used object
    return types.is_object_dtype(dtype) or types.is_string_dtype(dtype)


def _is_integer(series):
    if abs(pandas.Series(series, dtype="Int64") - series).max() == 0:
        return True
