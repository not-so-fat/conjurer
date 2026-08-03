import math
import copy
import logging

import numpy
import pandas
from pandas.api import types


logger = logging.getLogger(__name__)


def calc_column_stat(df):
    df_size = len(df)
    output_column_names = [
        "column_name", "dtype", "min", "max", "mean", "std", "ratio_na", 
        "ratio_zero", "unique_count", "is_unique"
    ]
    def _calc_stat(column, df_size):
        unique_count = len(get_unique_values(df, column))
        logger.info("...calculating {} (dtype: {})".format(column, df.dtypes[column]))
        return pandas.DataFrame(
            {
                "column_name": [column],
                "dtype": [str(df.dtypes[column])],
                "min": [df[column].min()] if _orderable(df.dtypes[column]) else [pandas.NA],
                "max": [df[column].max()] if _orderable(df.dtypes[column]) else [pandas.NA],
                "mean": [df[column].mean()] if types.is_numeric_dtype(df.dtypes[column]) else [pandas.NA],
                "std": [df[column].std()] if types.is_numeric_dtype(df.dtypes[column]) else [pandas.NA],
                "ratio_na": [pandas.isnull(df[column]).sum() / float(df_size)],
                "ratio_zero": [len(df[df[column]==0]) / float(df_size)]\
                    if types.is_numeric_dtype(df.dtypes[column]) else [pandas.NA],
                "unique_count": [unique_count],
                "is_unique": [unique_count == df_size]
            },
            columns=output_column_names
        )
    return pandas.concat([_calc_stat(c, df_size) for c in df.columns], axis=0)


def to_hashable(value):
    """Convert nested/unhashable values (dict/list/set) into a hashable form."""
    if isinstance(value, dict):
        return tuple(sorted((to_hashable(k), to_hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(to_hashable(v) for v in value)
    if isinstance(value, set):
        return frozenset(to_hashable(v) for v in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def count_duplicated_rows(df):
    """Count duplicated rows; works even when cells contain dict/list values."""
    try:
        return int(df.duplicated().sum())
    except TypeError:
        hashed = df.apply(lambda col: col.map(to_hashable))
        return int(hashed.duplicated().sum())


def get_unique_values(df, columns):
    df_tmp = df[columns].dropna()
    if isinstance(df_tmp, pandas.Series):
        values = list(df_tmp.values)
        try:
            return set(values)
        except TypeError:
            # object columns may contain dict/list; return hashable forms
            return {to_hashable(v) for v in values}
    else:
        rows = [tuple(v) for v in df_tmp.values]
        try:
            return set(rows)
        except TypeError:
            return {tuple(to_hashable(x) for x in row) for row in rows}


def calculate_percentiles_for_df(df, column_list, ratio_list):
    value_dic = {
        **{
            "name": ["min"] +
                     ["{0:.2%}-percentile".format(ratio) for ratio in ratio_list]
                     + ["max"]
        },
        **{
            c: calculate_percentiles(df[column].values, ratio_list)
            for c in column_list
        }
    }
    return pandas.DataFrame(value_dic, columns=["name"] + column_list).set_index("name")


def calculate_percentiles(array, ratio_list):
    n_record0 = array.shape[0]
    valid_array = array[numpy.isfinite(array)]
    n_record = valid_array.shape[0]
    if n_record0 - n_record > 0:
        logger.info("{0} records have missing values (out of {1} records)".format(n_record0 - n_record, n_record0))
    sorted_array = numpy.sort(valid_array)
    ind_list = [_get_ind(n_record, ratio) for ratio in ratio_list]
    return numpy.array([sorted_array[ind] for ind in ind_list])


def calculate_series_stats(column_name, column_dtype, grb_obj):
    if types.is_numeric_dtype(column_dtype):
        return [f"{agg}({column_name})" for agg in ["min", "max", "mean", "std"]], \
            [grb_obj[column_name].min(), grb_obj[column_name].max(),
             grb_obj[column_name].mean(), grb_obj[column_name].std()]
    elif types.is_datetime64_any_dtype(column_dtype):
        return [f"{agg}({column_name})" for agg in ["min", "max"]], \
            [grb_obj[column_name].min(), grb_obj[column_name].max()]
    else:
        return [f"nunique({column_name})"], [grb_obj[column_name].nunique()]


def _get_ind(n_record, ratio):
    return 0 if ratio == 0 else int(math.ceil(n_record * ratio)) - 1


def _orderable(dtype):
    if types.is_numeric_dtype(dtype) or types.is_datetime64_any_dtype(dtype):
        return True
    else:
        return False
