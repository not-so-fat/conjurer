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
                "dtype": df.dtypes[column],
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


def get_unique_values(df, columns):
    df_tmp = df[columns].dropna()
    if isinstance(df_tmp, pandas.Series):
        return set([v for v in df_tmp.values])
    else:
        return set([tuple(v) for v in df_tmp.values])


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


def _get_ind(n_record, ratio):
    return 0 if ratio == 0 else int(math.ceil(n_record * ratio)) - 1


def _orderable(dtype):
    if types.is_numeric_dtype(dtype) or types.is_datetime64_any_dtype(dtype):
        return True
    else:
        return False
