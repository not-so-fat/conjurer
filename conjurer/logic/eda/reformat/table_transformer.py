import itertools
import copy

import numpy
import pandas
from numpy import random

COLUMN_NEW_ID = "__new_column_for_composite_keys__"


def transform_columns_to_rows(df, column_group, suffixs):
    converted_columns = list(itertools.chain(*[["{}{}".format(c, s) for c in column_group] for s in suffixs]))
    base_columns = [c for c in df.columns if c not in converted_columns]
    ret_df = pandas.concat([
        df[base_columns + ["{}{}".format(c, s) for c in column_group]].rename(
            columns={
                "{}{}".format(c, s): c
                for c in column_group
            }
        )
        for s in suffixs
    ])
    return ret_df.dropna(how="all", subset=column_group)


def random_split_by_ids(df, id_columns, training_ratio):
    ret_df = _add_unique_id(df, id_columns)
    num_new_ids = ret_df[COLUMN_NEW_ID].max() + 1
    training_indices, validation_indices = _get_split_index(num_new_ids, training_ratio)
    return ret_df[ret_df[COLUMN_NEW_ID].isin(training_indices)].drop(columns=[COLUMN_NEW_ID]), \
           ret_df[ret_df[COLUMN_NEW_ID].isin(validation_indices)].drop(columns=[COLUMN_NEW_ID])


def transform_cl_target(series, positive_values, negative_values, positive_label=1, negative_label=0):
    ret_series = pandas.Series(data=[numpy.NaN]*len(series), index=series.index, dtype=series.dtype)
    ret_series[series.isin(positive_values)] = positive_label
    ret_series[series.isin(negative_values)] = negative_label
    return ret_series


def _add_unique_id(df, id_columns):
    id_values = list(set(zip(*[df[c] for c in id_columns])))
    newid_dict = {
        v: i
        for i, v in enumerate(id_values)
    }
    ret_df = copy.deepcopy(df)
    ret_df[COLUMN_NEW_ID] = [newid_dict[v] for v in zip(*[df[c] for c in id_columns])]
    return ret_df


def _get_split_index(num_new_ids, training_ratio):
    indices = list(range(num_new_ids))
    random.shuffle(indices)
    num_training = int(training_ratio * num_new_ids)
    return indices[:num_training], indices[num_training:]
