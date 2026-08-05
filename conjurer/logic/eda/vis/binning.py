import logging

import numpy
import pandas
from pandas.api import types


logger = logging.getLogger(__name__)

FREQUENCY_CNAME="# of records"
RATIO_CNAME="ratio of records"


def create_frequency_table(series, num_bins=50, minv=None, maxv=None):
    if is_quantitative(series, num_bins):
        return count_frequency_quantitative(series, num_bins, minv, maxv)
    else:
        return count_frequency_categorical(series, num_bins)


def is_quantitative(series, num_bins):
    if types.is_integer_dtype(series.dtype):
        return len(series.value_counts()) >= num_bins
    else:
        return types.is_numeric_dtype(series.dtype) or types.is_datetime64_any_dtype(series.dtype)


def count_frequency_quantitative(series, num_bins, minv, maxv):
    name = series.name or "value"
    bins = create_bins_quantitative(series, num_bins, minv, maxv)
    # Use Series, not .values — tz-aware datetimes lose tz in the ndarray.
    count_array = get_continuous_value_counts(series, bins)
    ratio_array = count_array / count_array.sum()
    return pandas.DataFrame({
        "{}_lb".format(name): [b.lb for b in bins],
        "{}_ub".format(name): [b.ub for b in bins],
        FREQUENCY_CNAME: count_array,
        RATIO_CNAME: ratio_array
    })


def count_frequency_categorical(series, num_bins):
    name = series.name or "value"
    vcounts = get_categorical_value_counts(series, num_bins)
    total = vcounts.values.sum()
    return pandas.DataFrame({
        name: list(vcounts.index),
        FREQUENCY_CNAME: list(vcounts.values),
        RATIO_CNAME: [v / total for v in vcounts.values]
    })


def create_bins_quantitative(series, num_bins, minv=None, maxv=None):
    array = series.values
    if numpy.array(array[~numpy.isnan(array)]).size == 0:
        raise BinCreationError("all values are null")
    minv = minv if minv is not None else series.min()  # cannot use "or" for the case minv==0
    maxv = maxv if maxv is not None else series.max()  # cannot use "or" for the case maxv==0
    if maxv == minv:
        raise BinCreationError("min == max")
    lb_list, ub_list, bin_size = get_bin_config(minv, maxv, num_bins)
    return [QuantitativeBin(lb, ub, False) for lb, ub in zip(lb_list[:-1], ub_list[:-1])] + \
           [QuantitativeBin(lb_list[-1], ub_list[-1], True)]


def create_bins_categorical(series, num_bins):
    vcounts = series.value_counts()
    bins = [CategoricalBin([v]) for v in vcounts[:num_bins].index]
    if len(vcounts) > num_bins:
        bins.append(CategoricalBin([v for v in vcounts[num_bins:].index]))
    return bins


def get_bin_config(minv, maxv, num_bins):
    bin_size = (maxv - minv) / num_bins
    lower_list = [minv + i * bin_size for i in range(num_bins)]
    upper_list = [minv + (i + 1) * bin_size for i in range(num_bins-1)] + [maxv]
    return lower_list, upper_list, bin_size


def get_continuous_value_counts(series, bins):
    return numpy.array([len(b.filter(series)) for b in bins])


def get_categorical_value_counts(series, num_bins=None):
    """
    Value counts for categorical histogram bins.

    When there are more unique values than ``num_bins``, keep the top bins and
    fold the rest into ``OTHER``. Use position-based slicing (not label
    indexing) so object columns with unhashable values (dict/list) work —
    ``Series.loc[Index[dict, ...]]`` raises ``TypeError: unhashable type: 'dict'``.
    """
    try:
        vcounts = series.value_counts()
    except TypeError:
        # Some pandas versions cannot hash nested object cells in value_counts.
        labels = series.map(_categorical_label)
        labels.name = series.name
        vcounts = labels.value_counts()
    if num_bins is not None and len(vcounts) > num_bins:
        top = vcounts.iloc[:num_bins]
        other_counts = vcounts.iloc[num_bins:].sum()
        vcounts = pandas.concat([top, pandas.Series([other_counts], index=["OTHER"])])
    return vcounts


def _categorical_label(value):
    """Stable, hashable label for nested/unhashable object cells."""
    try:
        hash(value)
        return value
    except TypeError:
        return str(value)


def create_frequency_table_2d(df, column_x, column_y, num_bins_x=50, num_bins_y=50,
                              xmin=None, xmax=None, ymin=None, ymax=None):
    is_quantitative_x = is_quantitative(df[column_x], num_bins_x)
    is_quantitative_y = is_quantitative(df[column_y], num_bins_y)
    bins_x = create_bins(df[column_x], num_bins_x, xmin, xmax, is_quantitative_x)
    bins_y = create_bins(df[column_y], num_bins_y, ymin, ymax, is_quantitative_y)
    frequency_list = []
    for bin_x in bins_x:
        for bin_y in bins_y:
            frequency_list.append(
                _count_2d(df, column_x, column_y, bin_x, bin_y)
            )
    count_df = pandas.concat(frequency_list, ignore_index=True)
    count_df[RATIO_CNAME] = count_df[FREQUENCY_CNAME] / count_df[FREQUENCY_CNAME].sum()
    return count_df


def create_bins(series, num_bins, minv, maxv, is_quantitative):
    if is_quantitative:
        return create_bins_quantitative(series, num_bins, minv, maxv)
    else:
        return create_bins_categorical(series, num_bins)


def _count_2d(df, column_x, column_y, bin_x, bin_y):
    return pandas.DataFrame({
        **bin_x.bin_info(column_x),
        **bin_y.bin_info(column_y),
        **{
            FREQUENCY_CNAME: [len(bin_y.filter(bin_x.filter(df, column_x), column_y))]
        }
    })


class QuantitativeBin(object):
    def __init__(self, lb, ub, is_last_bin):
        self.lb = lb
        self.ub = ub
        self.is_last_bin = is_last_bin

    def filter(self, df, column_name=""):
        if column_name:
            return df[(df[column_name]>=self.lb)&(df[column_name]<=self.ub)] if self.is_last_bin \
                else df[(df[column_name]>=self.lb)&(df[column_name]<self.ub)]
        else:
            return df[(df>=self.lb)&(df<=self.ub)] if self.is_last_bin \
                else df[(df>=self.lb)&(df<self.ub)]

    def bin_info(self, column_name=""):
        return {
            "{}_lb".format(column_name): [self.lb],
            "{}_ub".format(column_name): [self.ub]
        }


class CategoricalBin(object):
    def __init__(self, values):
        self.values = values

    def filter(self, df, column_name=""):
        return df[df[column_name].isin(self.values)] if column_name else df[df.isin(self.values)]

    def bin_info(self, column_name=""):
        return {
            column_name: self.values,
        } if len(self.values) == 1 else {column_name: ["OTHERS"]}


class BinCreationError(Exception):
    def __init__(self, message):
        self.message = message
