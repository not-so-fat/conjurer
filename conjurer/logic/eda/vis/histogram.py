import logging
from datetime import datetime

import numpy
import pandas
from pandas.api import types
from plotly import graph_objs
from plotly.offline import iplot


logger = logging.getLogger(__name__)


def plot_histogram(array, num_bins=50, normalize=False, minv=None, maxv=None, title=None, layout={}):
    trace, xaxis = \
        _generate_graph_for_integer(array, num_bins, normalize, minv, maxv) if types.is_integer_dtype(array.dtype) \
            else _generate_graph_for_numeric(array, num_bins, normalize, minv, maxv) \
            if types.is_numeric_dtype(array.dtype) \
            else _generate_graph_for_datetime(array, num_bins, normalize, minv, maxv) \
            if types.is_datetime64_any_dtype(array.dtype) \
            else _generate_graph_for_categorical(array, num_bins, normalize)
    if trace is not None:
        layout0 = _get_layout(normalize, title, xaxis)
        fig = graph_objs.Figure(data=[trace], layout=layout0)
        fig.update_layout(**layout)
        iplot(fig, show_link=False)


def plot_histogram_for_stats(df, stat_df, num_bins=50, normalize=False, minv=None, maxv=None):
    for ind, column in enumerate(df.columns):
        logger.info("...histogram for {0}".format(column))
        vmin, vmax, unique_count = \
            stat_df[stat_df["column_name"] == column][
                ["min", "max", "unique_count"]].values.tolist()[0]
        minv0 = minv or vmin
        maxv0 = maxv or vmax
        plot_histogram(df[column].values, num_bins, normalize, minv0, maxv0, column)


def _generate_graph_for_numeric(array, num_bins, normalize, minv=None, maxv=None):
    if numpy.array(array[~numpy.isnan(array)]).size == 0:
        logger.warning("skip histogram because all values are null")
        return None, None
    minv = minv or numpy.nanmin(array)
    maxv = maxv or numpy.nanmax(array)
    if maxv == minv:
        logger.warning("skip histogram because min == max")
        return None, None
    lower_list, upper_list, bin_size = get_bin_config(minv, maxv, num_bins)
    count_array = get_continuous_value_counts(array, lower_list, upper_list)
    if normalize:
        count_array = count_array / count_array.sum()
    hist_trace = format_trace_for_histogram_continuous(count_array, lower_list, upper_list, bin_size)
    xaxis = dict(range=[minv, maxv])
    return hist_trace, xaxis


def _generate_graph_for_datetime(array, num_bins, normalize, minv=None, maxv=None):
    if numpy.array(array[~numpy.isnan(array)]).size == 0:
        logger.warning("skip histogram because all values are null")
        return None, None
    _convert = lambda v: None if v is None else None if numpy.isnat(_to_datetime64(v)) else _to_datetime64(v)
    minv = _convert(minv)
    maxv = _convert(maxv)
    return _generate_graph_for_numeric(array, num_bins, normalize, minv, maxv)


def _generate_graph_for_integer(array, num_bins, normalize, minv, maxv):
    if numpy.array(array[~numpy.isnan(array)]).size == 0:
        logger.warning("skip histogram because all values are null")
        return None, None
    vcounts = pandas.Series(array).value_counts()
    if len(vcounts) <= num_bins:
        if normalize:
            total = vcounts.values.sum()
            vcounts.values = [v / total for v in vcounts.values]
        return graph_objs.Bar(x=vcounts.index, y=vcounts.values), {}
    else:
        return _generate_graph_for_numeric(array, num_bins, normalize, minv, maxv)


def _generate_graph_for_categorical(array, num_bins, normalize):
    vcounts = pandas.Series(array).value_counts()
    if len(vcounts) > num_bins:
        index = vcounts[:num_bins].index
        vcounts = get_categorical_value_counts(array, index)
    if normalize:
        total = vcounts.values.sum()
        vcounts.values = [v / total for v in vcounts.values]
    return graph_objs.Bar(x=vcounts.index, y=vcounts.values), {}


def format_trace_for_histogram_continuous(count_array, lower_list, upper_list, bin_size):
    lower_list = _datetime64_to_datetime_array(lower_list)
    upper_list = _datetime64_to_datetime_array(upper_list)
    if isinstance(bin_size, numpy.timedelta64):
        bin_size = bin_size / numpy.timedelta64(1, "ms")
    return graph_objs.Bar(
        x=lower_list,
        y=count_array,
        offset=0,
        hovertemplate="%{text}: %{y}",
        text=["[{}, {})".format(l, u) for l, u in zip(lower_list[:-1], upper_list[:-1])] + \
             ["[{}, {}]".format(lower_list[-1], upper_list[-1])],
        width=bin_size
    )


def format_trace_for_histogram_categorical(count_array, value_array):
    return graph_objs.Bar(
        x=value_array,
        y=count_array,
        hovertemplate="%{x}: %{y}"
    )


def _get_layout(normalize, name, xaxis):
    xaxis["title"] = name or "value"
    return graph_objs.Layout(
        title="Histogram of {}".format(name) if name else "Histogram",
        xaxis=xaxis,
        yaxis=dict(title="ratio" if normalize else "frequency"),
        hovermode="x"
    )


def get_bin_config(minv, maxv, num_bins):
    bin_size = (maxv - minv) / num_bins
    lower_list = [minv + i * bin_size for i in range(num_bins)]
    upper_list = [minv + (i + 1) * bin_size for i in range(num_bins-1)] + [maxv]
    return lower_list, upper_list, bin_size


def get_continuous_value_counts(array, lower_list, upper_list):
    def _count(array, lower, upper, is_last_bin):
        return numpy.array(array[(array >= lower) & (array <= upper)]).size if is_last_bin else \
            numpy.array(array[(array >= lower) & (array < upper)]).size

    return numpy.array([_count(array, lower_list[i], upper_list[i], False) for i in range(len(lower_list) - 1)] + \
                       [_count(array, lower_list[-1], upper_list[-1], True)])


def get_categorical_value_counts(array, index=None):
    vcounts = pandas.Series(array).value_counts()
    if index is not None:
        other_counts = vcounts[~vcounts.index.isin(index)].values.sum()
        vcounts = vcounts[index]
        vcounts["OTHER"] = other_counts
    return vcounts


def _datetime64_to_datetime_array(array):
    if isinstance(array[0], numpy.datetime64):
        array = numpy.array([_datetime64_to_datetime(v) for v in array])
    return array


def _datetime64_to_datetime(v):
    return pandas.Timestamp(_round_datetime(v)).to_pydatetime()


def _to_datetime64(v):
    if isinstance(v, datetime):
        return numpy.datetime64(v)
    elif isinstance(v, pandas.Timestamp):
        if not numpy.isnat(v):
            return v.to_datetime64().astype("datetime64")
    else:
        return v


def _round_datetime(v):
    """
    plot.ly cannot support time object which has less than micro second
    """
    return numpy.datetime64(v, "us")
