import logging

import altair as alt
from pandas.api import types

from conjurer.logic.eda.vis import binning


logger = logging.getLogger(__name__)


def plot_histogram(values, num_bins=50, normalize=False, minv=None, maxv=None):
    try:
        bin_df = binning.create_frequency_table(values, num_bins, minv, maxv)
    except Exception as e:
        raise e
    return plot_frequency_numeric(bin_df, normalize) if len(bin_df.columns) > 3\
        else plot_frequency_category(bin_df, normalize)


def plot_histogram_for_stats(df, stat_df, num_bins=50, normalize=False):
    for ind, column in enumerate(df.columns):
        minv, maxv = \
            stat_df[stat_df["column_name"] == column][
                ["min", "max"]].values.tolist()[0]
        try:
            plot_histogram(df[column], num_bins, normalize, minv, maxv).display()
        except binning.BinCreationError as e:
            logger.info("Histogram for {} was skipped: {}".format(column, e.message))
            pass


def plot_frequency_numeric(df, normalize, xname=None):
    column_lb = df.columns[0]
    column_ub = df.columns[1]
    xname = xname or column_lb.replace("_lb", "")
    column_y = binning.RATIO_CNAME if normalize else binning.FREQUENCY_CNAME
    y_args = dict(axis=alt.Axis(format="%")) if normalize else {}
    return alt.Chart(df).mark_bar().encode(
        alt.X(column_lb, bin="binned", axis=alt.Axis(title=xname)),
        x2=column_ub,
        y=alt.Y("{}:Q".format(column_y), **y_args),
        tooltip=[column_lb, column_ub, column_y]
    ).properties(height=200, width=800, title="Histogram of {}".format(xname)).interactive()


def plot_frequency_category(df, normalize, xname=None):
    x_args = {} if types.is_integer_dtype(df.dtypes[0]) else dict(sort="-y")
    y_args = dict(axis=alt.Axis(format="%")) if normalize else {}
    column_y = binning.RATIO_CNAME if normalize else binning.FREQUENCY_CNAME
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("{}:N".format(df.columns[0]), **x_args),
        y=alt.Y("{}:Q".format(column_y), **y_args)
    ).properties(height=200, width=800, title="Histogram of {}".format(df.columns[0])).interactive()
