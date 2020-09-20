import logging

import altair as alt
from pandas.api import types

from conjurer.logic.eda.vis import (
    binned_bar,
    binning
)


logger = logging.getLogger(__name__)


def plot_histogram(values, num_bins=50, normalize=False, minv=None, maxv=None):
    try:
        bin_df = binning.create_frequency_table(values, num_bins, minv, maxv)
    except Exception as e:
        raise e
    column_y = "ratio" if normalize else "frequency"
    if len(bin_df.columns) > 3:
        return binned_bar.plot_bar_with_binned(bin_df, bin_df.columns[0], bin_df.columns[1], column_y)
    else:
        args = {} if types.is_integer_dtype(values.dtype) else dict(sort="-y")
        return alt.Chart(bin_df).mark_bar().encode(
            x=alt.X("{}:N".format(bin_df.columns[0]), **args),
            y=column_y
        ).properties(height=200, width=800).interactive()


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
