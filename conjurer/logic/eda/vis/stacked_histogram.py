import numpy
import pandas
from plotly import graph_objs
from plotly.subplots import make_subplots
from plotly.offline import iplot

from conjurer.logic.eda.vis import stacked_bar
from conjurer.logic.eda.vis import histogram


def plot_stacked_histogram(array_list, data_type="numeric", num_bins=50, normalize=False,
                           minv=None, maxv=None, name_list=None, xname=None, stack=True, layout={}):
    """
    Plot stacked histogram of list of arrays. (only available for numeric)
    If normalize option is selected, plotted value becomes ratio of count for the array.
    (sum of values for all bins become 1)
    """

    name_list = name_list or ["input{}".format(i) for i in range(len(array_list))]
    if data_type == "numeric" or "datetime":
        minv = minv or min([array.min() for array in array_list])
        maxv = maxv or max([array.max() for array in array_list])
        lower_list, upper_list, bin_size = histogram.get_bin_config(minv, maxv, num_bins)
        count_df = pandas.DataFrame({
            name_list[i]: histogram.get_continuous_value_counts(array, lower_list, upper_list)
            for i, array in enumerate(array_list)
        })
        if normalize:
            count_df = _normalize(count_df)
        trace_list = [histogram.format_trace_for_histogram_continuous(count_df[name], lower_list, upper_list, bin_size)
                      for name in name_list]
        for name, trace in zip(name_list, trace_list):
            trace["name"] = name
        if stack:
            fig = graph_objs.Figure(data=trace_list, layout=_get_layout(normalize, xname))
        else:
            fig = make_subplots(rows=len(array_list), cols=1, shared_xaxes=True)
            for i, trace in enumerate(trace_list):
                fig.add_trace(trace, row=i + 1, col=1)
            fig.update_layout(height=300 * len(array_list))
        fig.update_layout(**layout)
        iplot(fig, show_link=False)
    else:
        vcounts = pandas.Series(numpy.concatenate(array_list)).value_counts()
        index = vcounts[:num_bins].index
        count_df = pandas.DataFrame({
            name_list[i]: histogram.get_categorical_value_counts(array, index)
            for i, array in enumerate(array_list)
        })
        if normalize:
            count_df = _normalize(count_df)
        stacked_bar.plot_stacked_bars(
            count_df.index, [count_df[name].values for name in name_list], name_list=name_list,
            xname=xname or "value", yname="ratio" if normalize else "frequency", single=stack, layout=layout)


def _normalize(df):
    return df.apply(lambda x: x / x.sum(), axis=1)


def _get_layout(normalize, xname):
    xaxis = dict(title=xname or "value")
    return graph_objs.Layout(
        title="Stacked Histogram",
        xaxis=xaxis,
        yaxis=dict(title="ratio" if normalize else "frequency"),
        barmode="stack",
        hovermode="x"
    )
