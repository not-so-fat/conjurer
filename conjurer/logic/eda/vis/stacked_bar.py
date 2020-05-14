import numpy
from plotly import graph_objs
from plotly.subplots import make_subplots
from plotly.offline import iplot
from matplotlib import cm


def _get_trace(y_array, x_array, name, color):
    return graph_objs.Bar(
        x=x_array,
        y=y_array,
        marker=dict(color=color),
        text=name,
        name=name,
        hovertemplate="(%{x}, %{y})"
    )


def plot_stacked_bars(xarray, yarray_list, name_list=None, xname=None, yname=None, single=True, layout={}):
    name_list = name_list or ["input{}".format(i) for i in range(len(yarray_list))]
    colors = ["rgb({0},{1},{2})".format(item[0], item[1], item[2])
              for item in cm.rainbow(numpy.linspace(0, 1, len(yarray_list)), None, True)]
    trace_list = [
        _get_trace(yarray_list[i], xarray, name_list[i], colors[i])
        for i in range(len(yarray_list))
    ]
    if single:
        fig = graph_objs.Figure(data=trace_list, layout=_get_layout(xname, yname))
    else:
        fig = make_subplots(rows=len(yarray_list), cols=1, shared_xaxes=True)
        for i, trace in enumerate(trace_list):
            fig.add_trace(trace, row=i+1, col=1)
        fig.update_layout(height=300*len(yarray_list))
    fig.update_layout(**layout)
    iplot(fig, show_link=False)


def _get_layout(xname, yname):
    xaxis = dict(title=xname or "x")
    return graph_objs.Layout(
        xaxis=xaxis,
        yaxis=dict(title=yname or "y"),
        barmode="relative",
        hovermode="x"
    )
