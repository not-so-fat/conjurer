from plotly import graph_objs
from plotly.offline import iplot


def plot_scatter(xarray, yarray, xname=None, yname=None, mode="markers", same_scale=False,
                 xerror=None, yerror=None, layout={}):
    xmin = xarray.min() if xerror is None else (xarray - xerror).min()
    xmax = xarray.max() if xerror is None else (xarray + xerror).max()
    ymin = yarray.min() if yerror is None else (yarray - yerror).min()
    ymax = yarray.max() if yerror is None else (yarray + yerror).max()
    if same_scale:
        xmin = ymin = min([xmin, ymin])
        xmax = ymax = max([xmax, ymax])
    trace = graph_objs.Scatter(
        x=xarray,
        y=yarray,
        error_x=dict(type="data", array=xerror),
        error_y=dict(type="data", array=yerror),
        mode=mode
    )
    fig = graph_objs.Figure(data=[trace], layout=_get_layout(xname, yname, xmin, xmax, ymin, ymax, same_scale))
    fig.update_layout(**layout)
    iplot(fig, show_link=False)


def plot_scatter_multiple(xarray_list, yarray_list, xname=None, yname=None, name_list=None,
                          mode="markers", same_scale=False, layout={}):
    xmin = min([x_array.min() for x_array in xarray_list])
    xmax = max([x_array.max() for x_array in xarray_list])
    ymin = min([y_array.min() for y_array in yarray_list])
    ymax = max([y_array.max() for y_array in yarray_list])
    if same_scale:
        xmin = ymin = min([xmin, ymin])
        xmax = ymax = max([xmax, ymax])
    name_list = name_list or ["input-{}".format(i) for i in range(len(xarray_list))]
    trace_list = [
        graph_objs.Scatter(
            x=x_array,
            y=y_array,
            mode=mode,
            name=name
        )
        for x_array, y_array, name in zip(xarray_list, yarray_list, name_list)
    ]
    fig = graph_objs.Figure(data=trace_list, layout=_get_layout(xname, yname, xmin, xmax, ymin, ymax, same_scale))
    fig.update_layout(**layout)
    iplot(fig, show_link=False)


def _get_layout(xname, yname, xmin, xmax, ymin, ymax, same_scale):
    xname = xname or "x"
    yname = yname or "y"
    xaxis = dict(title=xname, range=[xmin, xmax])
    yaxis = dict(title=yname, range=[ymin, ymax])
    return graph_objs.Layout(
        title="{} vs {}".format(yname, xname),
        xaxis=xaxis,
        yaxis=yaxis,
        height=800 if same_scale else 400,
        width=800,
        hovermode="x"
    )