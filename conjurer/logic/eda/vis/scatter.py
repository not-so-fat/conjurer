import altair as alt

from conjurer.logic.eda.vis import binning


altair_max_rows = 5000


def plot_scatter(df, column_x, column_y, **kwargs):
    if len(df) > altair_max_rows:
        return plot_heatmap(df, column_x, column_y, **kwargs)
    else:
        return plot_points(df, column_x, column_y, **kwargs)


def plot_heatmap(df, column_x, column_y, num_bins_x=50, num_bins_y=50, xmin=None, xmax=None, ymin=None, ymax=None):
    is_quantitative_x = binning.is_quantitative(df[column_x], num_bins_x)
    is_quantitative_y = binning.is_quantitative(df[column_y], num_bins_y)
    ft_df = binning.create_frequency_table_2d(
        df, column_x, column_y, num_bins_x, num_bins_y, xmin, xmax, ymin, ymax)
    ft_df = ft_df[ft_df[binning.FREQUENCY_CNAME] > 0]
    encode_args = _get_encode_args(column_x, column_y, is_quantitative_x, is_quantitative_y)
    title = "{} vs {} (Heatmap)".format(column_y, column_x)
    chart = alt.Chart(ft_df).encode(**encode_args).properties(title=title)\
        .configure_legend(orient="right", columns=1).interactive()
    return chart.mark_rect() if is_quantitative_x or is_quantitative_y else chart.mark_circle()


def _get_encode_args(column_x, column_y, is_quantitative_x, is_quantitative_y):
    x_tooltip = ["{}_lb".format(column_x), "{}_ub".format(column_x)] if is_quantitative_x else [column_x]
    y_tooltip = ["{}_lb".format(column_y), "{}_ub".format(column_y)] if is_quantitative_y else [column_y]
    common_args = dict(
        size=binning.FREQUENCY_CNAME,
        color=alt.Color(binning.FREQUENCY_CNAME, scale=alt.Scale(scheme="greys")),
        tooltip=x_tooltip + y_tooltip + [binning.FREQUENCY_CNAME]
    )
    x_args = dict(
        x=alt.X("{}_lb".format(column_x), title=column_x),
        x2="{}_ub".format(column_x)
    ) if is_quantitative_x else dict(x=column_x)
    y_args = dict(
        y=alt.X("{}_lb".format(column_y), title=column_y),
        y2="{}_ub".format(column_y)
    ) if is_quantitative_y else dict(y=column_y)
    return {**common_args, **x_args, **y_args}


def plot_points(df, column_x, column_y, xmin=None, xmax=None, ymin=None, ymax=None):
    title = "{} vs {} (Scatter)".format(column_y, column_x)
    return alt.Chart(df).mark_circle().encode(
        x=alt.X(column_x, **_get_axis_args(df, column_x, xmin, xmax)),
        y=alt.Y(column_y, **_get_axis_args(df, column_y, ymin, ymax)),
        tooltip=[column_x, column_y]
    ).properties(title=title).interactive()


def _get_axis_args(df, column, minv, maxv):
    if minv is not None or maxv is not None:
        minv = minv if minv is not None else df[column].min()
        maxv = maxv if maxv is not None else df[column].max()
        return dict(domain=[minv, maxv])
    else:
        return {}
