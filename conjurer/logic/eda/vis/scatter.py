import altair as alt

from conjurer.logic.eda.vis import binning


altair_max_rows = 5000


def plot_scatter(df, column_x, column_y, num_bins_x=50, num_bins_y=50, xmin=None, xmax=None, ymin=None, ymax=None):
    if len(df) > altair_max_rows or num_bins_x is not None or num_bins_y is not None:
        plot_heatmap(df, column_x, column_y, num_bins_x, num_bins_y, xmin, xmax, ymin, ymax)
    else:
        plot_points(df, column_x, column_y, xmin, xmax, ymin, ymax)


def plot_heatmap(df, column_x, column_y, num_bins_x=50, num_bins_y=50, xmin=None, xmax=None, ymin=None, ymax=None):
    is_quantitative_x = binning.is_quantitative(df[column_x], num_bins_x)
    is_quantitative_y = binning.is_quantitative(df[column_y], num_bins_y)
    ft_df = binning.create_frequency_table_2d(
        df, column_x, column_y, num_bins_x, num_bins_y, xmin, xmax, ymin, ymax)
    encode_args = _get_encode_args(column_x, column_y, is_quantitative_x, is_quantitative_y)
    return alt.Chart(ft_df).mark_rect().encode(**encode_args).interactive() if is_quantitative_x or is_quantitative_y \
        else alt.Chart(ft_df).mark_circle().encode(**encode_args).interactive()


def _get_encode_args(column_x, column_y, is_quantitative_x, is_quantitative_y):
    x_tooltip = ["{}_lb".format(column_x), "{}_ub".format(column_x)] if is_quantitative_x else [column_x]
    y_tooltip = ["{}_lb".format(column_y), "{}_ub".format(column_y)] if is_quantitative_y else [column_y]
    common_args = dict(
        size="frequency",
        color=alt.Color("frequency", scale=alt.Scale(scheme="greys")),
        tooltip=x_tooltip + y_tooltip + ["frequency"]
    )
    x_args = dict(
        x=alt.X("_lb".format(column_x), title=column_x),
        x2="_ub".format(column_x)
    ) if is_quantitative_x else dict(x=column_x)
    y_args = dict(
        y=alt.X("_lb".format(column_y), title=column_y),
        y2="_ub".format(column_y)
    ) if is_quantitative_y else dict(y=column_y)
    return {**common_args, **x_args, **y_args}


def plot_points(df, column_x, column_y, xmin=None, xmax=None, ymin=None, ymax=None):
    return alt.Chart(df).mark_circle().encode(
        x=alt.X(column_x, domain=[xmin, xmax]),
        y=alt.Y(column_y, domain=[ymin, ymax]),
        tooltip=[column_x, column_y]
    ).interactive()
