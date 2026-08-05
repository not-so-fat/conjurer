"""Altair charts for aggregated X vs Y (plot_aggy)."""

import altair as alt
from pandas.tseries.frequencies import to_offset

from conjurer.logic.eda.vis import aggregation


ALLOWED_MARKS = ("bar", "line")


def plot_aggy(
    df,
    column_x,
    column_y,
    agg="sum",
    freq=None,
    num_bins=50,
    mark="bar",
    fill_empty=None,
    xmin=None,
    xmax=None,
):
    if mark not in ALLOWED_MARKS:
        raise ValueError("mark must be one of {}; got {!r}".format(ALLOWED_MARKS, mark))

    agg_df = aggregation.create_aggregation_table(
        df,
        column_x,
        column_y,
        agg=agg,
        freq=freq,
        num_bins=num_bins,
        fill_empty=fill_empty,
        xmin=xmin,
        xmax=xmax,
    )
    title = "{}({}) by {}".format(agg, column_y, column_x)
    if freq is not None:
        title = "{} (freq={})".format(title, freq)

    y_enc = alt.Y("{}:Q".format(aggregation.AGG_VALUE_CNAME), title="{}({})".format(agg, column_y))

    if freq is not None:
        plot_df, bar_kwargs, line_kwargs = _calendar_encodings(
            agg_df, column_x, y_enc, freq, mark
        )
    else:
        plot_df, bar_kwargs, line_kwargs = _equal_width_encodings(
            agg_df, column_x, y_enc, mark
        )

    props = dict(height=200, width=800, title=title)
    if mark == "line":
        base = alt.Chart(plot_df).encode(**line_kwargs)
        # Same encoding for stroke and points so they share vertices
        return (base.mark_line() + base.mark_point()).properties(**props).interactive()

    return alt.Chart(plot_df).mark_bar().encode(**bar_kwargs).properties(**props).interactive()


def _calendar_encodings(agg_df, column_x, y_enc, freq, mark):
    """Bars span the full period [start, start+freq); lines use period midpoints."""
    plot_df = agg_df.copy()
    lb = column_x
    ub = "{}_ub".format(column_x)
    offset = to_offset(freq)
    plot_df[ub] = plot_df[lb] + offset

    if mark == "bar":
        return plot_df, dict(
            x=alt.X("{}:T".format(lb), bin="binned", title=column_x),
            x2=ub,
            y=y_enc,
            tooltip=[lb, ub, aggregation.AGG_VALUE_CNAME],
        ), None

    mid = "{}_mid".format(column_x)
    plot_df[mid] = plot_df[lb] + (plot_df[ub] - plot_df[lb]) / 2
    return plot_df, None, dict(
        x=alt.X("{}:T".format(mid), title=column_x),
        y=y_enc,
        tooltip=[lb, ub, mid, aggregation.AGG_VALUE_CNAME],
    )


def _equal_width_encodings(agg_df, column_x, y_enc, mark):
    lb = "{}_lb".format(column_x)
    ub = "{}_ub".format(column_x)
    if mark == "bar":
        return agg_df, dict(
            x=alt.X(lb, bin="binned", axis=alt.Axis(title=column_x)),
            x2=ub,
            y=y_enc,
            tooltip=[lb, ub, aggregation.AGG_VALUE_CNAME],
        ), None

    # Line/point need a single x; x/x2 binned spans put marks at different places
    plot_df = agg_df.copy()
    mid = "{}_mid".format(column_x)
    plot_df[mid] = plot_df[lb] + (plot_df[ub] - plot_df[lb]) / 2
    return plot_df, None, dict(
        x=alt.X(mid, title=column_x),
        y=y_enc,
        tooltip=[lb, ub, mid, aggregation.AGG_VALUE_CNAME],
    )
