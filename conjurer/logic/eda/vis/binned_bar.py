import altair as alt


def plot_bar_with_binned(df, column_lb, column_ub, column_y, xname=None):
    xname = xname or column_lb.replace("_lb", "")
    return alt.Chart(df).mark_bar().encode(
        alt.X(column_lb, bin="binned", axis=alt.Axis(title=xname)),
        x2=column_ub,
        y=alt.Y("{}:Q".format(column_y)),
        tooltip=[column_lb, column_ub, column_y]
    ).properties(height=200, width=800).interactive()


