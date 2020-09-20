import altair as alt


def plot_line(df, column_x, column_y, column_yerror=None):
    encode_args = dict(x=column_x, y=column_y)
    if column_yerror:
        encode_args["yError"] = column_yerror
    base = alt.Chart(df).encode(**encode_args)
    c1 = base.mark_line()
    if column_yerror:
        c2 = base.mark_errorband()
        return (c1 + c2).interactive()
    else:
        return c1.interactive()
