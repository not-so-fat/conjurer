import altair as alt
from pandas.api import types


def add_ruler_vertical(lines, column_name_x, column_name_y):
    type_ind_x = ":Q" if types.is_numeric_dtype(lines.data[column_name_x].dtype) else ":N"
    type_ind_y = ":Q" if types.is_numeric_dtype(lines.data[column_name_y].dtype) else ":N"
    column_x = "{}{}".format(column_name_x, type_ind_x)
    column_y = "{}{}".format(column_name_y, type_ind_y)
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=[column_x], empty='none')
    selectors = alt.Chart(lines.data).mark_point().encode(
        x=column_x,
        opacity=alt.value(0),
    ).add_selection(nearest)
    text = lines.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, column_y, alt.value(' '))
    )
    rules = alt.Chart(lines.data).mark_rule(color='gray').encode(x=column_x).transform_filter(nearest)
    return selectors, text, rules

