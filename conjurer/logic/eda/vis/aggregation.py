"""Build aggregation tables for plot_aggy (bin/group X, aggregate Y)."""

import numpy
import pandas
from pandas.api import types

from conjurer.logic.eda.vis import binning


ALLOWED_AGGS = ("sum", "count", "mean", "min", "max", "median")
AGG_VALUE_CNAME = "agg_value"


def create_aggregation_table(
    df,
    column_x,
    column_y,
    agg="sum",
    freq=None,
    num_bins=50,
    fill_empty=None,
    xmin=None,
    xmax=None,
):
    agg = _validate_agg(agg)
    if column_x not in df.columns:
        raise ValueError("column_x {!r} not in dataframe".format(column_x))
    if column_y not in df.columns:
        raise ValueError("column_y {!r} not in dataframe".format(column_y))

    work = df[[column_x, column_y]].copy()
    work = work.dropna(subset=[column_x])
    if len(work) == 0:
        raise ValueError("no non-null values in column_x {!r}".format(column_x))

    if xmin is not None:
        work = work[work[column_x] >= xmin]
    if xmax is not None:
        work = work[work[column_x] <= xmax]
    if len(work) == 0:
        raise ValueError("no rows left after applying xmin/xmax filters")

    if fill_empty is None:
        fill_empty = freq is not None

    if agg != "count" and not types.is_numeric_dtype(work[column_y]):
        raise ValueError(
            "column_y {!r} must be numeric for agg={!r} (use agg='count' for non-numeric)".format(
                column_y, agg
            )
        )

    if freq is not None:
        return _aggregate_calendar(work, column_x, column_y, agg, freq, fill_empty, xmin, xmax)
    return _aggregate_equal_width(work, column_x, column_y, agg, num_bins, fill_empty, xmin, xmax)


def _validate_agg(agg):
    if agg not in ALLOWED_AGGS:
        raise ValueError("agg must be one of {}; got {!r}".format(ALLOWED_AGGS, agg))
    return agg


def _assert_x_dtype(series, freq):
    if freq is not None:
        if not types.is_datetime64_any_dtype(series.dtype):
            raise ValueError(
                "column_x must be datetime-like when freq is set; got dtype={}".format(series.dtype)
            )
        return
    if not (types.is_numeric_dtype(series.dtype) or types.is_datetime64_any_dtype(series.dtype)):
        raise ValueError(
            "column_x must be numeric or datetime dtype when freq is None; "
            "got dtype={} (string/object/category not supported in v1)".format(series.dtype)
        )


def _empty_fill_value(agg):
    return 0 if agg in ("sum", "count") else numpy.nan


def _aggregate_calendar(work, column_x, column_y, agg, freq, fill_empty, xmin, xmax):
    _assert_x_dtype(work[column_x], freq)
    grouper = pandas.Grouper(key=column_x, freq=freq)
    counts = work.groupby(grouper).size()
    if agg == "count":
        series = counts.copy()
    else:
        series = work.groupby(grouper)[column_y].agg(agg)

    series = series[series.index.notna()]
    counts = counts[counts.index.notna()]

    if not fill_empty:
        # Grouper may insert empty periods (0 / NaN); keep only periods that had rows
        series = series[counts.reindex(series.index).fillna(0) > 0]
    else:
        start = pandas.Timestamp(xmin) if xmin is not None else series.index.min()
        end = pandas.Timestamp(xmax) if xmax is not None else series.index.max()
        if pandas.isna(start) or pandas.isna(end):
            start = work[column_x].min()
            end = work[column_x].max()
        # Prefer span of observed groups when xmin/xmax unset
        if xmin is None and len(counts[counts > 0]) > 0:
            start = counts[counts > 0].index.min()
        if xmax is None and len(counts[counts > 0]) > 0:
            end = counts[counts > 0].index.max()
        full_index = pandas.date_range(start=start, end=end, freq=freq)
        fill = _empty_fill_value(agg)
        series = series.reindex(full_index, fill_value=fill)

    out = series.rename(AGG_VALUE_CNAME).reset_index()
    out.columns = [column_x, AGG_VALUE_CNAME]
    if agg == "count":
        out[AGG_VALUE_CNAME] = out[AGG_VALUE_CNAME].fillna(0).astype(int)
    return out


def _aggregate_equal_width(work, column_x, column_y, agg, num_bins, fill_empty, xmin, xmax):
    _assert_x_dtype(work[column_x], freq=None)
    bins = binning.create_bins_quantitative(work[column_x], num_bins, xmin, xmax)
    rows = []
    for b in bins:
        subset = b.filter(work, column_x)
        if len(subset) == 0:
            if not fill_empty:
                continue
            value = _empty_fill_value(agg)
        elif agg == "count":
            value = len(subset)
        else:
            value = subset[column_y].agg(agg)
        rows.append({
            "{}_lb".format(column_x): b.lb,
            "{}_ub".format(column_x): b.ub,
            AGG_VALUE_CNAME: value,
        })
    if not rows:
        raise ValueError("no bins produced for column_x {!r}".format(column_x))
    return pandas.DataFrame(rows)
