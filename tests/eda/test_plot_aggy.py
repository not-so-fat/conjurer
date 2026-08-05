import pandas
import pytest
from numpy import random

from conjurer import eda
from conjurer.logic.eda.vis import aggregation


def _txn_df():
    # Day1: 10+20, Day2: missing, Day3: 5
    return pandas.DataFrame({
        "timestamp": pandas.to_datetime([
            "2024-01-01 10:00",
            "2024-01-01 15:00",
            "2024-01-03 09:00",
        ]),
        "cost": [10.0, 20.0, 5.0],
        "txn_id": ["a", "b", "c"],
    })


def test_calendar_sum_fill_empty_default():
    df = _txn_df()
    out = aggregation.create_aggregation_table(df, "timestamp", "cost", agg="sum", freq="D")
    assert list(out["timestamp"].dt.normalize()) == list(pandas.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03"]
    ))
    assert list(out[aggregation.AGG_VALUE_CNAME]) == [30.0, 0.0, 5.0]


def test_calendar_sum_no_fill():
    df = _txn_df()
    out = aggregation.create_aggregation_table(
        df, "timestamp", "cost", agg="sum", freq="D", fill_empty=False
    )
    assert list(out[aggregation.AGG_VALUE_CNAME]) == [30.0, 5.0]


def test_calendar_count():
    df = _txn_df()
    out = aggregation.create_aggregation_table(
        df, "timestamp", "txn_id", agg="count", freq="D", fill_empty=True
    )
    assert list(out[aggregation.AGG_VALUE_CNAME]) == [2, 0, 1]


def test_equal_width_mean():
    df = pandas.DataFrame({
        "score": [0.0, 1.0, 2.0, 3.0],
        "revenue": [10.0, 20.0, 30.0, 40.0],
    })
    out = aggregation.create_aggregation_table(
        df, "score", "revenue", agg="mean", num_bins=2, fill_empty=True
    )
    assert len(out) == 2
    # bins [0,1.5) and [1.5,3] → means 15 and 35
    assert out[aggregation.AGG_VALUE_CNAME].tolist() == pytest.approx([15.0, 35.0])


def test_agg_min_max_median():
    df = _txn_df()
    for agg, expected_day1 in [("min", 10.0), ("max", 20.0), ("median", 15.0)]:
        out = aggregation.create_aggregation_table(
            df, "timestamp", "cost", agg=agg, freq="D", fill_empty=False
        )
        day1 = out[out["timestamp"] == pandas.Timestamp("2024-01-01")][aggregation.AGG_VALUE_CNAME].iloc[0]
        assert day1 == expected_day1


def test_invalid_agg():
    df = _txn_df()
    with pytest.raises(ValueError, match="agg must be"):
        aggregation.create_aggregation_table(df, "timestamp", "cost", agg="mode", freq="D")


def test_freq_requires_datetime():
    df = pandas.DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="datetime"):
        aggregation.create_aggregation_table(df, "x", "y", agg="sum", freq="D")


def test_categorical_x_rejected():
    df = pandas.DataFrame({"x": ["a", "b", "a"], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="numeric or datetime"):
        aggregation.create_aggregation_table(df, "x", "y", agg="sum", num_bins=2)


def test_xmin_xmax_filter():
    df = _txn_df()
    out = aggregation.create_aggregation_table(
        df, "timestamp", "cost", agg="sum", freq="D", fill_empty=False,
        xmin="2024-01-03",
    )
    assert list(out[aggregation.AGG_VALUE_CNAME]) == [5.0]


def test_plot_aggy_returns_chart_bar_and_line():
    df = _txn_df()
    chart = eda.plot_aggy(df, "timestamp", "cost", agg="sum", freq="D")
    assert chart is not None
    line = eda.plot_aggy(df, "timestamp", "cost", agg="sum", freq="D", mark="line")
    assert line is not None


def test_calendar_bar_spans_full_period():
    """Timestamp bars must use x/x2 over the freq width, not a thin point mark."""
    import altair as alt
    from conjurer.logic.eda.vis import aggy as aggy_mod

    df = _txn_df()
    chart = eda.plot_aggy(df, "timestamp", "cost", agg="sum", freq="D")
    enc = chart.to_dict()["encoding"]
    assert "x2" in enc

    agg_df = aggregation.create_aggregation_table(df, "timestamp", "cost", agg="sum", freq="D")
    plot_df, _, _ = aggy_mod._calendar_encodings(
        agg_df, "timestamp", alt.Y("{}:Q".format(aggregation.AGG_VALUE_CNAME)), "D", "bar",
    )
    assert (plot_df["timestamp_ub"] - plot_df["timestamp"]).dt.total_seconds().iloc[0] == 86400


def test_plot_aggy_numeric():
    df = pandas.DataFrame({
        "score": random.normal(0, 1, 200),
        "revenue": random.uniform(1, 10, 200),
    })
    chart = eda.plot_aggy(df, "score", "revenue", agg="mean", num_bins=10)
    assert chart is not None


def test_invalid_mark():
    df = _txn_df()
    with pytest.raises(ValueError, match="mark"):
        eda.plot_aggy(df, "timestamp", "cost", agg="sum", freq="D", mark="area")
