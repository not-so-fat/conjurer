import logging

import pandas
from numpy import random

from conjurer import eda
from conjurer.logic.eda.check import stat_calculator
from conjurer.logic.eda.vis import binning, histogram


def test_integer_over_num_bins_array():
    series = pandas.Series(random.choice(range(100), 10000), dtype="Int64")
    eda.plot_histogram(series, num_bins=11).display()


def test_integer_under_num_bins_array():
    series = pandas.Series(random.choice(range(10), 10000), dtype="Int64")
    eda.plot_histogram(series, num_bins=11).display()


def test_histogram_tz_aware_datetime():
    """tz-aware Series must bin without tz-naive vs tz-aware compare errors."""
    series = pandas.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-05"]
    ).tz_localize("UTC")
    series.name = "ts"
    chart = eda.plot_histogram(series, num_bins=5)
    assert chart is not None


def test_frequency_table_tz_aware_counts_all_rows():
    series = pandas.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03"]
    ).tz_localize("UTC")
    series.name = "ts"
    freq = binning.create_frequency_table(series, num_bins=5)
    assert freq[binning.FREQUENCY_CNAME].sum() == len(series)


def test_histogram_tz_naive_datetime_still_works():
    series = pandas.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    series.name = "ts"
    chart = eda.plot_histogram(series, num_bins=5)
    assert chart is not None


def test_plot_histogram_for_stats_does_not_skip_tz_aware(caplog):
    """check_stats path must plot tz-aware cols, not swallow compare errors as skips."""
    df = pandas.DataFrame({
        "aware": pandas.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-05"]
        ).tz_localize("UTC"),
        "num": [1.0, 2.0, 3.0, 4.0],
    })
    stat_df = stat_calculator.calc_column_stat(df)
    with caplog.at_level(logging.INFO, logger="conjurer.logic.eda.vis.histogram"):
        histogram.plot_histogram_for_stats(df, stat_df, num_bins=5)
    skip_msgs = [r.message for r in caplog.records if "was skipped" in r.message]
    assert skip_msgs == [], skip_msgs


def test_check_stats_tz_aware_does_not_skip(caplog):
    df = pandas.DataFrame({
        "aware": pandas.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03"]
        ).tz_localize("UTC"),
    })
    with caplog.at_level(logging.INFO, logger="conjurer.logic.eda.vis.histogram"):
        eda.check_stats(df)
    skip_msgs = [r.message for r in caplog.records if "was skipped" in r.message]
    assert skip_msgs == [], skip_msgs
