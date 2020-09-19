import pandas
from numpy import random

from conjurer import eda


def test_integer_over_num_bins_array():
    series = pandas.Series(random.choice(range(100), 10000), dtype="Int64")
    eda.plot_histogram(series, num_bins=11).display()


def test_integer_under_num_bins_array():
    series = pandas.Series(random.choice(range(10), 10000), dtype="Int64")
    eda.plot_histogram(series, num_bins=11).display()
