import unittest

import pandas
from numpy import random

from conjurer import eda


class TestHistogram(unittest.TestCase):
    def test_integer_over_num_bins_array(self):
        array = pandas.Series(random.choice(range(100), 10000), dtype="Int64").values
        eda.plot_histogram(array, num_bins=11)

    def test_integer_under_num_bins_array(self):
        array = pandas.Series(random.choice(range(10), 10000), dtype="Int64").values
        eda.plot_histogram(array, num_bins=11)
