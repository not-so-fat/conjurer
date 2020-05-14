import io
import unittest

from conjurer import eda
from . import csv_data


class TestCheckStats(unittest.TestCase):
    def test_success_execution(self):
        df = eda.read_csv(io.StringIO(csv_data.ALL_TYPE_TEST_CSV))
        eda.check_stats(df)
