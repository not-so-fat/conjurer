import unittest

from . import utils
from conjurer import ml


class TestTuneCV(unittest.TestCase):
    def setUp(self):
        self.df_training = utils.get_input_df(100)
        self.df_validation = utils.get_input_df(100)
        self.df_test = utils.get_input_df(10)
        self.feature_columns = ["column{}".format(i) for i in range(6)]

    def test_grid_cv(self):
        is_cl = True
        target_column = utils.get_target_column(is_cl)
        param_grid = dict(penalty=["l1", "l2"], C=[1e-5, 1e-3, 1e-1])
        model = ml.tune_cv("linear_model", "cl", self.df_training, target_column, self.feature_columns,
                           cv_args=dict(cv_type="grid", param_dict=param_grid))
        analyzer = ml.CVAnalyzer(model.estimator)
        self._basic_flow(analyzer)

    def test_random_cv(self):
        is_cl = False
        target_column = utils.get_target_column(is_cl)
        model = ml.tune_cv("lightgbm", "rg", self.df_training, target_column, self.feature_columns, scoring="r2")
        analyzer = ml.CVAnalyzer(model.estimator)
        self._basic_flow(analyzer)

    def test_grid_sv(self):
        is_cl = True
        target_column = utils.get_target_column(is_cl)
        param_grid = dict(penalty=["l1", "l2"], C=[1e-5, 1e-3, 1e-1])
        model = ml.tune_sv("linear_model", "cl", self.df_training, target_column, self.feature_columns,
                           ratio_training=0.8, cv_args=dict(cv_type="grid", param_dict=param_grid))
        analyzer = ml.CVAnalyzer(model.estimator)
        self._basic_flow(analyzer)

    def test_random_sv(self):
        is_cl = False
        target_column = utils.get_target_column(is_cl)
        model = ml.tune_sv("lightgbm", "rg", self.df_training, target_column, self.feature_columns,
                           df_validation=self.df_validation, scoring="r2")
        analyzer = ml.CVAnalyzer(model.estimator)
        self._basic_flow(analyzer)

    def _basic_flow(self, analyzer):
        analyzer.plot_flat()
        analyzer.plot_by_param_all()
        analyzer.get_summary()
        analyzer.get_best_model_info()
        analyzer.get_param_stat_df()
