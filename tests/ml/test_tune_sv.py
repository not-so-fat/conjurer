import unittest

from . import utils
from conjurer import ml


class TestTuneSV(unittest.TestCase):
    def setUp(self):
        self.df_training = utils.get_input_df(100)
        self.df_validation = utils.get_input_df(100)
        self.df_test = utils.get_input_df(10)
        self.feature_columns = ["column{}".format(i) for i in range(6)]

    def test_default_cl(self):
        is_cl = True
        target_column = self._get_target_column(is_cl)
        model = ml.tune_sv("lightgbm", "cl", self.df_training, target_column, self.feature_columns, ratio_training=0.8)
        self._assert_prediction(model, is_cl)

    def test_default_rg(self):
        is_cl = False
        target_column = self._get_target_column(is_cl)
        model = ml.tune_sv("xgboost", "rg", self.df_training, target_column, self.feature_columns,
                           df_validation=self.df_validation)
        self._assert_prediction(model, is_cl)

    def test_change_scorer_cl(self):
        is_cl = True
        target_column = self._get_target_column(is_cl)
        model = ml.tune_sv(
            "random_forest", "cl", self.df_training, target_column, self.feature_columns,
            ratio_training=0.8, scoring="f1")
        self._assert_prediction(model, is_cl)

    def test_customized_cv_cl(self):
        is_cl = True
        target_column = self._get_target_column(is_cl)
        param_grid=dict(penalty=["l1", "l2"], C=[1e-5, 1e-3, 1e-1])
        model = ml.tune_sv("linear_model", "cl", self.df_training, target_column, self.feature_columns,
                           df_validation=self.df_validation, cv_args=dict(cv_type="grid", param_dict=param_grid))
        self._assert_prediction(model, is_cl)

    @staticmethod
    def _get_target_column(is_cl):
        return "target_cl" if is_cl else "target_rg"

    def _assert_prediction(self, model, is_cl):
        pred_df = model.predict(self.df_test)
        expected_columns = ["score", "id1", "id2", "target_cl", "target_rg"]
        if is_cl:
            expected_columns.insert(1, "predicted_class")
        self.assertListEqual(list(pred_df.columns), expected_columns)
        self.assertEqual(len(pred_df), 10)
