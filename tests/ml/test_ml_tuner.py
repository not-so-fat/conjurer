import unittest

from conjurer import ml
from tests.ml import utils


class TestHPTuner(unittest.TestCase):
    def test_lm_cv_grid(self):
        test_hps = dict(
            penalty=["l1"],
            C=[1e-5, 1e-3, 1e-1]
        )
        cv = ml.get_cv("linear_model", "cl", cv_type="grid", param_dict=test_hps)
        self._test_basic_flow_sv_pandas1(cv, True)
        self._test_basic_flow_sv_pandas2(cv, True)
        self._test_basic_flow_cv_pandas(cv, True)

    def test_lm_cv_random(self):
        cv = ml.get_cv("linear_model", "cl", cv_type="random", n_iter=2)
        self._test_basic_flow_sv_pandas1(cv, True)
        self._test_basic_flow_sv_pandas2(cv, True)
        self._test_basic_flow_cv_pandas(cv, True)

    def test_lightgbm_cv_grid(self):
        test_hps = dict(
            num_leaves=[10],
            feature_fraction=[0.1],
            learning_rate=[0.01],
            ratio_min_data_in_leaf=[None, 0.005, 0.01]
        )
        cv = ml.get_cv("lightgbm", "cl", cv_type="grid", param_dict=test_hps)
        self._test_basic_flow_sv_pandas1(cv, True)
        self._test_basic_flow_sv_pandas2(cv, True)
        self._test_basic_flow_cv_pandas(cv, True)

    def test_lightgbm_cv_random(self):
        cv = ml.get_cv("lightgbm", "cl", cv_type="random", n_iter=5)
        self._test_basic_flow_sv_pandas1(cv, True)
        self._test_basic_flow_sv_pandas2(cv, True)
        self._test_basic_flow_cv_pandas(cv, True)

    def test_lightgbm_cv_random_rg(self):
        cv = ml.get_cv("lightgbm", "rg", cv_type="random", n_iter=5)
        self._test_basic_flow_sv_pandas1(cv, False)
        self._test_basic_flow_sv_pandas2(cv, False)
        self._test_basic_flow_cv_pandas(cv, False)

    def test_xgboost_cv_grid(self):
        test_hps = dict(
            colsample_bynode=[0.1],
            learning_rate=[0.01],
            ratio_min_child_weight=[None, 0.005, 0.01]
        )
        cv = ml.get_cv("xgboost", "rg", cv_type="grid", param_dict=test_hps)
        self._test_basic_flow_sv_pandas1(cv, False)
        self._test_basic_flow_sv_pandas2(cv, False)
        self._test_basic_flow_cv_pandas(cv, False)

    def test_xgboost_cv_random(self):
        cv = ml.get_cv("xgboost", "cl", cv_type="random", n_iter=5)
        self._test_basic_flow_sv_pandas1(cv, True)
        self._test_basic_flow_sv_pandas2(cv, True)
        self._test_basic_flow_cv_pandas(cv, True)

    def test_random_forest_cv_grid(self):
        test_hps = dict(
            max_depth=[2],
            n_estimators=[100],
            max_features=["auto"],
            min_samples_leaf=[0.01, 0.05, 0.1]
        )
        cv = ml.get_cv("random_forest", "cl", cv_type="grid", param_dict=test_hps)
        self._test_basic_flow_sv_pandas1(cv, True)
        self._test_basic_flow_sv_pandas2(cv, True)
        self._test_basic_flow_cv_pandas(cv, True)

    def test_random_forest_cv_random(self):
        cv = ml.get_cv("random_forest", "rg", cv_type="random", n_iter=5)
        self._test_basic_flow_sv_pandas1(cv, False)
        self._test_basic_flow_sv_pandas2(cv, False)
        self._test_basic_flow_cv_pandas(cv, False)

    def _test_basic_flow_sv_pandas1(self, cv, is_cl):
        df_training = utils.get_input_df(100)
        df_test = utils.get_input_df(10)
        target_column = "target_cl" if is_cl else "target_rg"
        feature_columns = ["column{}".format(i) for i in range(6)]
        model = cv.fit_sv_pandas(df_training, target_column, feature_columns, ratio_training=0.8)
        self._assert_prediction(model, df_test, is_cl)

    def _test_basic_flow_sv_pandas2(self, cv, is_cl):
        df_training = utils.get_input_df(100)
        df_validation = utils.get_input_df(100)
        df_test = utils.get_input_df(10)
        target_column = "target_cl" if is_cl else "target_rg"
        feature_columns = ["column{}".format(i) for i in range(6)]
        model = cv.fit_sv_pandas(df_training, target_column, feature_columns, df_validation)
        self._assert_prediction(model, df_test, is_cl)

    def _test_basic_flow_cv_pandas(self, cv, is_cl):
        df_training = utils.get_input_df(100)
        df_test = utils.get_input_df(10)
        target_column = "target_cl" if is_cl else "target_rg"
        feature_columns = ["column{}".format(i) for i in range(6)]
        model = cv.fit_cv_pandas(df_training, target_column, feature_columns, 3)
        self._assert_prediction(model, df_test, is_cl)

    def _assert_prediction(self, model, df_test, is_cl):
        pred_df = model.predict(df_test)
        expected_columns = ["score", "id1", "id2", "target_cl", "target_rg"]
        if is_cl:
            expected_columns.insert(1, "predicted_class")
        self.assertListEqual(list(pred_df.columns), expected_columns)
        self.assertEqual(len(pred_df), 10)
