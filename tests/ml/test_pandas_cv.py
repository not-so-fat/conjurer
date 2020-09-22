import unittest

from scipy import stats
from sklearn import (
    linear_model,
    tree,
    pipeline,
    impute,
    preprocessing
)

from conjurer import ml
from tests.ml import utils


class TestPandasCV(unittest.TestCase):
    def test_random_linear_sv_ratio_cl(self):
        model_type, is_cl, with_prep, cv_type = "linear", True, False, "random"
        cv = self._get_cv(model_type, is_cl, with_prep, cv_type)
        self._test_basic_flow_sv_ratio(cv, is_cl)

    def test_random_tree_sv_2dfs_rg(self):
        model_type, is_cl, with_prep, cv_type = "tree", False, False, "random"
        cv = self._get_cv(model_type, is_cl, with_prep, cv_type)
        self._test_basic_flow_sv_2dfs(cv, is_cl)

    def test_random_tree_with_prep_cv_cl(self):
        model_type, is_cl, with_prep, cv_type = "tree", True, True, "random"
        cv = self._get_cv(model_type, is_cl, with_prep, cv_type)
        self._test_basic_flow_cv(cv, is_cl, 3)

    def test_grid_linear_sv_ratio_cl(self):
        model_type, is_cl, with_prep, cv_type = "linear", True, False, "grid"
        cv = self._get_cv(model_type, is_cl, with_prep, cv_type)
        self._test_basic_flow_sv_ratio(cv, is_cl)

    def test_grid_tree_sv_2dfs_rg(self):
        model_type, is_cl, with_prep, cv_type = "tree", False, False, "grid"
        cv = self._get_cv(model_type, is_cl, with_prep, cv_type)
        self._test_basic_flow_sv_2dfs(cv, is_cl)

    def test_grid_tree_with_prep_cv_cl(self):
        model_type, is_cl, with_prep, cv_type = "tree", True, True, "grid"
        cv = self._get_cv(model_type, is_cl, with_prep, cv_type)
        self._test_basic_flow_cv(cv, is_cl, 3)

    def _get_cv(self, model_type, is_cl, with_prep, cv_type):
        estimator = self._get_estimator(model_type, is_cl, with_prep)
        metric = "roc_auc" if is_cl else "neg_root_mean_squared_error"
        if cv_type == "random":
            params = self._get_params_random(model_type, is_cl, with_prep)
            return ml.RandomizedSearchCV(estimator, params, scoring=metric)
        else:
            params = self._get_params_grid(model_type, is_cl, with_prep)
            return ml.GridSearchCV(estimator, params, scoring=metric)

    def _get_estimator(self, model_type, is_cl, with_preprocessing):
        if model_type == "linear":
            ml_estimator = linear_model.LogisticRegression(solver="liblinear") if is_cl else linear_model.Lasso()
        else:
            ml_estimator = tree.DecisionTreeClassifier() if is_cl else tree.DecisionTreeRegressor()
        return self._add_preprocessing(ml_estimator) if with_preprocessing else ml_estimator

    @staticmethod
    def _add_preprocessing(estimator):
        return pipeline.Pipeline(
            steps=[
                ("mvi", impute.SimpleImputer()),
                ("std", preprocessing.StandardScaler()),
                ("ml", estimator)
            ]
        )

    def _get_params_random(self, model_type, is_cl, with_preprocessing):
        if model_type == "linear":
            ml_params = dict(
                penalty=["l1", "l2"],
                C=stats.loguniform(1e-5, 10)
            ) if is_cl else dict(alpha=stats.loguniform(1e-5, 10))
        else:
            ml_params = dict(max_depth=list(range(5, 16)))
        return self._convert_ml_params(ml_params) if with_preprocessing else ml_params

    def _get_params_grid(self, model_type, is_cl, with_preprocessing):
        if model_type == "linear":
            ml_params = dict(
                penalty=["l1", "l2"],
                C=[1e-5, 1e-3]
            ) if is_cl else dict(alpha=[1e-5, 1e-3, 1e-1, 10])
        else:
            ml_params = dict(max_depth=[5, 8, 11, 14])
        return self._convert_ml_params(ml_params) if with_preprocessing else ml_params

    @staticmethod
    def _convert_ml_params(ml_params):
        return {"{}__{}".format("ml", k): v for k, v in ml_params.items()}

    def _test_basic_flow_sv_ratio(self, cv, is_cl):
        df_training = utils.get_input_df(100)
        df_test = utils.get_input_df(10)
        target_column = "target_cl" if is_cl else "target_rg"
        feature_columns = ["column{}".format(i) for i in range(6)]
        model = cv.fit_sv_pandas(df_training, target_column, feature_columns, ratio_training=0.8)
        self._assert_prediction(model, df_test, is_cl)

    def _test_basic_flow_sv_2dfs(self, cv, is_cl):
        df_training = utils.get_input_df(100)
        df_validation = utils.get_input_df(100)
        df_test = utils.get_input_df(10)
        target_column = "target_cl" if is_cl else "target_rg"
        feature_columns = ["column{}".format(i) for i in range(6)]
        model = cv.fit_sv_pandas(df_training, target_column, feature_columns, df_validation)
        self._assert_prediction(model, df_test, is_cl)

    def _test_basic_flow_cv(self, cv, is_cl, n_fold):
        df_training = utils.get_input_df(100)
        df_test = utils.get_input_df(10)
        target_column = "target_cl" if is_cl else "target_rg"
        feature_columns = ["column{}".format(i) for i in range(6)]
        model = cv.fit_cv_pandas(df_training, target_column, feature_columns, n_fold=n_fold)
        self._assert_prediction(model, df_test, is_cl)

    def _assert_prediction(self, model, df_test, is_cl):
        pred_df = model.predict(df_test)
        expected_columns = ["score", "id1", "id2", "target_cl", "target_rg"]
        if is_cl:
            expected_columns.insert(1, "predicted_class")
        self.assertListEqual(list(pred_df.columns), expected_columns)
        self.assertEqual(len(pred_df), 10)
