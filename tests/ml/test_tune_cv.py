import pytest

from . import utils
from conjurer import ml


@pytest.fixture
def setup():
    df_training = utils.get_input_df(100)
    df_test = utils.get_input_df(10)
    feature_columns = ["column{}".format(i) for i in range(6)]
    return df_training, df_test, feature_columns


def tune_cv(ml_type, problem_type, scoring, df_training, target_column, feature_columns):
    cv_obj = ml.get_default_cv(ml_type, problem_type, scoring)
    return cv_obj.fit_cv_pandas(df_training, target_column, feature_columns, n_fold=3)


def test_default_cl(setup):
    df_training, df_test, feature_columns = setup
    is_cl = True
    target_column = utils.get_target_column(is_cl)
    model = tune_cv("lightgbm", "cl", None, df_training, target_column, feature_columns)
    utils.assert_prediction(model, df_test, is_cl)


def test_default_rg(setup):
    df_training, df_test, feature_columns = setup
    is_cl = False
    target_column = utils.get_target_column(is_cl)
    model = tune_cv("xgboost", "rg", None, df_training, target_column, feature_columns)
    utils.assert_prediction(model, df_test, is_cl)


def test_change_scorer_cl(setup):
    df_training, df_test, feature_columns = setup
    is_cl = True
    target_column = utils.get_target_column(is_cl)
    model = tune_cv("random_forest", "cl", "f1", df_training, target_column, feature_columns)
    utils.assert_prediction(model, df_test, is_cl)


def test_customized_cv_cl(setup):
    df_training, df_test, feature_columns = setup
    is_cl = True
    target_column = utils.get_target_column(is_cl)
    cv_obj = ml.get_default_cv(
        "linear_model", "cl", search_type="grid", param_dict=dict(penalty=["l1", "l2"], C=[1e-5, 1e-3, 1e-1])
    )
    model = cv_obj.fit_cv_pandas(df_training, target_column, feature_columns, n_fold=3)
    utils.assert_prediction(model, df_test, is_cl)
