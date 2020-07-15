import pytest

from . import utils
from conjurer import ml


@pytest.fixture
def setup():
    df_training = utils.get_input_df(100)
    df_test = utils.get_input_df(10)
    feature_columns = ["column{}".format(i) for i in range(6)]
    return df_training, df_test, feature_columns


def test_default_cl(setup):
    df_training, df_test, feature_columns = setup
    is_cl = True
    target_column = utils.get_target_column(is_cl)
    model = ml.tune_cv("lightgbm", "cl", df_training, target_column, feature_columns)
    utils.assert_prediction(model, df_test, is_cl)


def test_default_rg(setup):
    df_training, df_test, feature_columns = setup
    is_cl = False
    target_column = utils.get_target_column(is_cl)
    model = ml.tune_cv("xgboost", "rg", df_training, target_column, feature_columns)
    utils.assert_prediction(model, df_test, is_cl)


def test_change_scorer_cl(setup):
    df_training, df_test, feature_columns = setup
    is_cl = True
    target_column = utils.get_target_column(is_cl)
    model = ml.tune_cv(
        "random_forest", "cl", df_training, target_column, feature_columns, scoring="f1")
    utils.assert_prediction(model, df_test, is_cl)


def test_customized_cv_cl(setup):
    df_training, df_test, feature_columns = setup
    is_cl = True
    target_column = utils.get_target_column(is_cl)
    param_grid=dict(penalty=["l1", "l2"], C=[1e-5, 1e-3, 1e-1])
    model = ml.tune_cv("linear_model", "cl", df_training, target_column, feature_columns,
                       cv_args=dict(cv_type="grid", param_dict=param_grid))
    utils.assert_prediction(model, df_test, is_cl)
