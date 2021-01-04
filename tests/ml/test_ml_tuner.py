from conjurer import ml
from tests.ml import utils


def test_lm_cv_grid():
    cv = ml.get_default_cv("linear_model", "cl", search_type="grid")
    cv.param_grid = {
        "ml__penalty": ["l1"],
        "ml__C": [1e-5, 1e-3, 1e-1]
    }
    _test_basic_flow_sv_pandas1(cv, True)
    _test_basic_flow_sv_pandas2(cv, True)
    _test_basic_flow_cv_pandas(cv, True)


def test_lm_cv_random():
    cv = ml.get_default_cv("linear_model", "cl", search_type="random")
    cv.n_iter = 2
    _test_basic_flow_sv_pandas1(cv, True)
    _test_basic_flow_sv_pandas2(cv, True)
    _test_basic_flow_cv_pandas(cv, True)


def test_lightgbm_cv_grid():
    cv = ml.get_default_cv("lightgbm", "cl", search_type="grid")
    cv.param_grid = {
        "ml__num_leaves": [10],
        "ml__colsample_bytree": [0.1],
        "ml__learning_rate": [0.01],
        "ml__min_child_samples": [0, 20, 100]
    }
    _test_basic_flow_sv_pandas1(cv, True)
    _test_basic_flow_sv_pandas2(cv, True)
    _test_basic_flow_cv_pandas(cv, True)


def test_lightgbm_cv_random():
    cv = ml.get_default_cv("lightgbm", "cl", search_type="random")
    cv.n_iter = 5
    _test_basic_flow_sv_pandas1(cv, True)
    _test_basic_flow_sv_pandas2(cv, True)
    _test_basic_flow_cv_pandas(cv, True)


def test_lightgbm_cv_random_rg():
    cv = ml.get_default_cv("lightgbm", "rg", search_type="random")
    cv.n_iter = 5
    _test_basic_flow_sv_pandas1(cv, False)
    _test_basic_flow_sv_pandas2(cv, False)
    _test_basic_flow_cv_pandas(cv, False)


def test_xgboost_cv_grid():
    cv = ml.get_default_cv("xgboost", "rg", search_type="grid")
    cv.param_grid = dict(
        ml__colsample_bynode=[0.1],
        ml__learning_rate=[0.01],
        ml__ratio_min_child_weight=[None, 0.005, 0.01]
    )
    _test_basic_flow_sv_pandas1(cv, False)
    _test_basic_flow_sv_pandas2(cv, False)
    _test_basic_flow_cv_pandas(cv, False)


def test_xgboost_cv_random():
    cv = ml.get_default_cv("xgboost", "cl", search_type="random")
    cv.n_iter = 5
    _test_basic_flow_sv_pandas1(cv, True)
    _test_basic_flow_sv_pandas2(cv, True)
    _test_basic_flow_cv_pandas(cv, True)


def test_random_forest_cv_grid():
    cv = ml.get_default_cv("random_forest", "cl", search_type="grid")
    cv.param_grid = dict(
        ml__max_depth=[2],
        ml__n_estimators=[100],
        ml__max_features=["auto"],
        ml__min_samples_leaf=[0.01, 0.05, 0.1]
    )
    _test_basic_flow_sv_pandas1(cv, True)
    _test_basic_flow_sv_pandas2(cv, True)
    _test_basic_flow_cv_pandas(cv, True)


def test_random_forest_cv_random():
    cv = ml.get_default_cv("random_forest", "rg", search_type="random")
    cv.n_iter = 5
    _test_basic_flow_sv_pandas1(cv, False)
    _test_basic_flow_sv_pandas2(cv, False)
    _test_basic_flow_cv_pandas(cv, False)


def _test_basic_flow_sv_pandas1(cv, is_cl):
    df_training = utils.get_input_df(100)
    df_test = utils.get_input_df(10)
    target_column = "target_cl" if is_cl else "target_rg"
    feature_columns = ["column{}".format(i) for i in range(6)]
    model = cv.fit_sv_pandas(df_training, target_column, feature_columns, ratio_training=0.8)
    _assert_prediction(model, df_test, is_cl)


def _test_basic_flow_sv_pandas2(cv, is_cl):
    df_training = utils.get_input_df(100)
    df_validation = utils.get_input_df(100)
    df_test = utils.get_input_df(10)
    target_column = "target_cl" if is_cl else "target_rg"
    feature_columns = ["column{}".format(i) for i in range(6)]
    model = cv.fit_sv_pandas(df_training, target_column, feature_columns, df_validation)
    _assert_prediction(model, df_test, is_cl)


def _test_basic_flow_cv_pandas(cv, is_cl):
    df_training = utils.get_input_df(100)
    df_test = utils.get_input_df(10)
    target_column = "target_cl" if is_cl else "target_rg"
    feature_columns = ["column{}".format(i) for i in range(6)]
    model = cv.fit_cv_pandas(df_training, target_column, feature_columns, 3)
    _assert_prediction(model, df_test, is_cl)


def _assert_prediction(model, df_test, is_cl):
    pred_df = model.predict(df_test)
    expected_columns = ["score", "id1", "id2", "target_cl", "target_rg"]
    if is_cl:
        expected_columns.insert(1, "predicted_class")
    assert list(pred_df.columns) == expected_columns
    assert len(pred_df) == 10
