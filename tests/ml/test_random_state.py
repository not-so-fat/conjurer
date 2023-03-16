import pandas
from pandas import testing
from conjurer import ml
from tests.ml import utils


def test_lm_holdout1_grid():
    cv1 = ml.get_default_cv("linear_model", "cl", search_type="grid", random_state=0)
    cv2 = ml.get_default_cv("linear_model", "cl", search_type="grid", random_state=0)
    _test_cv_equal(cv1, cv2, True, "holdout1")


def test_lightgbm_holdout2_random():
    cv1 = ml.get_default_cv("lightgbm", "rg", search_type="random", random_state=0)
    cv2 = ml.get_default_cv("lightgbm", "rg", search_type="random", random_state=0)
    _test_cv_equal(cv1, cv2, False, "holdout2")


def test_lightgbm_cv_random():
    cv1 = ml.get_default_cv("lightgbm", "rg", search_type="random", random_state=0)
    cv2 = ml.get_default_cv("lightgbm", "rg", search_type="random", random_state=0)
    _test_cv_equal(cv1, cv2, False, "cv")


def _test_cv_equal(cv1, cv2, is_cl, cv_type):
    df_training = utils.get_input_df(100)
    df_validation = utils.get_input_df(100)
    df_test = utils.get_input_df(10)
    target_column = "target_cl" if is_cl else "target_rg"
    feature_columns = ["column{}".format(i) for i in range(6)]
    if cv_type == "holdout1":
        model1 = cv1.fit_holdout_pandas(df_training, target_column, feature_columns, df_validation)
        model2 = cv2.fit_holdout_pandas(df_training, target_column, feature_columns, df_validation)
    elif cv_type == "holdout2":
        model1 = cv1.fit_holdout_pandas(df_training, target_column, feature_columns, ratio_training=0.8)
        model2 = cv2.fit_holdout_pandas(df_training, target_column, feature_columns, ratio_training=0.8)
    else:
        model1 = cv1.fit_cv_pandas(df_training, target_column, feature_columns, cv=3)
        model2 = cv2.fit_cv_pandas(df_training, target_column, feature_columns, cv=3)
    pred1 = model1.predict(df_test)
    pred2 = model2.predict(df_test)
    _assert_cv_equal(cv1, cv2)
    testing.assert_frame_equal(pred1, pred2)


def _assert_cv_equal(cv1, cv2):
    columns = ["mean_test_score", "std_test_score", "mean_train_score", "std_train_score"]
    testing.assert_frame_equal(
        pandas.DataFrame(cv1.cv_results_)[columns], pandas.DataFrame(cv2.cv_results_)[columns]
    )
