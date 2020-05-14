from .logic.ml.cv_analyzer import CVAnalyzer
from .logic.ml.ml_tuner import(
    get_cv,
)


def tune_cv(ml_type, problem_type, df, target_column, feature_columns,
            n_fold=5, scoring=None, cv_args={}, fit_args={}):
    """
    Tune hyper parameter for specified machine learning algorithm by cross validation
    Args:
        ml_type (str): One of "lightgbm", "xgboost", "random_forest", "linear_model"
        problem_type (str): One of "cl", "rg", "mcl"
        df (pandas.DataFrame): Input data set
        target_column (str): Column name of target variable
        feature_columns (list of str): Column names of features
        n_fold (int): The number of folds in cross validation
        scoring (str or scorer): `scoring` for RandomizedSearchCV / GridSearchCV
        cv_args: keyword argument for `get_cv`
        fit_args: keyword argument for `fit` method

    Returns:
        sklearn_cv_pandas.Model
    """
    cv = get_cv(ml_type, problem_type, scoring, **cv_args)
    return cv.fit_cv_pandas(df, target_column, feature_columns, n_fold, **fit_args)


def tune_sv(ml_type, problem_type, df, target_column, feature_columns,
            ratio_training=None, df_validation=None, scoring=None, cv_args={}, fit_args={}):
    """
    Tune hyper parameter for specified machine learning algorithm by single validation
    Args:
        ml_type (str): One of "lightgbm", "xgboost", "random_forest", "linear_model"
        problem_type (str): One of "cl", "rg", "mcl"
        df (pandas.DataFrame): Input data set
        target_column (str): Column name of target variable
        feature_columns (list of str): Column names of features
        ratio_training (float): Ratio of training data set
        df_validation (pandas.DataFrame): Data set used for validation
        scoring (str or scorer): `scoring` for RandomizedSearchCV / GridSearchCV
        cv_args: keyword argument for `get_cv`
        fit_args: keyword argument for `fit` method

    Returns:
        sklearn_cv_pandas.Model
    """
    cv = get_cv(ml_type, problem_type, scoring, **cv_args)
    return cv.fit_sv_pandas(df, target_column, feature_columns,
                            df_validation=df_validation, ratio_training=ratio_training, **fit_args)


CVAnalyzer = CVAnalyzer
