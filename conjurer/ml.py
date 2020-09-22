import numpy
from sklearn import utils

from .logic.ml.tuner import (
    cv_analyzer,
    ml_tuner
)
from .logic.ml.sklearn_cv_pandas import pandas_cv


RandomizedSearchCV = pandas_cv.RandomizedSearchCV
GridSearchCV = pandas_cv.GridSearchCV
CVAnalyzer = cv_analyzer.CVAnalyzer


def get_default_cv(ml_type, problem_type, scoring=None, search_type="random"):
    """
    Get pandas-I/F [Randomized|Grid]SearchCV object with default search parameters and preprocessing
    Args:
        ml_type (str): One of "lightgbm", "xgboost", "random_forest", "linear_model"
        problem_type (str): One of "cl", "rg"
        scoring (str or scorer): `scoring` for RandomizedSearchCV / GridSearchCV
        search_type (str): "random" to use "RandomizedSearchCV", otherwise used "GridSearchCV"
        param_dict (str): parameter search space
    Returns:
        RandomizedSearchCV or GridSearchCV
    """
    return ml_tuner.get_cv(ml_type, problem_type, scoring, search_type)


def estimate_std(actual, predicted, metric, n_bootstrap=10, n_samples=None):
    """
    Estimate standard deviation for the metric in specified dataset by bootstrapping
    Args:
        actual(numpy.array):
        predicted(numpy.array):
        metric:
        n_bootstrap:
        n_samples:

    Returns:

    """
    return numpy.std(numpy.array([
        metric(*utils.resample(actual, predicted, n_samples=n_samples))
        for i_bootstramp in range(n_bootstrap)
    ]))
