from typing import Union
from collections.abc import Callable

import numpy
from sklearn import utils
from sklearn_cv_pandas import pandas_cv

from .logic.ml.tuner import (
    cv_analyzer,
    ml_tuner
)


RandomizedSearchCV = pandas_cv.RandomizedSearchCV
GridSearchCV = pandas_cv.GridSearchCV
CVAnalyzer = cv_analyzer.CVAnalyzer


def get_default_cv(ml_type: str, problem_type: str, scoring=None, search_type: str="random")\
        -> Union[GridSearchCV, RandomizedSearchCV]:
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


def estimate_std(actual: numpy.array, predicted: numpy.array, metric: Callable,
                 n_bootstrap: int = 10, n_samples: int = None):
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
