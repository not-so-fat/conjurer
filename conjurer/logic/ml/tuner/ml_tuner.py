from conjurer import ml
from .tuner_config.utils import make_pipeline
from .tuner_config import (
    lightgbm,
    xgboost,
    random_forest,
    linear_model
)

default_scoring = {
    "cl": "roc_auc",
    "rg": "neg_mean_squared_error",
    "mcl": "accuracy"
}


def get_cv(ml_type, problem_type, scoring=None, search_type="random", n_iter=20, **kwargs):
    """
    Get RandomizedSearchCV or GridSearchCV object for tuning
    Args:
        ml_type (str): One of "lightgbm", "xgboost", "random_forest", "linear_model"
        problem_type (str): One of "cl", "rg", "mcl"
        scoring (str or scorer): `scoring` for RandomizedSearchCV / GridSearchCV
        search_type (str): Return RandomizedSearchCV if "random", GridSearchCV if "grid"
        param_dict (dict): param_distribution or param_grid for CV object
        n_iter (int): `n_iter` for RandomizedSearchCV
        kwargs: keyword arguments for CV object

    Returns:
        conjurer.ml.RandomizedSearchCV or conjurer.ml.GridSearchCV
    """
    _validate_attr(ml_type, problem_type, search_type)
    config = getattr(globals()[ml_type], "config")
    scoring = scoring or default_scoring[problem_type]
    ml_estimator = config.estimator_dict[problem_type]
    estimator = make_pipeline.add_default_preprocessing("ml", ml_estimator)
    param_dict = _get_param_dict(problem_type, search_type, "ml", config.parameters_dict)
    cv_class = ml.RandomizedSearchCV if search_type == "random" else ml.GridSearchCV
    if search_type == "random":
        kwargs["n_iter"] = n_iter
    return cv_class(estimator, param_dict, scoring=scoring, **kwargs)


def _validate_attr(ml_type, problem_type, search_type):
    assert ml_type in ["lightgbm", "xgboost", "linear_model", "random_forest"], \
        "`ml_type` should be one of lightgbm, xgboost, linear_model, or random_forest"
    assert problem_type in ["cl", "rg", "mcl"], "`problem_type` should be one of cl, rg, or mcl"
    assert search_type in ["random", "grid"], "`search_type` should be one of random or grid"


def _get_param_dict(problem_type, cv_type, name, parameters_dict):
    ml_params = parameters_dict[cv_type][problem_type]
    return make_pipeline.convert_param_dict(name, ml_params)
