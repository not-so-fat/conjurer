import sklearn_cv_pandas
from .tuner_config.utils import make_pipeline
from .tuner_config import (
    lightgbm,
    linear_model,
    random_forest,
    xgboost
)


def get_cv(ml_type, problem_type, scoring=None, cv_type="random", param_dict=None, n_iter=20, **kwargs):
    """
    Get sklearn_cv_pandas.RandomizedSearchCV or sklearn_cv_pandas.GridSearchCV object for tuning
    Args:
        ml_type (str): One of "lightgbm", "xgboost", "random_forest", "linear_model"
        problem_type (str): One of "cl", "rg", "mcl"
        scoring (str or scorer): `scoring` for RandomizedSearchCV / GridSearchCV
        cv_type (str): Return RandomizedSearchCV if "random", GridSearchCV if "grid"
        param_dict (dict): param_distribution or param_grid for CV object
        n_iter (int): `n_iter` for RandomizedSearchCV
        kwargs: keyword arguments for CV object

    Returns:
        sklearn_cv_pandas.RandomizedSearchCV or sklearn_cv_pandas.GridSearchCV
    """
    _validate_attr(ml_type, problem_type, cv_type)
    config = getattr(globals()[ml_type], "config")
    scoring = scoring or "roc_auc" if problem_type == "cl" else \
        "neg_root_mean_squared_error" if problem_type == "rg" else "accuracy"
    ml_estimator = config.estimator_dict[problem_type]
    estimator = make_pipeline.add_default_preprocessing("ml", ml_estimator)
    param_dict = _get_param_dict(problem_type, cv_type, param_dict, "ml", config.parameters_dict)
    cv_class = sklearn_cv_pandas.RandomizedSearchCV if cv_type == "random" else sklearn_cv_pandas.GridSearchCV
    if cv_type == "random":
        kwargs["n_iter"] = n_iter
    return cv_class(estimator, param_dict, scoring=scoring, **kwargs)


def _validate_attr(ml_type, problem_type, cv_type):
    assert ml_type in ["lightgbm", "xgboost", "linear_model", "random_forest"], \
        "`ml_type` should be one of lightgbm, xgboost, linear_model, or random_forest"
    assert problem_type in ["cl", "rg", "mcl"], "`problem_type` should be one of cl, rg, or mcl"
    assert cv_type in ["random", "grid"], "`cv_type` should be one of random or grid"


def _get_param_dict(problem_type, cv_type, param_dict, name, parameters_dict):
    ml_params = param_dict or parameters_dict[cv_type][problem_type]
    return make_pipeline.convert_param_dict(name, ml_params)
