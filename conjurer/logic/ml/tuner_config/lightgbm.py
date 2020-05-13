from scipy import stats
import gbm_autosplit
from . import base


ESTIMATOR_PARAMS = dict(
    boosting_type="gbdt",
    max_depth=-1,
    max_n_estimators=5000,
    early_stopping_rounds=100,
    ratio_training=0.9,
    importance_type="gain"
)
estimator = {
    "cl": gbm_autosplit.LGBMClassifier(metric="auc", **ESTIMATOR_PARAMS),
    "rg": gbm_autosplit.LGBMRegressor(metric="rmse", **ESTIMATOR_PARAMS),
    "mcl": gbm_autosplit.LGBMClassifier(metric="multi_logloss", **ESTIMATOR_PARAMS)
}

grid_rg = dict(
    num_leaves=[10, 20, 40],
    feature_fraction=[0.1, 0.5],
    learning_rate=[0.01, 0.05, 0.1],
    ratio_min_data_in_leaf=[None, 0.005, 0.01]
)
grid_cl = {
    **grid_rg,
    **dict(class_weight=["balanced", None])
}
distributions_rg = dict(
    num_leaves=list(range(10, 101)),
    feature_fraction=stats.uniform(0.05, 0.95),
    learning_rate=stats.uniform(0.01, 0.5),
    ratio_min_data_in_leaf=stats.uniform(0.0, 0.05)
)
distributions_cl = {
    **distributions_rg,
    **dict(class_weight=["balanced", None])
}
config = base.TunerConfig(estimator, distributions_rg, distributions_cl, grid_rg, grid_cl)
