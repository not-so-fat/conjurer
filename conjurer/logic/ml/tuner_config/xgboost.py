from scipy import stats

from . import base
import gbm_autosplit


ESTIMATOR_PARAMS = dict(
    max_depth=8,
    max_n_estimators=5000,
    early_stopping_rounds=100,
    importance_type="gain"
)
estimator = {
    "cl": gbm_autosplit.XGBClassifier(metric="auc", **ESTIMATOR_PARAMS),
    "rg": gbm_autosplit.XGBRegressor(metric="rmse", **ESTIMATOR_PARAMS),
    "mcl": gbm_autosplit.XGBClassifier(metric="accuracy", **ESTIMATOR_PARAMS)
}

grid = dict(
    reg_lambda=[0.0, 0.01, 1.0],
    learning_rate=[0.01, 0.02, 0.1],
    ratio_min_child_weight=[0.0, 0.01, 0.1]
)
distributions = dict(
    reg_lambda=stats.uniform(0.0, 1.0),
    learning_rate=stats.uniform(0.01, 0.5),
    ratio_min_child_weight=stats.uniform(0.0, 0.1)
)
config = base.TunerConfig(estimator, distributions, distributions, grid, grid)
