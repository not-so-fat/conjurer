from scipy import stats
from sklearn import ensemble

from . import base


estimator = {
    "cl": ensemble.RandomForestClassifier(),
    "rg": ensemble.RandomForestRegressor(),
    "mcl": ensemble.RandomForestClassifier()
}
grid = dict(
    max_depth=[2, 4, 10],
    n_estimators=[100, 200],
    max_features=["auto", "sqrt"],
    min_samples_leaf=[0.01, 0.05]
)
distributions = dict(
    max_depth=list(range(2, 11)),
    n_estimators=[100, 200],
    max_features=["auto", "sqrt"],
    min_samples_leaf=stats.uniform(0.01, 0.1)
)
config = base.TunerConfig(estimator, distributions, distributions, grid, grid)
