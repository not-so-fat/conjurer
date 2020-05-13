from scipy import stats
from sklearn import linear_model

from . import base


estimator = {
    "cl": linear_model.LogisticRegression(solver="liblinear"),
    "rg": linear_model.Lasso(),
    "mcl": linear_model.LogisticRegression(solver="liblinear")
}
grid_rg = dict(
    alpha=[0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10]
)
grid_cl = dict(
    penalty=["l1", "l2"],
    C=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
)
distributions_rg = dict(
    alpha=stats.loguniform(1e-5, 10)
)
distributions_cl = dict(
    penalty=["l1", "l2"],
    C=stats.loguniform(1e-5, 10)
)
config = base.TunerConfig(estimator, distributions_rg, distributions_cl, grid_rg, grid_cl)
