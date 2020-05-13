from sklearn import (
    pipeline,
    impute,
    preprocessing
)


def add_default_preprocessing(name, ml_estimator):
    return pipeline.Pipeline(
        steps=[
            ("mvi", impute.SimpleImputer()),
            ("std", preprocessing.StandardScaler()),
            (name, ml_estimator)
        ]
    )


def convert_param_dict(name, params_dict):
    return {
        "{}__{}".format(name, key): item
        for key, item in params_dict.items()
    }
