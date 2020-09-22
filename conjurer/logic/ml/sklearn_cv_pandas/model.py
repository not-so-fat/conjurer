import logging

import numpy
import pandas


logger = logging.getLogger(__name__)


class Model(object):
    """
    Class to provide sklearn model with prediction interface with pandas.DataFrame
    """
    def __init__(self, estimator, feature_columns, target_column=None):
        """
        Args:
            estimator: sklearn estimator / pipeline
            feature_columns (list of str): column names used for features
            target_column (str): column name used for target column
        """
        self.estimator = estimator
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.model = get_ml_model_from_pipeline(get_estimator_from_cvobj(estimator))
        self.is_cl = hasattr(estimator, "predict_proba")

    def predict(self, df):
        index = df.index
        x = self.get_x_from_df(df)
        result_df = self._get_result_for_cl(x, index) if self.is_cl else self._get_result_for_rg(x, index)
        other_columns = [c for c in df.columns if c not in self.feature_columns]
        for c in other_columns:
            result_df[c] = df[c]
        return result_df

    def get_x_from_df(self, df):
        return df[self.feature_columns].values

    def get_y_from_df(self, df):
        return df[self.target_column].values

    def _get_result_for_rg(self, x, index):
        return pandas.DataFrame({
            "score": self.estimator.predict(x)
        }, index=index)

    def _get_result_for_cl(self, x, index):
        predicted_class = self.estimator.predict(x)
        proba_array = self.estimator.predict_proba(x)
        num_class = proba_array.shape[1]
        if num_class == 2:
            return pandas.DataFrame({
                "score": proba_array[:, 1],
                "predicted_class": predicted_class
            }, index=index, columns=["score", "predicted_class"])
        else:
            return pandas.DataFrame({
                **{
                    "score_{}".format(i): proba_array[:, i]
                    for i in range(num_class)
                },
                **{"predicted_class": predicted_class}
            }, index=index, columns=["score_{}".format(i) for i in range(num_class)] + ["predicted_class"])

    def get_feature_importance(self):
        return get_linear_importance(self.model, self.feature_columns) if hasattr(self.model, "coef_") \
            else get_feature_importance_sklearn(self.model, self.feature_columns)


def get_linear_importance(model, feature_columns):
    coeff_df = pandas.DataFrame({
        "feature_name": feature_columns,
        "importance": numpy.abs(model.coef_[0, :]),
        "coeff": model.coef_[0, :]
    })
    coeff_df = coeff_df.sort_values(by="importance", ascending=False)
    coeff_df["rank"] = range(len(coeff_df))
    return coeff_df


def get_feature_importance_sklearn(model, feature_columns):
    fi_df = pandas.DataFrame({
        "feature_name": feature_columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)
    fi_df["rank"] = range(len(fi_df))
    return fi_df


def get_estimator_from_cvobj(estimator):
    return estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator


def get_ml_model_from_pipeline(estimator):
    return estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
