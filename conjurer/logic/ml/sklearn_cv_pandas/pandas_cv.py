import logging
from functools import reduce
from operator import mul

import numpy
from sklearn import model_selection

from . import model


logger = logging.getLogger(__name__)


class RandomizedSearchCV(model_selection.RandomizedSearchCV):
    """
    sklearn.model_selection.RandomizedSearchCV with pandas DataFrame interface
    """
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None, n_jobs=None, pre_dispatch='2*n_jobs',
                 cv=None, refit=True, verbose=10, random_state=None, error_score=numpy.nan, return_train_score=True):
        """
        The same manner as [sklearn.model_selection.RandomizedSearchCV](
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
        """
        super(RandomizedSearchCV, self).__init__(
            estimator, param_distributions, n_iter=n_iter, scoring=scoring, n_jobs=n_jobs, pre_dispatch=pre_dispatch,
            cv=cv, refit=refit, verbose=verbose, random_state=random_state, error_score=error_score,
            return_train_score=return_train_score
        )

    def fit_sv_pandas(self, df_training, target_column, feature_columns,
                      df_validation=None, ratio_training=None, **kwargs):
        """
        `fit` for pandas DataFrame to perform single validation
        Args:
            df_training (pandas.DataFrame): training data set
            target_column (str): column name of prediction target
            feature_columns (list of str): column names of features
            df_validation (pandas.DataFrame): if specified, used as validation data set
            ratio_training (float): if specified, `df_training` is split for training / validation
            **kwargs: Other keyword arguments for original `fit`

        Returns:
            conjurer.ml.Model
        """
        x, y, num_training, num_validation = _split_for_sv(
            df_training, target_column, feature_columns, df_validation, ratio_training)
        self.cv = model_selection.PredefinedSplit(
            numpy.array([-1] * num_training + [0] * num_validation))
        logger.warning("start learning with {} hyper parameters".format(self.n_iter))
        self.fit(x, y, **kwargs)
        return model.Model(self, feature_columns=feature_columns, target_column=target_column)

    def fit_cv_pandas(self, df, target_column, feature_columns, n_fold, **kwargs):
        """
        `fit` for pandas DataFrame to perform cross validation
        Args:
            df (pandas.DataFrame): training data set
            target_column (str): column name of prediction target
            feature_columns (list of str): column names of features
            n_fold (int): The number of fold for CV
            **kwargs: Other keyword arguments for original `fit`

        Returns:
            conjurer.ml.Model
        """
        df = df.sample(len(df))  # shuffle
        x = df[feature_columns].values
        y = df[target_column].values
        self.cv = n_fold
        logger.warning("start learning with {} hyper parameters".format(self.n_iter))
        self.fit(x, y, **kwargs)
        return model.Model(self, feature_columns=feature_columns, target_column=target_column)


class GridSearchCV(model_selection.GridSearchCV):
    def __init__(self, estimator, param_grid, scoring=None,
                 n_jobs=None, pre_dispatch='2*n_jobs', cv=None, refit=True,
                 verbose=10, error_score=numpy.nan, return_train_score=True):
        super(GridSearchCV, self).__init__(
            estimator, param_grid, scoring=scoring, n_jobs=n_jobs, pre_dispatch=pre_dispatch,
            cv=cv, refit=refit, verbose=verbose, error_score=error_score, return_train_score=return_train_score
        )

    def fit_sv_pandas(self, df_training, target_column, feature_columns,
                      df_validation=None, ratio_training=None, **kwargs):
        """
        `fit` for pandas DataFrame to perform single validation
        Args:
            df_training (pandas.DataFrame): training data set
            target_column (str): column name of prediction target
            feature_columns (list of str): column names of features
            df_validation (pandas.DataFrame): if specified, used as validation data set
            ratio_training (float): if specified, `df_training` is split for training / validation
            **kwargs: Other keyword arguments for original `fit`

        Returns:
            conjurer.ml.Model
        """
        x, y, num_training, num_validation = _split_for_sv(
            df_training, target_column, feature_columns, df_validation, ratio_training)
        self.cv = model_selection.PredefinedSplit(
            numpy.array([-1] * num_training + [0] * num_validation))
        logger.warning("start learning with {} parameters".format(_get_num_parameters(self.param_grid)))
        self.fit(x, y, **kwargs)
        return model.Model(self, feature_columns=feature_columns, target_column=target_column)

    def fit_cv_pandas(self, df, target_column, feature_columns, n_fold, **kwargs):
        """
        `fit` for pandas DataFrame to perform cross validation
        Args:
            df (pandas.DataFrame): training data set
            target_column (str): column name of prediction target
            feature_columns (list of str): column names of features
            n_fold (int): The number of fold for CV
            **kwargs: Other keyword arguments for original `fit`

        Returns:
            conjurer.ml.Model
        """
        df = df.sample(len(df))  # shuffle
        x = df[feature_columns].values
        y = df[target_column].values
        self.cv = n_fold
        logger.warning("start learning with {} parameters".format(_get_num_parameters(self.param_grid)))
        self.fit(x, y, **kwargs)
        return model.Model(self, feature_columns=feature_columns, target_column=target_column)


def _split_for_sv(df_training, target_column, feature_columns, df_validation, ratio_training):
    if df_validation is not None:
        x = numpy.concatenate(
            (df_training[feature_columns].values, df_validation[feature_columns].values),
            axis=0
        )
        y = numpy.concatenate(
            (df_training[target_column].values, df_validation[target_column].values),
            axis=0
        )
        num_training = len(df_training)
        num_validation = len(df_validation)
    else:
        shuffled_df = df_training.sample(len(df_training))
        x = shuffled_df[feature_columns].values
        y = shuffled_df[target_column].values
        num_training = int(ratio_training * len(df_training))
        num_validation = len(df_training) - num_training
    return x, y, num_training, num_validation


def _get_num_parameters(param_grid):
    product = lambda list_values: reduce(mul, list_values, 1)
    return len(param_grid) if isinstance(param_grid, list) \
        else product([len(param_grid[elem]) for elem in param_grid.keys()])
