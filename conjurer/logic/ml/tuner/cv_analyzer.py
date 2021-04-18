import logging

import numpy
import pandas
from IPython.core.display import display
import altair as alt


logger = logging.getLogger(__name__)
METRIC_MINIMIZE = False  # sklearn.metrics.SCORERS are to be maximized


class CVAnalyzer(object):
    def __init__(self, pipeline):
        self.param_names = get_param_names(pipeline.cv_results_)
        self.result_df = create_result_df(pipeline.cv_results_, self.param_names)
        self.metric_name = pipeline.scoring
        self._validate()

    def plot_flat(self):
        return plot_flat(self.result_df, self.param_names, self.metric_name)

    def plot_by_param_all(self):
        for param_name in self.param_names:
            self.plot_by_param(param_name).display()

    def plot_by_param(self, param_name):
        return plot_by_param(self.result_df, param_name, self.metric_name, self.param_names)

    def get_summary(self):
        print("Best {} model".format(self.metric_name))
        print(self.get_best_model_info())
        print("Total training time: {} hours for {} params".format(
            self.result_df.fit_time.sum() / 3600, len(self.result_df)))
        display(self.get_param_stat_df())

    def get_best_model_info(self):
        best_model_row = self.result_df.sort_values(
                by="validation_score", ascending=METRIC_MINIMIZE).head(1)
        return dict(
            training=best_model_row.training_score.values[0],
            validation=best_model_row.validation_score.values[0],
            parameters={
                param_name: best_model_row["param_{}".format(param_name)].values[0]
                for param_name in self.param_names
            }
        )

    def get_param_stat_df(self):
        return pandas.concat(
            [get_parameter_wise_stat(self.result_df, param_name)
             for param_name in self.param_names], axis=0).sort_values(
                     by="std (validation score)", ascending=False)

    def _validate(self):
        def _warning(is_min):
            val = self.result_df["param_{}".format(param_name)].min() if is_min \
                else self.result_df["param_{}".format(param_name)].max()
            min_or_max = "minimum" if is_min else "maximum"
            if val == info_dict["parameters"][param_name]:
                logger.warning(
                    "Best performance is achieved at {} value of search: {}={}".format(
                        min_or_max, param_name, info_dict["parameters"][param_name]
                    ))

        info_dict = self.get_best_model_info()
        for param_name in self.param_names:
            if _is_numeric(info_dict["parameters"][param_name]):
                _warning(True)
                _warning(False)

        
def create_result_df(cv_result, param_names):
    values_dict = {
        "param_{}".format(key): [
            _transform_param_values_for_groupby(elem[key]) 
            for elem in cv_result["params"]
        ]
        for key in param_names
    }
    values_dict["training_score"] = cv_result["mean_train_score"]
    values_dict["validation_score"] = cv_result["mean_test_score"]
    values_dict["fit_time"] = cv_result["mean_fit_time"]
    values_dict["std_training_score"] = cv_result["std_train_score"]
    values_dict["std_validation_score"] = cv_result["std_test_score"]
    values_dict["std_fit_time"] = cv_result["std_fit_time"]
    return pandas.DataFrame(
        values_dict, 
        columns=["param_{}".format(key) for key in param_names] \
                + ["training_score", "validation_score", "fit_time",
                    "std_training_score", "std_validation_score", "std_fit_time"]
    )


def plot_flat(result_df, param_names, metric_name):
    melt_df = create_melt_df(result_df, param_names)
    return plot_metric_and_time(melt_df, "index", metric_name, param_names)


def plot_metric_and_time(melt_df, column_x, metric_name, param_names):
    points = alt.Chart(melt_df[melt_df["variable"].isin(["training_score", "validation_score"])]).encode(
        x=column_x, y=alt.Y("value", title=metric_name, scale=alt.Scale(zero=False)), color="variable"
    ).mark_circle(opacity=0.4)
    mean_df = melt_df.groupby(by=["variable", column_x]).mean().reset_index().sort_values(by=column_x)
    mean_chart = alt.Chart(
        mean_df[mean_df["variable"].isin(["training_score", "validation_score"])]).encode(
        x=column_x, y=alt.Y("value", title=metric_name, scale=alt.Scale(zero=False)),
        yError="std_value", color="variable")
    lines = mean_chart.mark_line(strokeDash=[2, 2])
    error = mean_chart.mark_errorband()
    time = alt.Chart(melt_df[melt_df["variable"].isin(["fit_time"])]).mark_bar().encode(
        x=column_x, y=alt.Y("value", title="fit time (sec)"),
        yError="std_value", tooltip=["param_{}".format(param_name) for param_name in param_names]
    )
    return alt.vconcat(
        alt.layer(points, lines+error).properties(height=150, width=400),
        time.properties(height=150, width=400)
    ).resolve_scale(x="shared")


def plot_by_param(result_df, param_name, metric_name, param_names):
    melt_df = create_melt_df(result_df, param_names)
    return plot_metric_and_time(melt_df, "param_{}".format(param_name), metric_name, param_names)


def create_melt_df(result_df, param_names):
    plot_df = result_df.sort_values(by="validation_score")
    plot_df["index"] = range(len(plot_df))
    plot_df["index"] = range(len(plot_df))
    melt_df1 = plot_df.melt(["index"] + ["param_{}".format(param_name) for param_name in param_names],
                            ["training_score", "validation_score", "fit_time"])
    melt_df2 = plot_df.melt(["index"], ["std_training_score", "std_validation_score", "std_fit_time"]).rename(
        columns={"value": "std_value"}
    )
    melt_df2["variable"] = melt_df2["variable"].str.replace("std_", "")
    return melt_df1.merge(melt_df2, on=["index", "variable"])


def get_parameter_wise_stat(result_df, param_name):
    param_column_name = "param_{}".format(param_name)
    param_values = numpy.sort(result_df[param_column_name].unique())
    mean_metrics = result_df[[param_column_name, "training_score", "validation_score", "fit_time"]].groupby(by=param_column_name).mean()
    return pandas.DataFrame({
            "name": [param_name],
            "uniqueParamValues": [len(param_values)],
            "std (training score)": [numpy.std(mean_metrics.training_score)],
            "std (validation score)": [numpy.std(mean_metrics.validation_score)],
            "std (computation time)": [numpy.std(mean_metrics.fit_time)]}).set_index("name")


def get_param_names(cv_results):
    param_names = set([])
    for elem in cv_results["params"]:
        param_names = param_names.union(set(elem.keys()))
    return list(param_names)


def _transform_param_values_for_groupby(value):
    if value is None:
        return "None"
    elif isinstance(value, list):
        return str(value)
    else:
        return value


def _is_numeric(obj):
    return True if obj is not None and numpy.issubdtype(numpy.array([obj]).dtype, numpy.number) else False
