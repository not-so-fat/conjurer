import logging

import numpy
import pandas
from IPython.core.display import display
from plotly.offline import iplot
from plotly import graph_objs


COLOR_TRAINING = 'blue'
COLOR_VALIDATION = 'red'
COLOR_TIME = 'green'
logger = logging.getLogger(__name__)


class CVAnalyzer(object):
    def __init__(self, pipeline):
        self.param_names = get_param_names(pipeline.cv_results_)
        self.result_df = create_result_df(pipeline.cv_results_, self.param_names)
        self.metric_name = pipeline.scoring
        self.metric_minimize = False  # sklearn.metrics.SCORERS are to be maximized
        self._validate()

    def plot_flat(self):
        plot_flat(self.result_df, self.param_names, self.metric_name)

    def plot_by_param_all(self, use_logscale=False, use_box=False):
        for param_name in self.param_names:
            self.plot_by_param(param_name, use_logscale, use_box)

    def plot_by_param(self, param_name, use_logscale=False, use_box=False):
        # TODO(@not-so-fat) support categorical values
        if use_box:
            plot_by_param_box(self.result_df, param_name, self.metric_name, use_logscale)
        else:
            plot_by_param(self.result_df, param_name, self.metric_name, use_logscale)

    def get_summary(self):
        print("Best {} model".format(self.metric_name))
        print(self.get_best_model_info())
        print("Total training time: {} hours for {} params".format(self.result_df.fit_time.sum() / 3600, len(self.result_df)))
        display(self.get_param_stat_df())

    def get_best_model_info(self):
        best_model_row = self.result_df.sort_values(by="validation_score", ascending=self.metric_minimize).head(1)
        return dict(
            training=best_model_row.training_score.values[0],
            validation=best_model_row.validation_score.values[0],
            parameters={param_name: best_model_row["param_{}".format(param_name)].values[0] for param_name in self.param_names}
        )

    def get_param_stat_df(self):
        return pandas.concat(
            [get_parameter_wise_stat(self.result_df, param_name)
             for param_name in self.param_names], axis=0).sort_values(by="std (validation score)", ascending=False)

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
        "param_{}".format(key): [_transform_param_values_for_groupby(elem[key]) for elem in cv_result["params"]]
        for key in param_names
    }
    values_dict["training_score"] = cv_result["mean_train_score"]
    values_dict["validation_score"] = cv_result["mean_test_score"]
    values_dict["fit_time"] = cv_result["mean_fit_time"]
    values_dict["std_training_score"] = cv_result["std_train_score"]
    values_dict["std_validation_score"] = cv_result["std_test_score"]
    values_dict["std_fit_time"] = cv_result["std_fit_time"]
    return pandas.DataFrame(
        values_dict, columns=["param_{}".format(key) for key in param_names] \
                             + ["training_score", "validation_score", "fit_time",
                                "std_training_score", "std_validation_score", "std_fit_time"])


def plot_flat(result_df, param_names, metric_name):
    plot_df = result_df.sort_values(by="validation_score")
    point_text = ["\n".join(["{}: {}".format(
            param_name, row["param_{}".format(param_name)]) for param_name in param_names])
                 for index, row in plot_df.iterrows()]
    training_line = graph_objs.Scatter(x=point_text, y=plot_df.training_score, yaxis="y2",
            mode="lines+markers", name="training", line=dict(color=COLOR_TRAINING), marker=dict(color=COLOR_TRAINING))
    validation_line = graph_objs.Scatter(x=point_text, y=plot_df.validation_score,  yaxis="y2",
            mode="lines+markers", name="validation", line=dict(color=COLOR_VALIDATION), marker=dict(color=COLOR_VALIDATION))
    time_bar = graph_objs.Bar(x=point_text, y=plot_df.fit_time, name="time", marker=dict(color=COLOR_TIME))
    layout = graph_objs.Layout(
            height=600, width=800,
            title="training/validation {}".format(metric_name),
            xaxis=dict(title="", showgrid=False, showticklabels=False, zeroline=False),
            yaxis2=dict(title=metric_name, domain=[0.5, 1.0], showgrid=True, zeroline=True),
            yaxis=dict(title="computation time (sec)", domain=[0.0, 0.5], showgrid=True, zeroline=True)
    )
    fig = graph_objs.Figure(data=[training_line, validation_line, time_bar], layout=layout)
    iplot(fig, show_link=False)


def plot_by_param(result_df, param_name, metric_name, logscale_param=False):
    COLOR_DICT = {
        "training_score": "blue",
        "validation_score": "red",
        "fit_time": "green"
    }
    COLOR_OP_DICT = {
        "training_score": "rgba(0, 0, 150, 0.2)",
        "validation_score": "rgba(150, 0, 0, 0.2)",
        "fit_time": "rgba(0, 150, 0, 0.5)"
    }
    AXIS_DICT = {
        "training_score": "y2",
        "validation_score": "y2",
        "fit_time": "y"
    }

    def _get_trace_for_mean(mean_value, name, yaxis, color):
        xmin, xmax = (result_df["param_{}".format(param_name)].min(), result_df["param_{}".format(param_name)].max()) \
            if _is_numeric(result_df["param_{}".format(param_name)][0]) \
            else (0, len(result_df["param_{}".format(param_name)].unique()))
        return graph_objs.Scatter(
            x=[xmin, xmax],
            y=[mean_value, mean_value], name="mean({})".format(name), yaxis=yaxis, line=dict(color=color, dash="dash"),
            mode="lines", showlegend=False
        )

    def _get_trace_for_line_with_lb_ub(x_array, y_array, std_array, name, yaxis, color, color_std):
        return (
            graph_objs.Scatter(x=x_array, y=y_array, name=name, yaxis=yaxis, marker=dict(color=color),
                               mode="lines+markers"),
            graph_objs.Scatter(
                x=x_array, y=y_array - std_array, name="{}_lb".format(name), yaxis=yaxis, line=dict(color=color_std),
                showlegend=False),
            graph_objs.Scatter(
                x=x_array, y=y_array + std_array, name="{}_ub".format(name), yaxis=yaxis, line=dict(color=color_std),
                fillcolor=color_std, fill='tonexty', showlegend=False)
        )

    result_df = result_df.sort_values(by="param_{}".format(param_name))
    x_array = result_df["param_{}".format(param_name)].values
    names = ["training_score", "validation_score", "fit_time"]
    line_trace = {}
    lb_trace = {}
    ub_trace = {}
    mean_trace = {}
    for name in names:
        line_trace[name], lb_trace[name], ub_trace[name] = _get_trace_for_line_with_lb_ub(
            x_array, result_df[name].values, result_df["std_{}".format(name)].values, name,
            AXIS_DICT[name], COLOR_DICT[name], COLOR_OP_DICT[name])
        mean_trace[name] = _get_trace_for_mean(result_df[name].mean(), name, AXIS_DICT[name], COLOR_DICT[name])
    layout = graph_objs.Layout(
        width=800,
        height=600,
        title="{} / computation time vs {}".format(metric_name, param_name),
        xaxis=dict(title=param_name, showgrid=True, showticklabels=True, zeroline=False,
                   type='log' if logscale_param else 'linear'),
        yaxis2=dict(title=metric_name, domain=[0.5, 1.0], showgrid=True, zeroline=True),
        yaxis=dict(title="computation time (sec)", domain=[0.0, 0.5], showgrid=False)
    )
    fig = graph_objs.Figure(
        data=sum(
            [[line_trace[name], lb_trace[name], ub_trace[name], mean_trace[name]]
             for name in reversed(names)], []),
        layout=layout)
    iplot(fig, show_link=False)


def plot_by_param_box(result_df, param_name, metric_name, logscale_param=False):
    training_scatter = graph_objs.Box(
            x=result_df["param_{}".format(param_name)], y=result_df.training_score,
            name="training", yaxis="y2", marker=dict(color=COLOR_TRAINING))
    validation_scatter = graph_objs.Box(
            x=result_df["param_{}".format(param_name)], y=result_df.validation_score,
            name="validation", yaxis="y2", marker=dict(color=COLOR_VALIDATION))
    computation_time = graph_objs.Box(
            x=result_df["param_{}".format(param_name)], y=result_df.fit_time,
            name="time", marker=dict(color=COLOR_TIME))
    layout = graph_objs.Layout(
            width=800,
            height=600,
            title="{} / computation time vs {}".format(metric_name, param_name),
            xaxis=dict(title=param_name, showgrid=True, showticklabels=True, zeroline=False, type='log' if logscale_param else 'linear'),
            yaxis2=dict(title=metric_name, domain=[0.5, 1.0], showgrid=True, zeroline=True),
            yaxis=dict(title="computation time (sec)", domain=[0.0, 0.5], showgrid=False)
    )
    fig = graph_objs.Figure(data=[training_scatter, validation_scatter, computation_time], layout=layout)
    iplot(fig, show_link=False)


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
    if isinstance(value, list):
        return str(value)
    else:
        return value


def _is_numeric(obj):
    return True if obj is not None and \
                   numpy.issubdtype(numpy.array([obj]).dtype, numpy.number) else False
