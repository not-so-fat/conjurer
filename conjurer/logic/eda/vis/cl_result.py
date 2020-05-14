import numpy
import pandas

from conjurer.logic.eda.vis import scatter


def plot_pr_curve(actual, score, xaxis_rank=False, subsample=-1, layout={}):
    cl_result_df = create_cl_result_df(actual, score)
    xname = "rank" if xaxis_rank else "score"
    if 0 < subsample < len(cl_result_df):
        indices = [int(i) for i in numpy.linspace(0, len(cl_result_df)-1, subsample)]
        cl_result_df = cl_result_df.iloc[indices, :]
    xvalues = cl_result_df[xname]
    layout["hovermode"] = "x"
    scatter.plot_scatter_multiple(
        [xvalues, xvalues], [cl_result_df[name].values for name in ["precision", "recall"]],
        xname=xname, yname="precision/recall", name_list=["precision", "recall"], mode="lines", layout=layout
    )


def create_cl_result_df(actual, score, positive_label=1):
    df = pandas.DataFrame({
        "actual": actual,
        "score": score
    }).sort_values(by="score", ascending=False)
    df["rank"] = [i+1 for i in range(len(df))]
    df["tmp_positive"] = 0
    df.loc[df["actual"]==positive_label, "tmp_positive"] = 1
    df["cum_tp"] = df["tmp_positive"].cumsum()
    df["cum_fp"] = (1 - df["tmp_positive"]).cumsum()
    df["precision"] = df["cum_tp"] / df["rank"]
    df["recall"] = df["cum_tp"] / len(df[df["actual"]==positive_label])
    return df.drop(columns=["tmp_positive"])
