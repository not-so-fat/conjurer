from typing import Union
from datetime import (
    date,
    datetime
)

import pandas
import altair
from pandas._typing import FilePathOrBuffer

from conjurer.logic.eda import check
from conjurer.logic.eda.load import (
    pandas_csv,
    df_dict_loader
)
from conjurer.logic.eda.vis import (
    histogram,
    scatter
)


StrOrList = Union[str, list]
Orderable = Union[int, float, datetime, date]


def read_csv(buffer_or_filepath: FilePathOrBuffer, **kwargs) -> pandas.DataFrame:
    """
    Load CSV as pandas.DataFrame with Int64 / datetime64 column inference
    Args:
        buffer_or_filepath: 1st argument for pandas.read_csv
        **kwargs: Arbitrary keyword arguments for pandas.read_csv

    Returns:
        pandas.DataFrame
    """
    return pandas_csv.read_csv(buffer_or_filepath, **kwargs)


def check_stats(df: pandas.DataFrame, skip_histogram: bool = False) -> pandas.DataFrame:
    """
    Calculate basic statistics for pandas.DataFrame
    Args:
        df (pandas.DataFrame): Data frame you want to calculate statistics
        skip_histogram (bool, optional): If True, function does not generate histogram

    Returns:
        pandas.DataFrame: Data frame of statistics for each column

        Each row represents statistics of each column in `df`
    """
    return check.check_stats(df, skip_histogram)


def get_unique_values(df: pandas.DataFrame, columns: StrOrList) -> set:
    """
    Get unique values (not null) for column(s) in pandas.DataFrame as set
    Args:
        df (pandas.DataFrame): Data frame which stores data
        columns (str or list of str): Column name(s) you want to confirm unique values

    Returns:
        set: unique values (only values which does NOT include any nulls)

    Notes:
        like https://github.com/numpy/numpy/issues/9358, "unique" concept for null is ambiguous,
        values which includes null are all eliminated from returned value
    """
    return check.get_unique_values(df, columns)


def get_columns_in_dfs(df_list: list, name_list: list) -> pandas.DataFrame:
    """
    Summarize df names and column name in multiple pandas.DataFrame
    Args:
        df_list (list of pandas.DataFrame): List of data frames
        name_list (list of str): Name of each data frame

    Returns:
        pandas.DataFrame: with columns "table_name" and "column_name"
    """
    return check.get_columns_in_dfs(df_list, name_list)


def get_fk_coverage(
        fk_df: pandas.DataFrame, k_df: pandas.DataFrame, fk_columns: StrOrList, k_columns: StrOrList,
        do_print: bool = True) -> float:
    """
    Check how many keys in `fk_df.fk_columns` exists in `k_df.k_columns`
    Args:
        fk_df (pandas.DataFrame): Data frame which has FK columns
        k_df (pandas.DataFrame): Data frame which has K columns
        fk_columns (str or list of str): FK columns
        k_columns (str or list of str): K columns
        do_print (bool): If True, print results

    Returns:
        float: ratio of records in `fk_df.fk_columns` which has records in `k_df.k_columns`
    """
    return check.get_fk_coverage(fk_df, k_df, fk_columns, k_columns, do_print)


def plot_histogram(series: pandas.Series, num_bins: int = 50, normalize: bool = False,
                   minv: Orderable = None, maxv: Orderable = None) -> altair.Chart:
    """
    Plot histogram with altair
    Args:
        series (pandas.Series): Values you want to plot histogram
        num_bins (int, optional): Default=50. The number of bins used in the histogram
        normalize (bool, optional): Default=False. If True, vertical axis is ratio of records, else the number of records in each bin
        minv (optional): Default=Minimum value in array. Minimum value to generate bins
        maxv (optional): Default=Maximum value in array. Maximum value to generate bins

    Returns:
       altair.Chart
    """
    return histogram.plot_histogram(series, num_bins, normalize, minv, maxv)


def plot_scatter(df: pandas.DataFrame, column_x: str, column_y: str, **kwargs) -> altair.Chart:
    """
    Plot scatter plot with altair. If sample size is too large, plot heatmap instead
    Args:
        df(pandas.DataFrame): Data frame which stores all the data used in the chart
        column_x(str): Column name for x value
        column_y(str): Column name for y value
        num_bins_x(int): The number of bins (used for heatmap)
        num_bins_y(int): The number of bins (used for heatmap)
        xmin(numeric or timestamp): Minimum X value of plot
        xmax(numeric or timestamp): Maximum X value of plot
        ymin(numeric or timestamp): Minimum Y value of plot
        ymax(numeric or timestamp): Maximum Y value of plot

    Returns:
        altair.Chart
    """
    return scatter.plot_scatter(df, column_x, column_y, **kwargs)


def plot_heatmap(df: pandas.DataFrame, column_x: str, column_y: str, num_bins_x: int = 50, num_bins_y: int = 50,
                 xmin: Orderable = None, xmax: Orderable = None, ymin: Orderable = None, ymax: Orderable = None) \
        -> altair.Chart:
    """
    Plot heatmap with altair.
    Args:
        df(pandas.DataFrame): Data frame which stores all the data used in the chart
        column_x(str): Column name for x value
        column_y(str): Column name for y value
        num_bins_x(int): The number of bins
        num_bins_y(int): The number of bins
        xmin(numeric or timestamp): Minimum X value of plot
        xmax(numeric or timestamp): Maximum X value of plot
        ymin(numeric or timestamp): Minimum Y value of plot
        ymax(numeric or timestamp): Maximum Y value of plot

    Returns:
        altair.Chart
    """
    return scatter.plot_heatmap(df, column_x, column_y, num_bins_x, num_bins_y, xmin, xmax, ymin, ymax)


DfDictLoader = df_dict_loader.DfDictLoader
