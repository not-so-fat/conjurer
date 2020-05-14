from conjurer.logic.eda import check
from conjurer.logic.eda.load import (
    pandas_csv
)
from conjurer.logic.eda.load import df_dict_loader
from conjurer.logic.eda.vis import (
    stacked_bar
)
from conjurer.logic.eda.vis import histogram, scatter


def read_csv(buffer_or_filepath, **kwargs):
    """
    Load CSV as pandas.DataFrame with Int64 / datetime64 column inference
    Args:
        buffer_or_filepath: 1st argument for pandas.read_csv
        **kwargs: Arbitrary keyword arguments for pandas.read_csv

    Returns:
        pandas.DataFrame
    """
    return pandas_csv.read_csv(buffer_or_filepath, **kwargs)


def check_stats(df, skip_histogram=False):
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


def get_unique_values(df, columns):
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


def get_columns_in_dfs(df_list, name_list):
    """
    Summarize df names and column name in multiple pandas.DataFrame
    Args:
        df_list (list of pandas.DataFrame): List of data frames
        name_list (list of str): Name of each data frame

    Returns:
        pandas.DataFrame: with columns "table_name" and "column_name"
    """
    return check.get_columns_in_dfs(df_list, name_list)


def get_fk_coverage(fk_df, k_df, fk_columns, k_columns, do_print=True):
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


def plot_histogram(array, num_bins=50, normalize=False, minv=None, maxv=None, title=None, layout={}):
    """
    Plot histogram with plot.ly
    Args:
        array (numpy.array): Array you want to plot histogram
        num_bins (int, optional): Default=50. The number of bins used in the histogram
        normalize (bool, optional): Default=False. If True, vertical axis is ratio of records, else the number of records in each bin
        minv (optional): Default=Minimum value in array. Minimum value to generate bins
        maxv (optional): Default=Maximum value in array. Maximum value to generate bins
        title (optional): Graph title
        layout (optional): Arguments you want to pass into graph layout

    Returns:
        None
    """
    histogram.plot_histogram(array, num_bins, normalize, minv, maxv, title, layout)


def plot_scatter(xarray, yarray, xname=None, yname=None, mode="markers", same_scale=False,
                 xerror=None, yerror=None, layout={}):
    """
    Plot scatter plot with plot.ly
    Args:
        xarray (numpy.array): Array for horizontal axis
        yarray (numpy.array): Array for vertical axis
        xname (str, optional): Title for horizontal axis
        yname (str, optional): Title for vertical axis
        mode (str, optional): Default="markers". Mode used in plot.ly
        same_scale (bool, optional): Default=False. If True make horizontal & vertical scale same
        xerror (numpy.array, optional): If specified, add horizontal error bar in the graph
        yerror (numpy.array, optional): If specified, add vertical error bar in the graph
        layout (optional): Arguments you want to pass into graph layout

    Returns:
        None
    """
    scatter.plot_scatter(xarray, yarray, xname, yname, mode, same_scale, xerror, yerror, layout)


def plot_scatter_multiple(xarray_list, yarray_list, xname=None, yname=None, name_list=None,
                          mode="markers", same_scale=False, layout={}):
    """
    Plot scatter plot for multiple pairs of arrays with plot.ly
    Args:
        xarray_list (list of numpy.array): Array for horizontal axis
        yarray_list (list of numpy.array): Array for vertical axis
        xname (str, optional): Title for horizontal axis
        yname (str, optional): Title for vertical axis
        name_list (list of srt, optional): Name for each scatter plot (used in legend)
        mode (str, optional): Default="markers". Mode used in plot.ly
        same_scale (bool, optional): Default=False. If True make horizontal & vertical scale same
        layout (optional): Arguments you want to pass into graph layout

    Returns:
        None
    """
    scatter.plot_scatter_multiple(xarray_list, yarray_list, xname, yname, name_list, mode, same_scale, layout)


def plot_multiple_bars(xarray, yarray_list, name_list=None, xname=None, yname=None, single=True, layout={}):
    """
    Plot bar charts with plot.ly
    Args:
        xarray (numpy.array): Array for horizontal axis
        yarray_list (list of numpy.array): Array for vertical axis
        name_list (list of str): Name for each bar (used in legend)
        xname (str, optional): Title for horizontal axis
        yname (str, optional): Title for vertical axis
        single (str, optional): Default=True. If True, stacked bar in single chart, else plot many charts
        layout (optional): Arguments you want to pass into graph layout

    Returns:
        None
    """
    stacked_bar.plot_stacked_bars(xarray, yarray_list, name_list, xname, yname, single, layout)


DfDictLoader = df_dict_loader.DfDictLoader
