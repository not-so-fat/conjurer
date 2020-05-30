import pandas
from pandas.api import types
from IPython.core.display import display

from conjurer.logic.eda.check import stat_calculator
from conjurer.logic.eda.vis import histogram


def check_stats(df, skip_histogram=False):
    print("[table-wise confirmation]")
    print("shape: {}x{}".format(len(df), len(df.columns)))
    print("duplication: {}".format(len(df[df.duplicated()])))
    print("[column-wise confirmation]")
    stat_df = stat_calculator.calc_column_stat(df)
    display(stat_df)
    if not skip_histogram:
        print("[histogram]")
        histogram.plot_histogram_for_stats(df, stat_df)
    print("[head]")
    display(df.head())
    return stat_df


def get_unique_values(df, columns):
    return stat_calculator.get_unique_values(df, columns)


def get_columns_in_dfs(df_list, name_list):
    return pandas.concat([
        pandas.DataFrame({
            "table_name": [name] * len(df.columns),
            "column_name": df.columns
        })
        for df, name in zip(df_list, name_list)
    ], ignore_index=True)


def get_fk_coverage(fk_df, k_df, fk_columns, k_columns, do_print=True):
    """
    Check how many keys in the first df exist in the second df
    Parameters
    ----------
    fk_df : pandas.DataFrame
    k_df : pandas.DataFrame
    fk_columns : str or list of str
    k_columns : str or list of str
    do_print : bool

    Returns
    -------
    float

    """
    keys_in_fk_df = get_unique_values(fk_df, fk_columns)
    keys_in_k_df = get_unique_values(k_df, k_columns)
    num_intersection = len(keys_in_fk_df.intersection(keys_in_k_df))
    num_target_keys = len(keys_in_fk_df)
    if do_print:
        print("{:.2%} ({} / {})".format(num_intersection / num_target_keys, num_intersection, num_target_keys))
    return num_intersection / num_target_keys
