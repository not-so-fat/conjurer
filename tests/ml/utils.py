from numpy import random
import pandas


def get_input_df(sample_size):
    return pandas.DataFrame({
        **{
            "id1": range(sample_size),
            "id2": range(10000, 10000+sample_size),
            "target_cl": random.choice([1, 0], sample_size),
            "target_rg": random.normal(0, 1, sample_size),
        },
        **{
            "column{}".format(i): random.normal(0, 1, sample_size)
            for i in range(3)
        },
        **{
            "column{}".format(i): random.choice([1, 0], sample_size)
            for i in range(3, 6)
        }
    })


def get_xy(sample_size, for_cl):
    input_df = get_input_df(sample_size)
    feature_columns = ["column{}".format(i) for i in range(6)]
    target_column = "target_cl" if for_cl else "target_rg"
    return input_df[feature_columns].values, input_df[target_column].values
