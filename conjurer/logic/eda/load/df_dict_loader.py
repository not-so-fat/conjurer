from conjurer.logic.eda.load import pandas_csv


class DfDictLoader(object):
    """
    Class to load df dictionary as the same dtypes as previous dictionary
    """
    def __init__(self, df_dict):
        self.dtypes = {}
        for name in df_dict.keys():
            self.dtypes[name] = df_dict[name].dtypes

    def load(self, filepath_dict):
        self._validate(filepath_dict)
        return {
            name: pandas_csv.read_csv(filepath_dict[name], dtype=self.dtypes[name])
            for name in filepath_dict.keys()
        }

    def _validate(self, filepath_dict):
        assert set(filepath_dict.keys()) == set(self.dtypes.keys())
