# conjurer

Python library to help you to perform magic on your data analytics project; which helps
- EDA (load & check data)
- Automatic machine learning tuning

For detailed background please refer https://github.com/not-so-fat/conjurer/wiki


## Usage

You can build prediction pipeline from multiple data sources with following simple code. 
```
from conjurer import (
    eda,
    ml
)

# Load CSVs as pandas.DataFrame
df_dict = {
    name: eda.read_csv("{}_training.csv".format(name)
    for name in ["target", "demand_history", "product", "customer"]
}

# Do feature engineering (not implemented)
feature_training, feature_names = engineer_feature(df_dict)

# Automatic lightgbm tuning 
model = ml.tune_cv("lightgbm", "rg", feature_training, "sales_amount", feature_names, 5)
```

and produce prediction results.

```
# Load CSV files for test data set as the same data types as training
loader = eda.DfDictLoader(df_dict)
df_dict_test = loader.load({
    name: "{}_test.csv".format(name)
    for name in ["target", "demand_history", "product", "customer"]
})

# Feature generation for test data set (not implemented)
feature_test = generate_feature(df_dict)

# Get prediction on test data set
model.predict(feature_test)
```

## Road Map

Add automatic feature engineering from multiple data sources.
