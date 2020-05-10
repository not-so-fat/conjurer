# conjurer

Python library to help you to perform magic on your data analytics project; which helps
- EDA (load & check data)
- Automatic feature engineering
- Automatic machine learning tuning

For detailed background please refer https://github.com/not-so-fat/conjurer/wiki


## Usage

Just by following simple code, you can build prediction pipeline!

```
from conjurer import (
    eda,
    feature,
    ml
)

# Load CSVs as pandas.DataFrame
df_dict = {
    name: eda.read_csv("{}_training.csv".format(name)
    for name in ["target", "demand_history", "product", "customer"]
}

# Specify keys which can be used for joining tables
keys_list = [
    {"target": "customer_id", "demand_history": "customer_id", "customer": "customer_id"},
    {"demand_history": "product_id", "product": "product_id"}
]

# Automatic feature engineering
f_conjurer = feature.FeatureConjurer()
f_conjurer.fit(df_dict, "target", "sales_amount", keys_list)
feature_training = f_conjurer.transform(df_dict_training)

# Automatic lightgbm tuning 
lgbm_tuner = ml.LGBMRGTuner()
model = lgbm_tuner.tune_cv_pandas(feature_training, "sales_amount", [f.name for f in f_conjure.features], 5)
```

and produce prediction results!

```
# Load CSV files for test data set as the same data types as training
loader = eda.DfDictLoader(df_dict)
df_dict_test = loader.load({
    name: "{}_test.csv".format(name)
    for name in ["target", "demand_history", "product", "customer"]
})

# Feature generation for test data set
feature_test = f_conjurer.transform(df_dict_test)

# Get prediction on test data set
model.predict_pandas(feature_test)
```

