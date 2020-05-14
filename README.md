# conjurer

Python library to help you to perform magic on your data analytics project; which helps
- EDA (load & check data)
- Automatic machine learning tuning

For detailed background please refer https://github.com/not-so-fat/conjurer/wiki

## Install

```
pip install conjurer
```

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

## supported ml algorithms

- LightGBM `lightgbm` (`gbm_autosplit.LGBMClassifier` or `gbm_autosplit.LGBMRegressor`)
- XGBoost `xgboost` (`gbm_autosplit.XGBClassfier` or `gbm_autosplit.XGBRegressor`)
- Random Forest `random_forest` (`sklearn.ensemble.RandomForestClassifier` or `sklearn.ensemble.RandomForestRegressor`)
- Lasso / Logistic Regression `linear_model` (`sklearn.linear_model.Lasso` or `sklearn.linear_model.LogisticRegression`)

This module uses CV by `sklearn_cv_pandas.RandomizedSearchCV` or `sklearn_cv_pandas.GridSearchCV` to use 
pandas.DataFrame for arguments
