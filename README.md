# conjurer
Interface for all data analytics tools


# Key Functionalities

## EDA (Data Load and Check)

```buildoutcfg
from conjurer import (
    eda,
    feature,
    ml
)

# Load CSVs as pandas.DataFrame (but automatically infer Int64 / datetime64 columns)

df_dict = {
    name: eda.read_csv("{}.csv".format(name)
    for name in ["target_training", "target_validation", "demand_history", "product", "customer"]
}

# Check columns in dfs (same key in different dfs?)
columns_df = eda.get_columns_in_dfs(df_dict.values(), df_dict.keys())
columns_df["column_name"].value_counts()

# Check key coverage in dfs
eda.get_fk_coverage(df_dict["target_training"], df_dict["demand_history"], "customer_id", "customer_id")
eda.get_fk_coverage(df_dict["target_training"], df_dict["customer"], "customer_id", "customer_id")
eda.get_fk_coverage(df_dict["demand_history"], df_dict["product"], "product_id", "product_id")
eda.get_fk_coverage(df_dict["demand_history"], df_dict["customer"], "customer_id", "customer_id")

# Feature engineering
df_dict_training = {
    "target": df_dict["target_training"],
    "demand_history": df_dict["demand_history"],
    "product": df_dict["product"],
    "customer": df_dict["customer"]
}
keys_list = [
    {"target": "customer_id", "demand_history": "customer_id", "customer": "customer_id"},
    {"demand_history": "product_id", "product": "product_id"}
]
f_conjurer = feature.FeatureConjurer()
f_conjurer.fit(df_dict_training, "target", "sales_amount", keys_list)
feature_training = f_conjurer.transform(df_dict_training)

# feature generation for validation data set
df_dict_validation = {
    "target": df_dict["target_validation"],
    "demand_history": df_dict["demand_history"],
    "product": df_dict["product"],
    "customer": df_dict["customer"]
}
feature_validation = f_conjurer.transform(df_dict_validation)

# Machine learning

# Train models by 5-fold CV
ml_tuner = ml.LGBMRGTuner()
model = ml_tuner.tune_cv_pandas(feature_training, "sales_amount", [f.name for f in f_conjure.features], 5)

# Make prediction on validation data set
model.predict_pandas(feature_validation)

```


