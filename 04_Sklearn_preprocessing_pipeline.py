import os
import zipfile
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

os.system("kaggle competitions download -c titanic")

with zipfile.ZipFile("titanic.zip", "r") as zip_ref:
    zip_ref.extractall("titanic_data")

df = pd.read_csv("titanic_data/train.csv")

X = df.drop(columns=["Survived"])
y = df["Survived"]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("drop_columns", "drop", ["PassengerId", "Name", "Ticket", "Cabin"]),
        ("numeric", numeric_pipeline, make_column_selector(dtype_include=["int64", "float64"])),
        ("categorical", categorical_pipeline, make_column_selector(dtype_include=["object"]))
    ]
)

X_processed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()
processed_df = pd.DataFrame(X_processed, columns=feature_names)

print("Data preprocessing completed.")
print(processed_df.head())
print(f"Processed dataset shape: {processed_df.shape}")