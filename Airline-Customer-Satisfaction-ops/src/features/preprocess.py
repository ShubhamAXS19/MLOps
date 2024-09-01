# src/features/preprocess_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def preprocess_data(data_dir: str) -> tuple:
    """
    Preprocess the dataset: handle missing values, encode categorical features, and scale numeric features.
    """
    # Load the data
    data_file = os.path.join(data_dir, "airline_satisfaction.csv")
    df = pd.read_csv(data_file)
    
    features = df.drop(columns=["satisfaction"])
    target = df["satisfaction"]

    numeric_features = features.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = features.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    data_dir = "data/raw"
    preprocess_data(data_dir)
