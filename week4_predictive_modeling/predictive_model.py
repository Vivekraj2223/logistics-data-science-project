"""Predict UrbanExpress delivery transit duration with a Random Forest.

The pipeline keeps imputation, encoding, and model fitting together to prevent
information leakage from the test set. Categorical columns are one-hot encoded;
numeric columns are median-imputed and standardized.

Example:
    python predictive_model.py --input data/deliveries_processed.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "transit_duration_minutes"
DEFAULT_CATEGORICAL = ["weather_condition", "vehicle_type", "traffic_level"]
DEFAULT_NUMERIC = ["distance_km", "delivery_cost"]


def build_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    """Build preprocessing and Random Forest steps as one estimator."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    features = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", features),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )


def train_and_evaluate(data: pd.DataFrame) -> dict[str, float]:
    """Split data, train the model, and return standard regression metrics."""
    if TARGET not in data:
        raise ValueError(f"The input must contain the target column '{TARGET}'.")

    numeric = [column for column in DEFAULT_NUMERIC if column in data]
    categorical = [column for column in DEFAULT_CATEGORICAL if column in data]
    if not numeric and not categorical:
        raise ValueError("No supported feature columns were found in the input.")

    model_data = data[numeric + categorical + [TARGET]].dropna(subset=[TARGET])
    X = model_data[numeric + categorical]
    y = model_data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(numeric, categorical)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    return {
        "rmse_minutes": float(mean_squared_error(y_test, predictions) ** 0.5),
        "mae_minutes": float(mean_absolute_error(y_test, predictions)),
        "r2_score": float(r2_score(y_test, predictions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the processed delivery CSV")
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")

    metrics = train_and_evaluate(pd.read_csv(input_path))
    print("Model evaluation")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
