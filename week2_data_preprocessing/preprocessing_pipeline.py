"""Prepare UrbanExpress delivery records for analysis and modeling.

The pipeline performs conservative, reusable transformations:
* imputes numeric and categorical missing values;
* converts delivery timestamps and derives transit duration;
* removes numeric outliers with the IQR rule; and
* applies Min-Max scaling to numeric columns.

Example:
    python preprocessing_pipeline.py --input deliveries.csv --output processed.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


DEFAULT_NUMERIC_COLUMNS = ["distance_km", "delivery_cost"]
DEFAULT_CATEGORICAL_COLUMNS = ["weather_condition", "vehicle_type", "traffic_level"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Read a CSV and fail with a useful message when it is missing."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")
    return pd.read_csv(input_path)


def prepare_timestamps(data: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps and create transit_duration_minutes when possible."""
    result = data.copy()
    for column in ("order_timestamp", "pickup_timestamp", "delivery_timestamp"):
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce", utc=True)

    if {"order_timestamp", "delivery_timestamp"}.issubset(result.columns):
        duration = (
            result["delivery_timestamp"] - result["order_timestamp"]
        ).dt.total_seconds() / 60
        # Negative durations are invalid operational records and become missing.
        result["transit_duration_minutes"] = duration.where(duration >= 0)
    return result


def impute_missing_values(
    data: pd.DataFrame,
    numeric_columns: Iterable[str],
    categorical_columns: Iterable[str],
) -> pd.DataFrame:
    """Fill numeric gaps with medians and categorical gaps with a sentinel label."""
    result = data.copy()
    for column in numeric_columns:
        if column in result:
            median = result[column].median()
            result[column] = result[column].fillna(0 if pd.isna(median) else median)
    for column in categorical_columns:
        if column in result:
            result[column] = result[column].astype("string").fillna("Unknown")
    return result


def remove_iqr_outliers(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Keep rows inside each column's Tukey fences (1.5 times the IQR)."""
    result = data.copy()
    mask = pd.Series(True, index=result.index)
    for column in columns:
        if column not in result:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        mask &= values.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return result.loc[mask].reset_index(drop=True)


def min_max_scale(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Scale available numeric columns to [0, 1], retaining their column names."""
    result = data.copy()
    available = [column for column in columns if column in result]
    if available:
        scaler = MinMaxScaler()
        result[available] = scaler.fit_transform(result[available])
    return result


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Run the complete preprocessing workflow."""
    numeric = DEFAULT_NUMERIC_COLUMNS + ["transit_duration_minutes"]
    prepared = prepare_timestamps(data)
    prepared = impute_missing_values(prepared, numeric, DEFAULT_CATEGORICAL_COLUMNS)
    prepared = remove_iqr_outliers(prepared, numeric)
    return min_max_scale(prepared, numeric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the raw delivery CSV")
    parser.add_argument("--output", required=True, help="Path for the processed CSV")
    args = parser.parse_args()

    processed = preprocess(load_data(args.input))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # ISO timestamps are portable; omit the index because it is not a business field.
    processed.to_csv(output_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    print(f"Saved {len(processed):,} processed records to {output_path}")


if __name__ == "__main__":
    main()
