"""Exploratory visualizations for UrbanExpress delivery data.

Example:
    python eda_visualization.py --input data/deliveries.csv --output-dir reports/eda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the delivery extract and validate that it is readable."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")
    return pd.read_csv(input_path)


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric and non-numeric columns."""
    return data.describe(include="all").transpose()


def create_plots(data: pd.DataFrame, output_dir: str | Path) -> None:
    """Create the three core EDA charts when their source columns are available."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    required = {"distance_km", "delivery_cost"}
    if required.issubset(data.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x="distance_km", y="delivery_cost", hue="traffic_level" if "traffic_level" in data else None, alpha=0.7)
        plt.title("Delivery cost versus distance")
        plt.tight_layout()
        plt.savefig(destination / "cost_vs_distance.png", dpi=150)
        plt.close()

    if {"weather_condition", "transit_duration_minutes"}.issubset(data.columns):
        plt.figure(figsize=(11, 6))
        sns.boxplot(data=data, x="weather_condition", y="transit_duration_minutes")
        plt.title("Transit duration by weather condition")
        plt.xlabel("Weather condition")
        plt.ylabel("Transit duration (minutes)")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(destination / "weather_impact_boxplot.png", dpi=150)
        plt.close()

    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric.corr(), annot=True, cmap="vlag", center=0, fmt=".2f")
        plt.title("Correlation between numeric features")
        plt.tight_layout()
        plt.savefig(destination / "feature_correlation_heatmap.png", dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the delivery CSV")
    parser.add_argument("--output-dir", default="reports/eda", help="Directory for reports and plots")
    args = parser.parse_args()

    data = load_data(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = descriptive_statistics(data)
    stats.to_csv(output_dir / "descriptive_statistics.csv")
    create_plots(data, output_dir)
    print(stats)
    print(f"EDA outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
