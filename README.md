# UrbanExpress Logistics Data Science Project

UrbanExpress is a fictional urban retail distribution network serving same-day and next-day deliveries across a dense city. This repository documents a four-week data science workflow for understanding delivery performance, preparing operational data, visualizing logistics patterns, and predicting delivery transit duration.

## Project goals

- Improve visibility into delivery cost, distance, weather, and transit-time drivers.
- Establish a repeatable data-preparation workflow for operational records.
- Identify relationships and bottlenecks through exploratory analysis.
- Forecast transit duration so dispatch teams can plan capacity and communicate reliable ETAs.

## Project structure

```text
logistics-data-science-project/
├── README.md
├── week1_strategic_planning/
│   └── week1_strategy_report.md
├── week2_data_preprocessing/
│   └── preprocessing_pipeline.py
├── week3_eda_visualization/
│   └── eda_visualization.py
└── week4_predictive_modeling/
    └── predictive_model.py
```

### Weekly deliverables

1. **Strategic planning** — Define UrbanExpress's business questions, KPIs, assumptions, and analysis approach.
2. **Data preprocessing** — Clean delivery data, convert timestamps, derive transit duration, remove IQR outliers, and scale numerical features.
3. **Exploratory data analysis** — Produce descriptive statistics and visualizations for cost, distance, weather, and feature relationships.
4. **Predictive modeling** — Train and evaluate a Random Forest regression pipeline with separate categorical and numerical preprocessing.

## Expected input data

The scripts accept a CSV file. Common columns are:

- `order_id`
- `order_timestamp` and `delivery_timestamp`
- `distance_km`
- `delivery_cost`
- `weather_condition`
- `vehicle_type`
- `traffic_level`
- `transit_duration_minutes` (optional for preprocessing; required as the target for modeling)

Column names can be changed with command-line arguments where indicated in each script. The scripts also include small, documented defaults so they can be imported and adapted for a real UrbanExpress extract.

## Installation and usage

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

python week2_data_preprocessing/preprocessing_pipeline.py \
  --input data/deliveries.csv --output data/deliveries_processed.csv

python week3_eda_visualization/eda_visualization.py \
  --input data/deliveries_processed.csv --output-dir reports/eda

python week4_predictive_modeling/predictive_model.py \
  --input data/deliveries_processed.csv
```

The repository contains no customer-identifying information. Before using the workflow with production data, validate units, timezone assumptions, business definitions, and privacy controls.
