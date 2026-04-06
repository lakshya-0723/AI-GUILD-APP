# HaulMark Challenge - Fuel Consumption Prediction

## Project Overview
This repository contains a data-driven solution for predicting daily fuel consumption of mining dumpers. It leverages high-frequency telemetry data, geospatial mine polygons, and domain-specific feature engineering to provide accurate predictions and actionable efficiency benchmarks.

## Codebase Structure
- **`guild_app.py`**: The main execution script. It handles data loading, feature extraction, model training (LightGBM), and submission generation.
- **`spatial_features.py`**: A specialized geospatial library used by `guild_app.py` to extract haul cycles and hauling distances from raw GPS paths. **(Required dependency)**.
- **`secondary_outputs.py`**: A diagnostic tool that calculates fleet-wide route benchmarks, individual vehicle efficiency gaps, and daily consumption consistency.
- **`report.md`**: A comprehensive analysis report documenting the methodology, feature engineering choices, and results.


## Requirements
- Python 3.10+
- `pandas`
- `numpy`
- `geopandas`
- `shapely`
- `lightgbm`
- `joblib`
- `pyarrow` (for parquet files)

## How to Run
1. **Setup**: Ensure all data files (`telemetry_*.parquet`, `smry_*.csv`, `fleet.csv`, and `.gpkg` mine files) are in the same directory as the scripts.
2. **Train & Predict**: Run `python3 guild_app.py` to process telemetry and generate `submission.csv`.
3. **Secondary Analysis**: Run `python3 secondary_outputs.py` to generate the benchmarks and efficiency metrics (outputted as CSV files).
