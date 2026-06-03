# AI Production Prediction Workflow — Data

A machine learning pipeline for predicting unconventional reservoir production using a Random Forest regressor and optional Bayesian Neural Network, with automated spatial clustering for stratified train/test sampling.

## What it does

1. Loads well data from SQLite (local dev) or PostgreSQL (production Docker service)
2. Engineers and normalizes features (StandardScaler → RobustScaler → PowerTransformer → MinMaxScaler)
3. Runs correlation, mutual information, and feature importance analysis — with optional data-driven feature selection
4. Automatically selects the best spatial clustering method (GMM or K-means) via silhouette score sweep
5. Uses clusters to create stratified train/test splits
6. Sweeps RF random seeds, tree depth, and number of trees to find optimal hyperparameters
7. Optionally runs a PyTorch BNN with Langevin MCMC in parallel on the same split
8. Writes run results back to the database for UI consumption
9. Saves all plots and artifacts to `Figures and Results/` and logs them to MLflow

## Features

- **Predictive features:** Porosity, Brittleness, Latitude, Longitude
- **Target:** Production (cumulative oil)
- **Clustering:** Auto-selects best GMM covariance type or K-means via parallelized silhouette scoring
- **Hyperparameter tuning:** Parallelized sweep over split seeds, RF seeds, max depth, and number of trees

## Data

Place `Unconventional_Synthetic_Dataset.csv` in this `Data/` directory before running locally.

The dataset is a synthetic unconventional reservoir dataset sourced from Michael J. Pyrcz (GeostatsGuy):
- Source: https://github.com/GeostatsGuy/GeoDataSets/blob/master/unconv_MV.csv
- 1,000 wells with features: `Por`, `LogPerm`, `AI`, `Brittle`, `TOC`, `VR`, `Production`, `Latitude`, `Longitude`

The `.gitignore` excludes raw data files from version control. Do not commit data to the repo.

## Database

**Local (SQLite):** A `production.db` file is created automatically in this `Data/` directory on first run when `localTesing=True`. No setup required.

**Production (PostgreSQL):** Spun up automatically as a Docker Compose service. Connection config is set via environment variables in `docker-compose.yml`. No manual setup required.

Two tables are created automatically on first connect:

- `well_data` — raw input well records tagged by `run_id`
- `run_results` — one row per pipeline run containing MLflow run ID, timestamp, selected features, sampling method, model hyperparameters, and performance metrics

## Run

```bash
python productionPredictionCalculator/Calculator.py
```

## Output

All figures and results save to `Figures and Results/` and are logged as MLflow artifacts viewable at `http://localhost:5000`.