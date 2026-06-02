# AI Production Prediction Workflow

A machine learning pipeline for predicting unconventional reservoir production using a Random Forest regressor with automated spatial clustering for stratified train/test sampling.

## What it does

1. Loads and cleans a synthetic unconventional reservoir dataset
2. Engineers and normalizes features (StandardScaler → RobustScaler → PowerTransformer → MinMaxScaler)
3. Runs correlation and feature importance analysis
4. Automatically selects the best spatial clustering method (GMM or K-means) via silhouette score sweep
5. Uses clusters to create stratified train/test splits
6. Sweeps RF random seeds, tree depth, and number of trees to find optimal hyperparameters
7. Saves all plots and results to an output directory

## Features

- **Predictive features:** Porosity, Brittleness, Latitude, Longitude
- **Target:** Production (cumulative oil)
- **Clustering:** Auto-selects best GMM covariance type or K-means random state via parallelized silhouette scoring
- **Hyperparameter tuning:** Parallelized sweep over split seeds, RF seeds, max depth, and number of trees

## Data

Place `Unconventional_Synthetic_Dataset.csv` in the `Data/` directory before running.

The dataset is a synthetic unconventional reservoir dataset sourced from Michael J. Pyrcz (GeostatsGuy):
- Source: https://github.com/GeostatsGuy/GeoDataSets/blob/master/unconv_MV.csv
- 1,000 wells with features: `Por`, `LogPerm`, `AI`, `Brittle`, `TOC`, `VR`, `Production`, `Latitude`, `Longitude`

The `.gitignore` excludes raw data files from version control. Do not commit data to the repo.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python productionPredictionCalculator/Calculator.py
```

## Output

All figures and results are saved to `z_Feature_Selection_Production_Results/`:

- `Original_Histograms.png` / `Engineered_Histograms.png`
- `Correlation_Heatmap.png` / `Rank_Correlation_Heatmap.png`
- `Mutual_Info_and_Feature_Import.png`
- `Well_Data_ScatterPlot.png`
- `Sp_Clustering.png`
- `RF_Final_Fit.png`
- `RF_Hyperparameter_Tuning.png`
- `RF_Histograms.png`
- `RF_Production_ScatterPlot.png`
- `RF_Results.csv`

## Configuration

All parameters are set in `checkData()` inside `utils/utils.py`, including normalization bounds, clustering sweep ranges, predictive features, and RF seed ranges.