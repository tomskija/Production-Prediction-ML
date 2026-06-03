# Production-Prediction-ML

**By: Jackson R. Tomski**

An end-to-end production ML pipeline for predicting unconventional reservoir oil production. The project combines spatially-aware sampling strategies, automated feature selection, Random Forest regression, Bayesian Neural Network (BNN) inference, SHAP explainability, SQL data persistence, and MLflow experiment tracking — all running inside a Docker Compose multi-service environment.

---

## What This Does

Starting from raw petrophysical well data, the pipeline moves through feature engineering, feature selection, spatial clustering, hyperparameter tuning, model explainability, and optional BNN comparison. The core question being answered: does how you split your training data geographically matter for model performance and generalization?

**Feature Engineering** — Sequential normalization pipeline (StandardScaler → RobustScaler → PowerTransformer → MinMaxScaler) applied to raw features to condition the data and reduce outlier effects.

**Feature Selection** — Two modes controlled via `testOrg.json`:
- *User mode* — supply your own feature list directly
- *Auto mode* — data-driven selection using a rank ensemble of Pearson correlation, Spearman rank correlation, mutual information, and RF feature importance. If average MI falls below a configurable threshold, PCA-based dimensionality reduction is used instead, with component names abbreviated from their top loading features for interpretability.

**Spatial Clustering** — Parallelized silhouette score sweep across GMM (full, spherical, diagonal, tied covariance types) and K-means to automatically select the best spatial clustering method and cluster count from latitude/longitude. Clusters drive stratified train/test splits.

**Hyperparameter Tuning** — Parallelized sweep over 10,000+ random seed combinations, max depth (1–45), and number of trees (1–200) to find the optimal RF configuration per sampling strategy.

**SHAP Explainability** — TreeExplainer runs on the final best RF model, producing beeswarm summary plots, bar charts, and waterfall plots for individual prediction breakdowns. For the BNN, GradientExplainer provides comparable feature attribution using integrated gradients — works with both PyTorch and TensorFlow implementations. Feature rankings printed to console after every run.

**Bayesian Neural Network** — Optional BNN with Langevin dynamics MCMC sampling runs in parallel on the same train/test split as the RF, enabling direct model comparison. Produces convergence plots and P10/mean/P90 uncertainty bands. Two implementations available — PyTorch (`bnn_pt.py`) using `torch.distributions.Normal` and TensorFlow (`bnn_tf.py`) using `tensorflow_probability.distributions.Normal`. Toggle via `run_bnn` and `bnn_library` in `testOrg.json`.

**SQL Data Layer** — Well data is read from and results are written back to a database. SQLite is used locally for development; PostgreSQL runs as a dedicated Docker service in production. Run metadata, model performance, and selected features are persisted per run and available for the UI layer to query.

**MLflow Experiment Tracking** — Every run logs to a persistent MLflow tracking server running as a separate Docker service. Parent runs capture pipeline-level params and artifacts; nested child runs log individual RF training metrics (R², RMSE, MAPE, explained variance) for every seed sweep iteration. BNN metrics logged separately when enabled.

---

## Results

GMM spherical consistently produces the lowest MAPE on the test set, suggesting spatially-aware cluster splits reduce sampling bias compared to random splits.

| Sampling Method | Test R² (%) | Test RMSE (%) | Test MAPE (%) |
|-----------------|-------------|---------------|---------------|
| Random          | 99.257      | 0.970         | 8.851         |
| K-means         | 99.233      | 1.014         | 8.101         |
| GMM full        | 99.118      | 1.057         | 7.888         |
| GMM spherical   | 99.360      | 0.829         | 6.124         |
| GMM diagonal    | 99.342      | 0.918         | 6.684         |
| GMM tied        | 99.325      | 0.903         | 7.105         |

---

## Project Structure

```
Production-Prediction-ML/
├── .devcontainer/
│   └── devcontainer.json               # VS Code dev container config
├── productionPredictionCalculator/
│   ├── Calculator.py                   # Pipeline orchestrator
│   ├── Data/
│   │   ├── Unconventional_Synthetic_Dataset.csv
│   │   └── README.md
│   ├── tests/
│   │   └── testOrg.json               # Input config (params + feature list)
│   └── utils/
│       ├── utils.py                    # Core pipeline functions
│       ├── inputWrapper.py             # Generic JSON input parser
│       ├── outputWrapper.py            # Generic output wrapper class
│       ├── mlFlowConfig.py             # MLflow setup + autolog helpers
│       ├── dbConfig.py                 # SQLite / PostgreSQL data layer
│       ├── bnn_pt.py                   # PyTorch BNN + MCMC sampler (torch.distributions)
│       ├── bnn_tf.py                   # TensorFlow BNN + MCMC sampler (tensorflow_probability)
│       └── shapAnalysis.py             # SHAP TreeExplainer (RF) + GradientExplainer (BNN)
├── tests/
│   └── test_calculator.py
├── Dockerfile                          # Calculator service image
├── Dockerfile.mlflow                   # MLflow tracking server image
├── docker-compose.yml                  # Multi-service orchestration
├── requirements.txt
└── README.md
```

---

## Getting Started

**Prerequisites:** Docker Desktop, VS Code with the Dev Containers extension.

```bash
git clone https://github.com/tomskija/Production-Prediction-ML.git
cd Production-Prediction-ML
```

Open in VS Code → `Reopen in Container`. The calculator container, MLflow tracking server, and PostgreSQL database all spin up automatically via Docker Compose.

Once inside the container, run the pipeline:

```bash
python productionPredictionCalculator/Calculator.py
```

To view experiment runs, open `http://localhost:5000` in your browser while the containers are running.

---

## Configuration

All parameters are set in `productionPredictionCalculator/tests/testOrg.json`. Key options:

| Parameter | Description |
|-----------|-------------|
| `predictive_features` | Feature list used when `auto_select_features` is off |
| `auto_select_features` | `0` = user-supplied features, `1` = data-driven selection |
| `mi_threshold` | Average MI gate — below this triggers PCA path |
| `variance_threshold` | Cumulative explained variance cutoff for feature/component selection |
| `run_test` | `1` = short sweep (fast), `0` = full sweep |
| `run_bnn` | `1` = run BNN alongside RF, `0` = RF only |
| `bnn_library` | `"pytorch"` or `"tensorflow"` — selects BNN implementation |
| `bnn_n_samples` | Number of MCMC samples for BNN |
| `bnn_burn_in` | Burn-in fraction for BNN posterior (default 0.85) |
| `bnn_hidden_neurons` | Number of neurons in the BNN hidden layer |

---

## Output

All figures save to `Figures and Results/` at runtime and are logged as MLflow artifacts:

- `Original_Histograms.png` / `Engineered_Histograms.png`
- `Correlation_Heatmap.png` / `Rank_Correlation_Heatmap.png`
- `Mutual_Info_and_Feature_Import.png` / `Feature_Selection_Summary.png`
- `Well_Data_ScatterPlot.png` / `Sp_Clustering.png`
- `RF_Final_Fit.png` / `RF_Hyperparameter_Tuning.png`
- `RF_Histograms.png` / `RF_Production_ScatterPlot.png`
- `SHAP_Summary_*.png` / `SHAP_Bar_*.png` / `SHAP_Waterfall_*.png`
- `BNN_Convergence_*.png` / `BNN_Uncertainty_*.png` *(PyTorch BNN, when `run_bnn=1`)*
- `BNN_TF_Convergence_*.png` / `BNN_TF_Uncertainty_*.png` *(TensorFlow BNN, when `run_bnn=1`)*
- `BNN_PT_SHAP_Summary_*.png` / `BNN_PT_SHAP_Bar_*.png` *(PyTorch BNN SHAP)*
- `BNN_TF_SHAP_Summary_*.png` / `BNN_TF_SHAP_Bar_*.png` *(TensorFlow BNN SHAP)*

Run results are also persisted to the SQL database for downstream UI consumption.

---

## Citation

This workflow originated from graduate research at the University of Texas at Austin.

**Master's Thesis**
```
Tomski, J.R. (2020). Unconventional reservoir parameter estimation by seismic inversion
and machine learning of the Bakken Formation, North Dakota.
Master's Thesis, University of Texas at Austin.
```

**Book Chapter**
```
Tomski, J.R. et al. (2024). Enhanced artificial intelligence workflow for predicting production
within the Bakken formation. In Developments in Structural Geology and Tectonics (Vol. 6, pp. 83–139).
Elsevier. https://doi.org/10.1016/B978-0-32-399593-1.00016-1
```

---

## Dataset

Synthetic unconventional reservoir dataset from [Michael J. Pyrcz (GeostatsGuy)](https://github.com/GeostatsGuy/GeoDataSets) — 1,000 wells with petrophysical and spatial features. See `Data/README.md` for details.

---

## Dependencies

Python 3.11 · scikit-learn · scipy · numpy · pandas · matplotlib · seaborn · shap · mlflow · torch · tensorflow · tensorflow-probability · psycopg2 · joblib · openpyxl