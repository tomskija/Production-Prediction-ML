# Production-Prediction-ML

**By: Jackson R. Tomski**

This project builds a full-stack production ML system for predicting unconventional reservoir oil output — from raw well data all the way to a running web dashboard. It's designed the way real ML applications are built: modular pipeline, REST API, SQL persistence, experiment tracking, and a React frontend, all containerized and running locally with Docker Compose.

The central research question driving the ML work: **does how you spatially split your training data affect model performance and generalization?** The pipeline answers this by automatically finding the best geographic clustering strategy and letting both the Random Forest and Bayesian Neural Network independently find their optimal train/test splits — then comparing results side by side.

---

## What's Inside

### Machine Learning Pipeline

The pipeline runs through six stages after loading the data:

1. **Feature Engineering** — four sequential scalers (StandardScaler → RobustScaler → PowerTransformer → MinMaxScaler) normalize features and the production target independently.

2. **Feature Selection** — either user-supplied or automatic. In auto mode, a rank ensemble of Pearson correlation, Spearman correlation, mutual information, and RF feature importance selects the best features. If the average mutual information is too low, it falls back to PCA and names the components from their top loadings.

3. **Spatial Clustering** — a parallelized silhouette score sweep across GMM (four covariance types) and K-Means finds the best way to group wells geographically. These clusters drive stratified train/test splits so the model trains and tests on spatially representative subsets.

4. **Hyperparameter Tuning** — parallelized sweep over 10,000+ RF seed combinations, max depth (1–45), and number of trees (1–200).

5. **SHAP Explainability** — TreeExplainer on the RF and GradientExplainer on the BNN both produce feature attribution plots so you can see what's driving predictions.

6. **RF vs BNN Comparison** — once both models finish (running concurrently via `asyncio.gather`), a side-by-side chart compares their test metrics. RF gives point estimates; BNN gives posterior mean ± 1σ across all post burn-in MCMC samples.

### Bayesian Neural Network

The BNN runs Langevin dynamics MCMC to sample the posterior over network weights — giving calibrated uncertainty estimates (P10/mean/P90) rather than a single prediction. Two implementations are available: PyTorch (`torch.distributions`) and TensorFlow (`tensorflow-probability`). Both find their own optimal spatial split independently of the RF, which is the scientifically correct approach since the two models have different inductive biases.

A key design choice: the BNN split seed sweep can run full MCMC (default, accurate) or a fast forward pass approximation (useful during development), gated by the `run_bnn_split_fast` flag.

### Full-Stack Architecture

```
React Dashboard (port 3000)
    ↓
FastAPI REST API (port 8000) — async job submission, status polling, results retrieval
    ↓
ML Pipeline (Calculator.py) — RF + BNN via asyncio.gather
    ↓
PostgreSQL / SQLite — run results persisted per job
    ↓
MLflow (port 5000) — experiment tracking, nested runs, artifact storage
```

The API follows an async job pattern — submit a run, get a `job_id` back immediately, poll for status, retrieve results once complete. This is the same pattern used in production ML serving systems.

---

## Results

GMM spherical clustering consistently produces the lowest test MAPE, confirming that spatially-aware splits reduce sampling bias compared to random splits.

| Sampling Method | Test R² (%) | Test RMSE (%) | Test MAPE (%) |
|-----------------|-------------|---------------|---------------|
| Random          | 99.257      | 0.970         | 8.851         |
| K-means         | 99.233      | 1.014         | 8.101         |
| GMM full        | 99.118      | 1.057         | 7.888         |
| GMM spherical   | 99.360      | 0.829         | 6.124         |
| GMM diagonal    | 99.342      | 0.918         | 6.684         |
| GMM tied        | 99.325      | 0.903         | 7.105         |

---

## Getting Started

**You need:** Docker Desktop and VS Code with the Dev Containers extension.

```bash
git clone https://github.com/tomskija/Production-Prediction-ML.git
cd Production-Prediction-ML
```

Open the folder in VS Code and select **Reopen in Container**. Docker Compose spins up four services automatically: the calculator, FastAPI server, MLflow tracking server, and PostgreSQL. `npm install` runs automatically so the frontend is ready to go.

**Run the pipeline from the terminal:**
```bash
python productionPredictionCalculator/Calculator.py
```

**Start the API server:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Start the React dashboard:**
```bash
cd frontend && npm run dev
```

**Run the test suite:**
```bash
pytest tests -v
```

| Service       | URL                        |
|---------------|----------------------------|
| React UI      | http://localhost:3000      |
| FastAPI       | http://localhost:8000      |
| API Docs      | http://localhost:8000/docs |
| MLflow UI     | http://localhost:5000      |

---

## API Endpoints

| Method   | Endpoint                | Description                                        |
|----------|-------------------------|----------------------------------------------------|
| `GET`    | `/health`               | Check if the API is running                        |
| `POST`   | `/calculate`            | Submit a pipeline run — returns `job_id` immediately |
| `GET`    | `/runs/{run_id}/status` | Poll run status: pending / running / complete / failed |
| `GET`    | `/results/{run_id}`     | Fetch results from SQL by MLflow run ID            |
| `GET`    | `/runs`                 | List all past runs                                 |
| `DELETE` | `/runs/{run_id}`        | Delete a run from SQL and the job store            |

The full interactive API docs are at `http://localhost:8000/docs` once the server is running.

---

## Configuration

All parameters are passed through the React dashboard or directly in the `POST /calculate` request body. The most important ones:

| Parameter | What it does |
|-----------|--------------|
| `predictive_features` | Which well log features to use as inputs |
| `auto_select_features` | `0` = use your list, `1` = let the pipeline pick |
| `run_test` | `1` = fast short sweep for dev, `0` = full production sweep |
| `run_bnn` | `1` = run BNN alongside RF, `0` = RF only |
| `run_bnn_split_fast` | `0` = full MCMC for BNN split selection (default), `1` = fast approximation |
| `bnn_library` | `"pytorch"` or `"tensorflow"` |
| `bnn_n_samples` | Number of MCMC samples |
| `bnn_burn_in` | Fraction of samples to discard as burn-in (default 0.85) |
| `bnn_hidden_neurons` | Hidden layer size for the BNN |

For manual backend testing, add JSON configs to `productionPredictionCalculator/tests/` (e.g. `testCase01.json`). The `tests/in_jsons/` folder is reserved for pytest CI fixtures only.

**Azure ML** — to route MLflow tracking to an Azure ML workspace instead of the local server, set these environment variables (leave blank for local dev):

```
AZURE_SUBSCRIPTION_ID
AZURE_RESOURCE_GROUP
AZURE_ML_WORKSPACE
AZURE_TENANT_ID
AZURE_CLIENT_ID
AZURE_CLIENT_SECRET
```

---

## Output Figures

Everything saves to `Figures and Results/` and is logged to MLflow as artifacts:

**Feature analysis:** `Original_Histograms.png`, `Engineered_Histograms.png`, `Correlation_Heatmap.png`, `Rank_Correlation_Heatmap.png`, `Mutual_Info_and_Feature_Import.png`, `Feature_Selection_Summary.png`

**Clustering & splits:** `Well_Data_ScatterPlot.png`, `Sp_Clustering.png`, `RF_Histograms.png`, `RF_Production_ScatterPlot.png`

**RF model:** `RF_Final_Fit.png`, `RF_Hyperparameter_Tuning.png`, `SHAP_Summary_*.png`, `SHAP_Bar_*.png`, `SHAP_Waterfall_*.png`

**BNN model** *(when `run_bnn=1`):* `BNN_Convergence_*.png`, `BNN_Uncertainty_*.png`, `BNN_PostBurnin_Distributions_*.png` *(R², RMSE, MAPE histograms across posterior samples)*, BNN SHAP plots

**Comparison** *(when `run_bnn=1`):* `RF_vs_BNN_Comparison_*.png` — RF point estimates vs BNN posterior mean ± 1σ

---

## Project Structure

```
Production-Prediction-ML/
├── .devcontainer/devcontainer.json     # Dev container config (ports 3000, 5000, 8000)
├── .github/workflows/
│   ├── ci.yml                          # GitHub Actions CI — builds Docker stack, runs pytest
│   └── cd.yml                          # CD placeholder — wired up in Sprint 4
├── frontend/                           # React dashboard (Vite + React 18)
│   └── src/App.jsx                     # Pipeline, Jobs, Results, History, Health tabs
├── productionPredictionCalculator/
│   ├── Calculator.py                   # Orchestrator — asyncio.gather for RF + BNN
│   ├── Data/                           # Synthetic well dataset + README
│   ├── tests/                          # Dev test cases (testCase01.json etc.)
│   └── utils/
│       ├── utils.py                    # All pipeline logic — clustering, RF, BNN, plots
│       ├── bnn_pt.py                   # PyTorch BNN + Langevin MCMC
│       ├── bnn_tf.py                   # TensorFlow BNN + Langevin MCMC
│       ├── shapAnalysis.py             # SHAP for RF and BNN
│       ├── mlFlowConfig.py             # MLflow — local / Docker / Azure ML
│       ├── azureConfig.py              # Azure ML URI builder
│       ├── dbConfig.py                 # SQLite / PostgreSQL
│       ├── inputWrapper.py             # JSON input parser
│       └── outputWrapper.py           # Output wrapper
├── tests/
│   ├── in_jsons/                       # pytest input fixtures
│   ├── out_jsons/                      # pytest output fixtures
│   └── test_calculator.py             # pytest suite
├── api.py                              # FastAPI — 6 endpoints, async job pattern
├── conftest.py                         # pytest path config
├── Dockerfile                          # Python 3.11 + Node 20
├── Dockerfile.mlflow                   # MLflow server
├── docker-compose.yml                  # calculator, api, mlflow, postgres
└── requirements.txt
```

---

## Citation

This project is grounded in graduate research at the University of Texas at Austin.

**Master's Thesis**
```
Tomski, J.R. (2020). Unconventional reservoir parameter estimation by seismic inversion
and machine learning of the Bakken Formation. University of Texas at Austin.
```

**Book Chapter**
```
Tomski, J.R. et al. (2024). Enhanced artificial intelligence workflow for predicting production
within the Bakken formation. Developments in Structural Geology and Tectonics, Vol. 6, pp. 83–139.
Elsevier. https://doi.org/10.1016/B978-0-32-399593-1.00016-1
```

---

## Dataset

Synthetic unconventional reservoir dataset from [Michael J. Pyrcz (GeostatsGuy)](https://github.com/GeostatsGuy/GeoDataSets) — 1,000 wells with petrophysical and spatial features. See `Data/README.md` for details.

---

## Dependencies

Python 3.11 · numpy · pandas · scikit-learn · scipy · matplotlib · seaborn · shap · mlflow · azureml-mlflow · torch · tensorflow · tensorflow-probability · psycopg2 · joblib · openpyxl · pytest · fastapi · uvicorn · pydantic

Node 20 · React 18 · Vite