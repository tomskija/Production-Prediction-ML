# Production-Prediction-ML

**By: Jackson R. Tomski**

An end-to-end ML workflow for predicting oil production in an unconventional reservoir. The core of this project is comparing how different spatially-aware sampling strategies — random sampling, K-means clustering, and Gaussian Mixture Models — affect Random Forest model performance and training set bias.

---

## What This Does

Starting from raw petrophysical well data, the notebook moves through feature engineering, feature selection, sampling strategy comparison, hyperparameter tuning, and validation. The key question being answered: does how you split your training data geographically matter?

**Feature Engineering** — Standardization, robust scaling, power transformation, and min/max normalization applied to raw features (`Por`, `LogPerm`, `AI`, `Brittle`, `TOC`, `VR`) to condition the data and reduce outlier effects.

**Feature Selection** — Pearson correlation, Spearman rank correlation, mutual information, and RF feature importance computed to narrow down to 4 optimal predictors: `Brittle`, `Por`, `Latitude`, `Longitude`.

**Sampling Methods (7 total)** — 70/30 train/test splits generated using:
1. Random sampling
2. K-means clustering (random state 9307)
3. K-means clustering (random state 0)
4. GMM — full covariance
5. GMM — spherical covariance
6. GMM — diagonal covariance
7. GMM — tied covariance

Optimal cluster counts determined via elbow curve (K-means) or BIC score (GMM).

**Hyperparameter Tuning** — Max depth (1–45) and number of trees (1–200) swept per sampling method. Optimal train/test and RF random states found over 10,000 seed iterations.

**Validation** — Feature distribution histograms and spatial production scatter plots confirm geographic coverage and similar value distributions across training and testing sets.

---

## Results

All 7 methods achieve R² > 99% and RMSE < 1.1% after tuning. GMM-based methods tend to produce lower MAPE on the test set, suggesting reduced sampling bias from spatially-aware cluster splits.

| Sampling Method    | Test R² (%) | Test RMSE (%) | Test MAPE (%) |
|--------------------|-------------|---------------|---------------|
| Random             | 99.257      | 0.970         | 8.851         |
| K-means (9307)     | 99.183      | 1.040         | 8.548         |
| K-means (0)        | 99.233      | 1.014         | 8.101         |
| GMM full           | 99.118      | 1.057         | 7.888         |
| GMM spherical      | 99.360      | 0.829         | 6.124         |
| GMM diag           | 99.342      | 0.918         | 6.684         |
| GMM tied           | 99.325      | 0.903         | 7.105         |

---

## Citation

This workflow originated from graduate research at the University of Texas at Austin and was extended into a published book chapter.

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

## Project Structure

```
Production-Prediction-ML/
├── .devcontainer/
│   └── devcontainer.json
├── data/
│   └── README.md               # Dataset placement instructions
├── notebooks/
│   └── Production_Prediction_Workflow_Using_Various_Sampling_Methods.ipynb
├── results/                    # Figures and CSVs generated at runtime
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .gitignore
```

---

## Getting Started

**Option 1 — VS Code Dev Container (recommended)**

```bash
git clone https://github.com/tomskija/Production-Prediction-ML.git
cd Production-Prediction-ML
```

Open in VS Code → `Reopen in Container`. Place `Unconventional_Synthetic_Dataset.xlsx` in `data/` before running the notebook.

**Option 2 — Docker Compose**

```bash
docker-compose up --build
```

Navigate to `http://localhost:8888` and open the notebook from `notebooks/`.

**Option 3 — Local**

```bash
pip install -r requirements.txt
jupyter lab
```

---

## Selecting a Sampling Method

Near the top of the prediction section in the notebook:

```python
sampling_method = sampling_method01  # Random
sampling_method = sampling_method04  # GMM full
sampling_method = sampling_method07  # GMM tied
```

The workflow adapts automatically from there.

---

## Dependencies

Python 3.11 · scikit-learn · scipy · numpy · pandas · matplotlib · seaborn · JupyterLab · openpyxl

---

## Dataset

Synthetic unconventional reservoir dataset from [Michael J. Pyrcz (GeostatsGuy)](https://github.com/GeostatsGuy/GeoDataSets) — 1,000 wells with petrophysical and spatial features. Not included in this repo. See `data/README.md`.
