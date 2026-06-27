# Applying Explainable Machine Learning to Retail Sales Forecasting

> University Project — Research Skills Course — Technische Universität Ilmenau — 2026  
> Rossmann Store Sales Dataset | Linear Regression · Random Forest · XGBoost · SHAP

---

## Overview

This project implements a machine learning pipeline for retail sales forecasting using the **Rossmann Store Sales** dataset from Kaggle. Three models of increasing complexity are trained and compared. **SHAP (SHapley Additive exPlanations)** is applied to both ensemble models to make predictions interpretable and actionable.

```
Raw Data → Preprocessing → Feature Engineering → ML Models → Evaluation → SHAP Explainability → Forecast
```

---

## Results Summary

| Model | MAE (€) | RMSE (€) | R² | MAPE (%) |
|---|---|---|---|---|
| Linear Regression | 2,012.28 | 2,772.64 | 0.2033 | 33.66 |
| Random Forest | 1,034.71 | 1,483.27 | 0.7720 | 16.54 |
| **XGBoost** | **854.15** | **1,198.13** | **0.8512** | **13.78** |

**Top SHAP features:** CompetitionDistance · Promo · Store · DayOfWeek

**2026 Forecast:** December predicted as peak month — 45.5% uplift over January.

---

## Project Structure

```
RetailSalesForecastingProject/
├── data/
│   ├── train.csv               ← 1,017,209 rows (download from Kaggle)
│   ├── test.csv                ← 41,088 rows (download from Kaggle)
│   ├── store.csv               ← 1,115 store metadata rows
│   └── sample_submission.csv
├── notebooks/
│   ├── Retail_Sales_Forecasting.ipynb         ← main notebook
│   └── Retail_Sales_Forecasting_output.ipynb  ← executed output
├── figures/                    ← 16 auto-generated plots
│   ├── sales_distribution.png
│   ├── sales_trend.png
│   ├── missing_values.png
│   ├── heatmap.png
│   ├── pairplot.png
│   ├── sales_by_features.png
│   ├── model_comparison.png
│   ├── prediction_vs_actual.png
│   ├── residual_plots.png
│   ├── feature_importance.png
│   ├── shap_summary_rf.png
│   ├── shap_summary_xgb.png
│   ├── shap_importance_rf.png
│   ├── shap_waterfall.png
│   ├── shap_dependence.png
│   └── shap_force.png
├── models/
│   ├── linear_regression.pkl   ← 1.2 KB
│   ├── random_forest.pkl       ← 108 MB
│   └── xgboost.pkl             ← ~1 MB
├── paper/
│   └── University_Report.md   ← full academic report
├── requirements.txt
└── README.md
```

---

## How to Run

**Requirements:** Anaconda with the `retail_forecast` conda environment.

```bash
# Step 1 — Activate the environment
conda activate retail_forecast

# Step 2 — Go to the project folder
cd /Users/aman/Desktop/RS/RetailSalesForecastingProject

# Step 3 — Launch Jupyter
jupyter notebook notebooks/Retail_Sales_Forecasting.ipynb
```

In the browser: **Cell → Run All**

Runtime: approximately **3–5 minutes**.

> The notebook loads the pre-trained Linear Regression and Random Forest models from `models/` to avoid out-of-memory issues, and retrains XGBoost fresh each run. All figures are saved automatically to `figures/`.

---

## Dataset

**Rossmann Store Sales** — Kaggle Competition  
1,017,209 daily sales records across 1,115 German drugstore locations (2013–2015).

| File | Rows | Description |
|---|---|---|
| `train.csv` | 1,017,209 | Historical daily sales with features |
| `test.csv` | 41,088 | Stores for prediction (no sales column) |
| `store.csv` | 1,115 | Store metadata (type, competition, promotions) |

**Download:** [Kaggle — Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)

Place all three CSV files in the `data/` folder before running the notebook.

---

## Models

| Model | Role | Key Hyperparameters |
|---|---|---|
| Linear Regression | Interpretable baseline | fit_intercept=True |
| Random Forest | Non-linear baseline | n_estimators=100, max_depth=15 |
| XGBoost | Best performer | n_estimators=500, lr=0.05, max_depth=6, L1=0.1, L2=1.0 |

---

## SHAP Explainability

SHAP explains individual predictions by attributing a contribution score to each feature.

| Artefact | Description |
|---|---|
| Beeswarm plot | Global feature importance with directionality |
| Bar importance plot | Mean \|SHAP\| per feature — ranked list |
| Waterfall plot | Single prediction breakdown from baseline |
| Dependence plot | Feature value vs SHAP value (non-linear effects) |
| Force plot | Additive push/pull visualisation |

---

## Key Findings

- XGBoost achieves R² = 0.8512, explaining 85% of the variance in daily store sales.
- **CompetitionDistance** is the most influential feature — stores within 500 m of a competitor lose ~€800/day on average.
- **Promo** is the most controllable lever — an active promotion adds ~€667/day net uplift on average.
- **December** is the predicted peak month for 2026, with 45.5% higher daily sales than January.
- SHAP waterfall plots make individual store forecasts auditable and interpretable for business managers.

---

## Paper

See [paper/University_Report.md](paper/University_Report.md) for the full academic report including methodology, results, SHAP analysis, limitations, and an IEEE-style self-review.

---

## Technologies

`Python 3.10` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `SHAP` · `Matplotlib` · `Seaborn` · `Jupyter`

---

*Research Skills Course — Master's Programme — Technische Universität Ilmenau — 2026*
