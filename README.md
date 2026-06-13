# Apple Dry Matter Prediction using Machine Learning

Predicting apple dry matter content from light absorption spectra using supervised machine learning.

---

## Project Overview

This project uses **141 wavelength absorption features** (430nm-990nm) measured from 240 apple samples to predict dry matter content using various ML algorithms.

| Detail | Value |
|--------|-------|
| **Samples** | 240 apples (A1-A240) |
| **Features** | 141 absorption values at 4nm intervals |
| **Wavelength Range** | 430nm - 990nm |
| **Target** | Dry Matter (0.135 - 0.174) |

---

## Project Structure

```
apple1/
|
|-- DATA
|   |-- apple1.csv                    <- Raw dataset (240 samples x 141 wavelengths)
|
|-- PYTHON SCRIPTS (Run in order)
|   |
|   |-- Step 1: dry_matter_model.py       <- LightGBM baseline + hyperparameter tuning
|   |-- Step 2: dry_matter_model_v2.py    <- Multi-model comparison (6 models)
|   |-- Step 3: dry_matter_full.py        <- Full EDA + KNN + F1 + Hyperparameter Tuning
|   |-- Step 4: models_with_plsr.py       <- Added PLSR model (7 models total)
|   |-- Step 5: plsr_hypertune.py         <- PLSR hyperparameter tuning
|   |-- Step 6: optimize_model.py         <- Advanced preprocessing (SG, MSC, Ratios)
|   |-- Step 7: save_plsr_model.py        <- Save final trained model
|   |-- Experiment: dry_matter_95.py      <- 95% accuracy attempt (documented failure)
|
|-- SAVED MODELS (.pkl files)
|   |
|   |-- optimized_pls_model.pkl           <- Best PLSR model (MSC + n=20)
|   |-- optimized_pls_scaler.pkl          <- Scaler for optimized model
|   |-- pls_model.pkl                     <- PLSR model (Derivative + n=19)
|   |-- pls_scaler.pkl                    <- Standard scaler
|   |-- wavelength_columns.pkl            <- Wavelength names
|
|-- RESULT IMAGES (.png files)
|   |
|   |-- dry_matter_results_tuned.png      <- LightGBM tuned results (baseline vs tuned)
|   |-- dry_matter_results.png            <- LightGBM baseline results
|   |-- dry_matter_full_analysis.png      <- EDA + Model comparison plots
|   |-- models_with_plsr.png             <- 7 models comparison
|   |-- plsr_hypertuned.png              <- PLSR tuning analysis
|   |-- optimized_results.png            <- Before vs After optimization
|   |-- dry_matter_95_accuracy.png        <- 95% experiment results
|
|-- DOCUMENTATION
|   |-- README.md                         <- This file
|   |-- LICENSE                           <- MIT License
|
|-- CONFIGURATION
    |-- .gitignore                        <- Git ignore rules
```

---

## ML Pipeline (What We Did)

```
STEP 1: DATA EXPLORATION (EDA)
  - Loaded dataset: 240 samples x 141 features
  - Checked missing values: None
  - Analyzed DryMatter distribution
  - Found: NIR region (870-970nm) most important
                    |
                    v
STEP 2: PREPROCESSING
  - SNV (Standard Normal Variate)
  - MSC (Multiplicative Scatter Correction) <- BEST
  - Savitzky-Golay Smoothing
  - 1st/2nd Derivatives
                    |
                    v
STEP 3: FEATURE ENGINEERING
  - Spectral Ratios (band ratios)
  - NDVI-like Indices
  - Water Indices
  - Mutual Information Selection
                    |
                    v
STEP 4: MODEL TRAINING
  - LightGBM (Gradient Boosting) <- BEST for Regression
  - PLSR (Partial Least Squares Regression)
  - KNN (K-Nearest Neighbors) <- BEST for Classification
  - Random Forest
  - Gradient Boosting
  - SVM (Support Vector Machine)
  - Extra Trees
                    |
                    v
STEP 5: HYPERPARAMETER TUNING
  - GridSearchCV / RandomizedSearchCV
  - Cross-validation (5-fold)
  - Best parameters for each model
                    |
                    v
STEP 6: EVALUATION
  - Regression: R2, RMSE, MAE
  - Classification: Accuracy, F1, Confusion Matrix
  - Before vs After comparison
                    |
                    v
STEP 7: SAVE & DEPLOY
  - Save best model as .pkl file
  - Save scaler for predictions
  - Ready for real apple testing
```

---

## Model Results

### Regression Models (Predict DryMatter value)

| Model | R2 | RMSE | MAE |
|-------|-----|------|-----|
| **LightGBM (Tuned)** | **0.3567** | **0.006005** | **0.004939** |
| LightGBM (Baseline) | 0.3031 | 0.006250 | 0.005024 |
| PLSR (MSC, n=20) | 0.2847 | 0.00633 | 0.00518 |
| PLSR (Derivative, n=19) | 0.2708 | 0.00639 | 0.00518 |
| Random Forest | 0.27 | 0.00632 | 0.00504 |
| Gradient Boosting | 0.21 | 0.00664 | 0.00509 |

### Classification Models (Predict High/Low DryMatter)

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **KNN (Tuned)** | **85.4%** | **0.877** |
| Soft Voting (Ensemble) | 79.2% | 0.800 |
| PLSR | 77.1% | 0.800 |
| Gradient Boosting | 72.9% | 0.745 |
| Random Forest | 68.8% | 0.694 |
| LightGBM | 68.8% | 0.706 |
| SVM | 66.7% | 0.667 |

---

## LightGBM Hyperparameter Tuning

### Baseline vs Tuned

| Metric | Baseline | Tuned | Improvement |
|--------|----------|-------|-------------|
| **R2** | 0.3031 | **0.3567** | **+17.7%** |
| **RMSE** | 0.006250 | **0.006005** | **+3.9%** |
| **MAE** | 0.005024 | **0.004939** | **+1.7%** |

### Best Hyperparameters Found

```
num_leaves:       16
learning_rate:    0.1
feature_fraction: 1.0
bagging_fraction: 0.7
bagging_freq:     5
min_data_in_leaf: 10
n_estimators:     300
```

### Top 10 Important Wavelengths (After Tuning)

| Wavelength | Importance |
|------------|------------|
| 650 nm | 21 |
| 654 nm | 18 |
| 662 nm | 17 |
| 702 nm | 15 |
| 978 nm | 14 |
| 722 nm | 12 |
| 910 nm | 12 |
| 618 nm | 12 |
| 970 nm | 11 |
| 906 nm | 11 |

---

## Installation

```bash
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn scipy joblib
```

---

## Usage

### Option 1: Run Complete Pipeline

```bash
cd apple1
python dry_matter_model.py
python dry_matter_model_v2.py
python dry_matter_full.py
python models_with_plsr.py
python plsr_hypertune.py
python optimize_model.py
python save_plsr_model.py
```

### Option 2: Predict with Saved Model

```python
import numpy as np
import joblib

# Load saved model and scaler
pls = joblib.load('optimized_pls_model.pkl')
scaler = joblib.load('optimized_pls_scaler.pkl')

# Input: 141 absorption values (430nm to 990nm, 4nm step)
my_apple = [0.12, 0.15, 0.18, ...]  # 141 values

# Preprocess
X = np.array(my_apple).reshape(1, -1)
X_scaled = scaler.transform(X)

# Predict
dry_matter = pls.predict(X_scaled)
print(f'Dry Matter: {dry_matter[0][0]:.4f}')
```

---

## Key Concepts Explained

| Concept | What It Means |
|---------|---------------|
| **LightGBM** | Fast gradient boosting framework |
| **PLSR** | Reduces features to latent components, then predicts |
| **KNN** | Finds K similar samples, averages their values |
| **Random Forest** | Averages many decision trees |
| **SNV** | Normalizes each spectrum individually |
| **MSC** | Removes scatter effects from light |
| **Savitzky-Golay** | Smooths noise while preserving peaks |
| **R2** | How much variance model explains (0-1, higher=better) |
| **RMSE** | Average prediction error (lower=better) |
| **F1 Score** | Balance of precision and recall (0-1, higher=better) |
| **Baseline** | First model with default parameters |
| **Hyperparameter Tuning** | Finding optimal model settings |

---

## Key Findings

1. **LightGBM (Tuned)** gives best regression results (R2=0.3567)
2. **KNN** gives best classification results (85.4% accuracy)
3. **MSC preprocessing** gives best results for PLSR
4. **NIR region (870-970nm)** has highest feature importance
5. **Smaller num_leaves (16)** works better than default (31)
6. Dataset is small (240 samples) - limits model performance

---

## Results Summary

| Task | Best Model | Best Score |
|------|-----------|------------|
| **Regression** | LightGBM (Tuned) | R2=0.3567 |
| **Classification** | KNN (Tuned) | Accuracy=85.4% |
| **Preprocessing** | MSC | R2=0.2847 |

| Metric | Before Tuning | After Tuning | Improvement |
|--------|---------------|--------------|-------------|
| Regression R2 | 0.3031 | 0.3567 | +17.7% |
| RMSE | 0.006250 | 0.006005 | +3.9% |
| Classification Accuracy | 77.1% | 85.4% (KNN) | +8.3% |

---

## Files to Run

| Order | File | What It Does |
|-------|------|--------------|
| 1 | `dry_matter_model.py` | LightGBM baseline + tuning (R2=0.3567) |
| 2 | `dry_matter_model_v2.py` | Compare 6 models |
| 3 | `dry_matter_full.py` | Full EDA + KNN + Tuning |
| 4 | `models_with_plsr.py` | Add PLSR (7 models) |
| 5 | `plsr_hypertune.py` | Tune PLSR parameters |
| 6 | `optimize_model.py` | Advanced preprocessing |
| 7 | `save_plsr_model.py` | Save final model |
| - | `dry_matter_95.py` | Experiment: 95% accuracy attempt |

---

## Experiment: 95% Accuracy Attempt

### File: `dry_matter_95.py`

This file represents an experiment to achieve 95% accuracy using advanced techniques.

| Technique | Purpose |
|-----------|---------|
| Multiple Thresholds | Test different High/Low split points |
| Spectral Ratios | Create band ratios between wavelengths |
| PCA | Reduce to 20 principal components |
| 6 Models | RF, ET, LGB, GBM, SVM, KNN |
| Multi-Split Ensemble | 50 random splits for variance reduction |
| Voting Classifier | Combine all models for better predictions |

**Result:** Best accuracy = 79.2% (did NOT reach 95%)

### Why 95% Was Not Achieved

| Reason | Explanation |
|--------|-------------|
| **Small dataset** | Only 240 samples for 141 features |
| **Narrow target range** | DryMatter only varies by 0.04 |
| **Overlapping classes** | High/Low classes are not well-separated |
| **Low correlation** | Max wavelength-DM correlation is only 0.195 |

---

## License

MIT License - Feel free to use and modify for your projects.

---

## Contributors

- **Nandini Eslavath** - Initial work and ML pipeline
