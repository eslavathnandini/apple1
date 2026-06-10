# 🍎 Apple Dry Matter Prediction using Machine Learning

Predicting apple dry matter content from light absorption spectra using supervised machine learning.

---

## 📋 Project Overview

This project uses **141 wavelength absorption features** (430nm–990nm) measured from 240 apple samples to predict dry matter content using various ML algorithms.

| Detail | Value |
|--------|-------|
| **Samples** | 240 apples (A1–A240) |
| **Features** | 141 absorption values at 4nm intervals |
| **Wavelength Range** | 430nm – 990nm |
| **Target** | Dry Matter (0.135 – 0.174) |

---

## 📁 Project Structure

```
apple1/
│
├── 📊 DATA
│   └── apple1.csv                    ← Raw dataset (240 samples × 141 wavelengths)
│
├── 🐍 PYTHON SCRIPTS (Run in order)
│   │
│   ├── Step 1: dry_matter_model.py       ← First LightGBM model (baseline)
│   ├── Step 2: dry_matter_model_v2.py    ← Multi-model comparison (6 models)
│   ├── Step 3: dry_matter_full.py        ← Full EDA + KNN + F1 + Hyperparameter Tuning
│   ├── Step 4: models_with_plsr.py       ← Added PLSR model (7 models total)
│   ├── Step 5: plsr_hypertune.py         ← PLSR hyperparameter tuning
│   ├── Step 6: optimize_model.py         ← Advanced preprocessing (SG, MSC, Ratios)
│   ├── Step 7: save_plsr_model.py        ← Save final trained model
│   └── Experiment: dry_matter_95.py      ← 95% accuracy attempt (documented failure)
│
├── 🤖 SAVED MODELS (.pkl files)
│   │
│   ├── optimized_pls_model.pkl           ← Best PLSR model (MSC + n=20)
│   ├── optimized_pls_scaler.pkl          ← Scaler for optimized model
│   ├── pls_model.pkl                     ← PLSR model (Derivative + n=19)
│   ├── pls_scaler.pkl                    ← Standard scaler
│   └── wavelength_columns.pkl            ← Wavelength names
│
├── 📈 RESULT IMAGES (.png files)
│   │
│   ├── dry_matter_results.png            ← LightGBM baseline results
│   ├── dry_matter_full_analysis.png      ← EDA + Model comparison plots
│   ├── models_with_plsr.png             ← 7 models comparison
│   ├── plsr_hypertuned.png              ← PLSR tuning analysis
│   └── optimized_results.png            ← Before vs After optimization
│
├── 📄 DOCUMENTATION
│   ├── README.md                         ← This file
│   └── LICENSE                           ← MIT License
│
└── ⚙️ CONFIGURATION
    └── .gitignore                        ← Git ignore rules
```

---

## 🔄 ML Pipeline (What We Did)

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: DATA EXPLORATION (EDA)                            │
│  ─────────────────────────────────────────                  │
│  • Loaded dataset: 240 samples × 141 features              │
│  • Checked missing values: None                            │
│  • Analyzed DryMatter distribution                         │
│  • Found: NIR region (870-970nm) most important            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: PREPROCESSING                                     │
│  ─────────────────────────────                              │
│  • SNV (Standard Normal Variate)                           │
│  • MSC (Multiplicative Scatter Correction) ← BEST          │
│  • Savitzky-Golay Smoothing                                │
│  • 1st/2nd Derivatives                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: FEATURE ENGINEERING                               │
│  ─────────────────────────────                              │
│  • Spectral Ratios (band ratios)                           │
│  • NDVI-like Indices                                        │
│  • Water Indices                                           │
│  • Mutual Information Selection                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: MODEL TRAINING                                    │
│  ─────────────────────────                                  │
│  • PLSR (Partial Least Squares Regression) ← BEST          │
│  • KNN (K-Nearest Neighbors)                               │
│  • Random Forest                                           │
│  • Gradient Boosting                                       │
│  • LightGBM                                                │
│  • SVM (Support Vector Machine)                            │
│  • Extra Trees                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: HYPERPARAMETER TUNING                             │
│  ─────────────────────────────                              │
│  • GridSearchCV / RandomizedSearchCV                       │
│  • Cross-validation (5-fold)                               │
│  • Best parameters for each model                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: EVALUATION                                        │
│  ─────────────────────                                      │
│  • Regression: R², RMSE, MAE                               │
│  • Classification: Accuracy, F1, Confusion Matrix          │
│  • Before vs After comparison                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 7: SAVE & DEPLOY                                     │
│  ─────────────────────                                      │
│  • Save best model as .pkl file                            │
│  • Save scaler for predictions                             │
│  • Ready for real apple testing                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧪 Experiment: 95% Accuracy Attempt

### File: `dry_matter_95.py`

This file represents an **experiment to achieve 95% accuracy** using advanced techniques.

### What It Tested

| Technique | Purpose |
|-----------|---------|
| Multiple Thresholds | Test different High/Low split points (25th-50th percentile) |
| Spectral Ratios | Create band ratios between wavelengths |
| Difference Features | Calculate wavelength differences |
| PCA | Reduce to 20 principal components |
| 6 Models | RF, ET, LGB, GBM, SVM, KNN |
| Multi-Split Ensemble | 50 random splits for variance reduction |
| Voting Classifier | Combine all models for better predictions |

### Key Results

```python
Best Accuracy: 79.2% (Soft Voting)
Did NOT reach 95% target
```

### Why 95% Was Not Achieved

| Reason | Explanation |
|--------|-------------|
| **Small dataset** | Only 240 samples for 141 features |
| **Narrow target range** | DryMatter only varies by 0.04 |
| **Overlapping classes** | High/Low classes are not well-separated |
| **Low correlation** | Max wavelength-DM correlation is only 0.195 |

### What We Learned

1. **Spectral ratios didn't help** — raw wavelengths already contain the information
2. **Ensembles help** but are limited by data size
3. **More data needed** — 1000+ samples would significantly improve results
4. **Feature engineering has limits** — can't create information that isn't there

### Why Keep This File?

| Reason | Explanation |
|--------|-------------|
| **Documents negative results** | Shows what didn't work |
| **Demonstrates thoroughness** | We tried everything possible |
| **Learning resource** | Others can learn from this experiment |
| **Scientific integrity** | Honest reporting of all attempts |

---

## 📊 Model Results

### Regression Models (Predict DryMatter value)

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| **PLSR (MSC, n=20)** | **0.2847** | **0.00633** | **0.00518** |
| PLSR (Derivative, n=19) | 0.2708 | 0.00639 | 0.00518 |
| Random Forest | 0.27 | 0.00632 | 0.00504 |
| LightGBM | 0.25 | 0.00649 | 0.00509 |
| Gradient Boosting | 0.21 | 0.00664 | 0.00509 |

### Classification Models (Predict High/Low DryMatter)

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **KNN (Tuned)** | **85.4%** | **0.877** |
| PLSR | 77.1% | 0.800 |
| Gradient Boosting | 72.9% | 0.745 |
| Random Forest | 68.8% | 0.694 |
| LightGBM | 68.8% | 0.706 |
| SVM | 66.7% | 0.667 |

---

## 🛠️ Installation

```bash
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn scipy joblib
```

---

## 🚀 Usage

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

## 🧠 Key Concepts Explained

| Concept | What It Means |
|---------|---------------|
| **PLSR** | Reduces features to latent components, then predicts |
| **KNN** | Finds K similar samples, averages their values |
| **Random Forest** | Averages many decision trees |
| **SNV** | Normalizes each spectrum individually |
| **MSC** | Removes scatter effects from light |
| **Savitzky-Golay** | Smooths noise while preserving peaks |
| **R²** | How much variance model explains (0-1, higher=better) |
| **RMSE** | Average prediction error (lower=better) |
| **F1 Score** | Balance of precision and recall (0-1, higher=better) |

---

## 🔍 Key Findings

1. **MSC preprocessing** gives best results for PLSR
2. **PLSR with n=20 components** is optimal
3. **NIR region (870–970nm)** has highest feature importance
4. Dataset is small (240 samples) — limits model performance
5. **KNN achieves 85.4% accuracy** for classification

---

## 📈 Results Summary

| Metric | Before Optimization | After Optimization | Improvement |
|--------|--------------------|--------------------|-------------|
| Regression R² | 0.2708 | 0.2847 | +5.1% |
| RMSE | 0.00639 | 0.00633 | -0.9% |
| Classification Accuracy | 77.1% | 85.4% (KNN) | +8.3% |

---

## 📝 Files to Run

| Order | File | What It Does |
|-------|------|--------------|
| 1 | `dry_matter_model.py` | First LightGBM attempt |
| 2 | `dry_matter_model_v2.py` | Compare 6 models |
| 3 | `dry_matter_full.py` | Full EDA + KNN + Tuning |
| 4 | `models_with_plsr.py` | Add PLSR (7 models) |
| 5 | `plsr_hypertune.py` | Tune PLSR parameters |
| 6 | `optimize_model.py` | Advanced preprocessing |
| 7 | `save_plsr_model.py` | Save final model |
| - | `dry_matter_95.py` | Experiment: 95% accuracy attempt |

---

## 📜 License

MIT License - Feel free to use and modify for your projects.

---

## 👥 Contributors

- **Nandini Eslavath** - Initial work and ML pipeline
