# Apple Dry Matter Prediction using Machine Learning

Predicting apple dry matter content from light absorption spectra using supervised machine learning.

## Project Overview

This project uses **141 wavelength absorption features** (430nm–990nm) measured from 240 apple samples to predict dry matter content using various ML algorithms.

## Dataset

- **Samples:** 240 apples (A1–A240)
- **Features:** 141 absorption values at 4nm intervals (430–990nm)
- **Target:** Dry Matter (0.135–0.174 range)

## Models Used

| Model | R² | Accuracy | F1 |
|-------|-----|----------|-----|
| PLSR (Tuned) | 0.2847 | 77.1% | 0.800 |
| KNN (Tuned) | - | 85.4% | 0.877 |
| Random Forest | 0.27 | 68.8% | 0.694 |
| Gradient Boosting | 0.21 | 72.9% | 0.745 |
| LightGBM | 0.25 | 68.8% | 0.706 |
| SVM | - | 66.7% | 0.667 |
| Extra Trees | - | 70.8% | 0.720 |

## Preprocessing Methods

- **SNV** (Standard Normal Variate)
- **MSC** (Multiplicative Scatter Correction) — Best performing
- **Savitzky-Golay Smoothing**
- **1st/2nd Derivatives**

## Project Structure

```
apple1/
├── apple1.csv                    # Raw dataset
├── dry_matter_model.py           # Initial LightGBM model
├── dry_matter_model_v2.py        # Multi-model comparison
├── dry_matter_full.py            # Full EDA + KNN + F1 + Tuning
├── models_with_plsr.py           # Added PLSR model
├── plsr_hypertune.py             # PLSR hyperparameter tuning
├── optimize_model.py             # Advanced preprocessing
├── save_plsr_model.py            # Save final model
├── optimized_pls_model.pkl       # Best PLSR model
├── optimized_pls_scaler.pkl      # Scaler for predictions
├── *.png                         # Result visualizations
└── README.md                     # This file
```

## Installation

```bash
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn scipy joblib
```

## Usage

### Training the Model

```bash
python save_plsr_model.py
```

### Predicting Dry Matter

```python
import numpy as np
import joblib

# Load model and scaler
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

## Key Findings

1. **MSC preprocessing** gives best results for PLSR
2. **PLSR with n=20 components** is optimal
3. **NIR region (870–970nm)** has highest feature importance
4. Dataset is small (240 samples) — limits model performance

## Results

Best model achieves:
- **R² = 0.2847** (explains 28% of variance)
- **RMSE = 0.00633** (average error ~0.006)
- **Classification Accuracy = 77.1%** (High/Low dry matter)

## License

MIT License
