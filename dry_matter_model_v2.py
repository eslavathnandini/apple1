import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('apple1.csv')
print(f"Dataset: {df.shape[0]} samples, {df.shape[1]-2} wavelength features\n")

X = df.drop(columns=['Apple', 'DryMatter']).values
y = df['DryMatter'].values
wavelength_names = [int(c) for c in df.columns if c not in ['Apple', 'DryMatter']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
X_all_s = scaler.fit_transform(X)

models = {
    'Ridge': Ridge(alpha=1.0),
    'PLS-10': PLSRegression(n_components=10),
    'PLS-20': PLSRegression(n_components=20),
    'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05, verbose=-1, seed=42),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"{'Model':<20} {'Test R²':>10} {'RMSE':>12} {'MAE':>10} {'CV R²':>15}")
print("=" * 72)

best_r2 = -999
best_name = ""
best_pred = None

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    cv = cross_val_score(model, X_all_s, y, cv=kf, scoring='r2')
    print(f"{name:<20} {r2:>10.4f} {rmse:>12.6f} {mae:>10.6f} {cv.mean():>8.4f} ± {cv.std():.4f}")
    if r2 > best_r2:
        best_r2, best_name, best_pred = r2, name, y_pred

print(f"\n>>> Best: {best_name} (R²={best_r2:.4f})")

mi = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({'nm': wavelength_names, 'MI': mi}).sort_values('MI', ascending=False)

print(f"\nTop 15 wavelengths (mutual information):")
print(mi_df.head(15).to_string(index=False))

print(f"\nPLS with top-K wavelengths:")
print(f"{'K':>5} {'R²':>10} {'CV R²':>15}")
for k in [5, 10, 15, 20, 30, 50]:
    idx = [wavelength_names.index(w) for w in mi_df.head(k)['nm'].values]
    Xk = X_all_s[:, idx]
    Xk_tr, Xk_te, yk_tr, yk_te = train_test_split(Xk, y, test_size=0.2, random_state=42)
    pls = PLSRegression(n_components=min(k, 10))
    pls.fit(Xk_tr, yk_tr)
    yp = pls.predict(Xk_te).ravel()
    r2k = r2_score(yk_te, yp)
    cvk = cross_val_score(pls, Xk, y, cv=kf, scoring='r2')
    print(f"{k:>5} {r2k:>10.4f} {cvk.mean():>8.4f} ± {cvk.std():.4f}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].scatter(y_test, best_pred, alpha=0.7, c='steelblue', edgecolors='k', s=50)
mn, mx = min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())
axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2)
axes[0].set_xlabel('Actual Dry Matter')
axes[0].set_ylabel('Predicted Dry Matter')
axes[0].set_title(f'{best_name}: R²={best_r2:.4f}')

correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
axes[1].plot(wavelength_names, correlations, color='steelblue')
axes[1].axhline(y=0, color='gray', linestyle='--')
axes[1].set_xlabel('Wavelength (nm)')
axes[1].set_ylabel('r with Dry Matter')
axes[1].set_title('Spectral Correlation')

axes[2].bar(range(15), mi_df.head(15)['MI'].values, color='steelblue')
axes[2].set_xticks(range(15))
axes[2].set_xticklabels([f'{int(w)}' for w in mi_df.head(15)['nm'].values], rotation=45, ha='right', fontsize=8)
axes[2].set_ylabel('MI Score')
axes[2].set_title('Top 15 Wavelengths')

plt.tight_layout()
plt.savefig('dry_matter_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: dry_matter_results.png")
