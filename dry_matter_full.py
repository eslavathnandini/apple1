import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             f1_score, classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('apple1.csv')
print("=" * 70)
print("1. DATA EXPLORATORY ANALYSIS")
print("=" * 70)
print(f"\nDataset: {df.shape[0]} samples x {df.shape[1]-2} wavelength features")
print(f"\n--- DryMatter Statistics ---")
print(df['DryMatter'].describe())
print(f"\n--- Absorption Stats ---")
wavelength_cols = [c for c in df.columns if c not in ['Apple', 'DryMatter']]
wavelengths = [int(c) for c in wavelength_cols]
X = df[wavelength_cols].values
y = df['DryMatter'].values
print(f"Mean: {X.mean():.4f}, Std: {X.std():.4f}, Min: {X.min():.4f}, Max: {X.max():.4f}")
print(f"Wavelength range: {min(wavelengths)}-{max(wavelengths)} nm, Step: {wavelengths[1]-wavelengths[0]} nm")
print(f"Missing values: {df.isnull().sum().sum()}")

correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
corr_df = pd.DataFrame({'nm': wavelengths, 'r': correlations})
corr_df['abs_r'] = corr_df['r'].abs()
print(f"\n--- Top 10 Correlated Wavelengths ---")
print(corr_df.sort_values('abs_r', ascending=False).head(10).to_string(index=False))

# 2. PREPROCESSING
print("\n" + "=" * 70)
print("2. DATA PREPROCESSING")
print("=" * 70)

mi = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({'nm': wavelengths, 'MI': mi}).sort_values('MI', ascending=False)
top20 = mi_df.head(20)['nm'].values
top_idx = [wavelengths.index(w) for w in top20]
X_sel = X[:, top_idx]

median_dm = np.median(y)
y_class = (y >= median_dm).astype(int)
print(f"Selected 20 features by MI. Median DM={median_dm:.4f}")
print(f"Class split: Low={(y_class==0).sum()}, High={(y_class==1).sum()}")

X_train, X_test, y_train, y_test, yc_train, yc_test = train_test_split(
    X_sel, y, y_class, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. MODEL TRAINING WITH KNN TUNING
print("\n" + "=" * 70)
print("3. MODEL TRAINING & KNN HYPERPARAMETER TUNING")
print("=" * 70)

# KNN tuning
print("\n--- KNN Hyperparameter Search ---")
knn_param_dist = {'n_neighbors': [3,5,7,9,11,15,20], 'weights': ['uniform','distance'],
                   'metric': ['euclidean','manhattan']}
knn_search = RandomizedSearchCV(KNeighborsRegressor(), knn_param_dist, n_iter=14, cv=kf,
                                 scoring='r2', random_state=42, n_jobs=-1)
knn_search.fit(X_train_s, y_train)
knn_best = knn_search.best_estimator_
y_pred_knn = knn_best.predict(X_test_s)

# KNN classification
knn_cls = RandomizedSearchCV(KNeighborsClassifier(), knn_param_dist, n_iter=14, cv=kf,
                              scoring='f1', random_state=42, n_jobs=-1)
knn_cls.fit(X_train_s, yc_train)
yc_pred_knn = knn_cls.best_estimator_.predict(X_test_s)

r2_knn = r2_score(y_test, y_pred_knn)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
f1_knn = f1_score(yc_test, yc_pred_knn)
acc_knn = accuracy_score(yc_test, yc_pred_knn)

print(f"Best KNN params: {knn_search.best_params_}")
print(f"KNN Test R²={r2_knn:.4f}, RMSE={rmse_knn:.6f}, F1={f1_knn:.4f}, Acc={acc_knn:.4f}")

# Other models
models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.001, max_iter=50000),
    'PLS': PLSRegression(n_components=10),
    'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
    'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=300, verbose=-1, seed=42),
}

results = []
print(f"\n{'Model':<20} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'F1':>8} {'Acc':>8} {'CV R²':>12}")
print("=" * 80)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    yp = model.predict(X_test_s)
    r2 = r2_score(y_test, yp)
    rmse = np.sqrt(mean_squared_error(y_test, yp))
    mae = mean_absolute_error(y_test, yp)
    cv = cross_val_score(model, X_train_s, y_train, cv=kf, scoring='r2')
    yc_pred = (yp >= median_dm).astype(int)
    f1 = f1_score(yc_test, yc_pred)
    acc = accuracy_score(yc_test, yc_pred)
    results.append({'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'F1': f1, 'Acc': acc, 'y_pred': yp})
    print(f"{name:<20} {r2:>8.4f} {rmse:>10.6f} {mae:>10.6f} {f1:>8.4f} {acc:>8.4f} {cv.mean():>8.4f}±{cv.std():.4f}")

results.append({'Model': 'KNN (Tuned)', 'R2': r2_knn, 'RMSE': rmse_knn, 'MAE': 0, 'F1': f1_knn, 'Acc': acc_knn, 'y_pred': y_pred_knn})
print(f"{'KNN (Tuned)':<20} {r2_knn:>8.4f} {rmse_knn:>10.6f} {'---':>10} {f1_knn:>8.4f} {acc_knn:>8.4f}")

# 4. KNN K-VALUE ANALYSIS
print("\n--- KNN: Effect of K ---")
print(f"{'K':>4} {'R²':>8} {'RMSE':>10} {'F1':>8}")
k_vals = [1,3,5,7,9,11,15,20,25,30]
knn_r2s, knn_f1s = [], []
for k in k_vals:
    m = KNeighborsRegressor(n_neighbors=k, weights='distance', metric='manhattan')
    m.fit(X_train_s, y_train)
    yp = m.predict(X_test_s)
    mc = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='manhattan')
    mc.fit(X_train_s, yc_train)
    knn_r2s.append(r2_score(y_test, yp))
    knn_f1s.append(f1_score(yc_test, mc.predict(X_test_s)))
    print(f"{k:>4} {r2_score(y_test,yp):>8.4f} {np.sqrt(mean_squared_error(y_test,yp)):>10.6f} {knn_f1s[-1]:>8.4f}")

# 5. GRAPHS
print("\n" + "=" * 70)
print("5. GENERATING GRAPHS")
print("=" * 70)

fig = plt.figure(figsize=(20, 24))

# 1: Wavelength vs Absorbance
ax1 = fig.add_subplot(4, 3, 1)
ax1.plot(wavelengths, np.abs(X).mean(axis=0), color='steelblue', linewidth=1.5)
ax1.fill_between(wavelengths, np.abs(X).mean(axis=0)-np.abs(X).std(axis=0),
                 np.abs(X).mean(axis=0)+np.abs(X).std(axis=0), alpha=0.3, color='steelblue')
ax1.set_xlabel('Wavelength (nm)'); ax1.set_ylabel('Absorbance')
ax1.set_title('Wavelength vs Mean Absorbance', fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2: Sample Spectra
ax2 = fig.add_subplot(4, 3, 2)
for i in range(min(10, X.shape[0])):
    ax2.plot(wavelengths, X[i], alpha=0.5, linewidth=0.8)
ax2.set_xlabel('Wavelength (nm)'); ax2.set_ylabel('Absorbance')
ax2.set_title('Individual Sample Spectra', fontweight='bold')

# 3: Dry Matter Distribution
ax3 = fig.add_subplot(4, 3, 3)
ax3.hist(y, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax3.axvline(y.mean(), color='red', linestyle='--', label=f'Mean={y.mean():.4f}')
ax3.axvline(median_dm, color='orange', linestyle='--', label=f'Median={median_dm:.4f}')
ax3.set_xlabel('Dry Matter'); ax3.set_title('Dry Matter Distribution', fontweight='bold')
ax3.legend()

# 4: Correlation
ax4 = fig.add_subplot(4, 3, 4)
ax4.plot(wavelengths, correlations, color='steelblue')
ax4.axhline(0, color='gray', linestyle='--')
ax4.set_xlabel('Wavelength (nm)'); ax4.set_ylabel('Correlation')
ax4.set_title('Wavelength-DM Correlation', fontweight='bold')

# 5: Model R²
ax5 = fig.add_subplot(4, 3, 5)
names = [r['Model'] for r in results]
r2s = [r['R2'] for r in results]
colors = ['coral' if m=='KNN (Tuned)' else 'steelblue' for m in names]
bars = ax5.barh(names, r2s, color=colors, edgecolor='black')
ax5.set_xlabel('R²'); ax5.set_title('Model Comparison - R²', fontweight='bold')
for b,v in zip(bars,r2s): ax5.text(v+0.005, b.get_y()+b.get_height()/2, f'{v:.3f}', va='center', fontsize=8)

# 6: Model F1
ax6 = fig.add_subplot(4, 3, 6)
f1s = [r['F1'] for r in results]
bars = ax6.barh(names, f1s, color=colors, edgecolor='black')
ax6.set_xlabel('F1'); ax6.set_title('Model Comparison - F1', fontweight='bold')
for b,v in zip(bars,f1s): ax6.text(v+0.005, b.get_y()+b.get_height()/2, f'{v:.3f}', va='center', fontsize=8)

# 7: Best Actual vs Predicted
ax7 = fig.add_subplot(4, 3, 7)
best = max(results, key=lambda x: x['R2'])
ax7.scatter(y_test, best['y_pred'], alpha=0.7, c='steelblue', edgecolors='k')
mn,mx = y_test.min(), y_test.max()
ax7.plot([mn,mx],[mn,mx],'r--',lw=2)
ax7.set_xlabel('Actual'); ax7.set_ylabel('Predicted')
ax7.set_title(f'Best ({best["Model"]}): R²={best["R2"]:.4f}', fontweight='bold')

# 8: KNN Actual vs Predicted
ax8 = fig.add_subplot(4, 3, 8)
ax8.scatter(y_test, y_pred_knn, alpha=0.7, c='coral', edgecolors='k')
ax8.plot([mn,mx],[mn,mx],'r--',lw=2)
ax8.set_xlabel('Actual'); ax8.set_ylabel('Predicted')
ax8.set_title(f'KNN: R²={r2_knn:.4f}', fontweight='bold')

# 9: Confusion Matrix
ax9 = fig.add_subplot(4, 3, 9)
cm = confusion_matrix(yc_test, yc_pred_knn)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax9,
            xticklabels=['Low','High'], yticklabels=['Low','High'])
ax9.set_xlabel('Predicted'); ax9.set_ylabel('Actual')
ax9.set_title('KNN Confusion Matrix', fontweight='bold')

# 10: KNN K-value
ax10 = fig.add_subplot(4, 3, 10)
ax10t = ax10.twinx()
ax10.plot(k_vals, knn_r2s, 'o-', color='steelblue', label='R²')
ax10t.plot(k_vals, knn_f1s, 's-', color='coral', label='F1')
ax10.set_xlabel('K'); ax10.set_ylabel('R²', color='steelblue')
ax10t.set_ylabel('F1', color='coral')
ax10.set_title('KNN: K vs Performance', fontweight='bold')
ax10.legend(loc='upper left'); ax10t.legend(loc='upper right')

# 11: Top 20 wavelengths correlation heatmap
ax11 = fig.add_subplot(4, 3, 11)
top20_corr = np.corrcoef(X_sel.T)
sns.heatmap(top20_corr, ax=ax11, cmap='coolwarm', center=0,
            xticklabels=[f'{int(w)}' for w in top20], yticklabels=[f'{int(w)}' for w in top20])
ax11.set_title('Top 20 Wavelengths Correlation', fontweight='bold')
ax11.tick_params(labelsize=7)

# 12: Residuals
ax12 = fig.add_subplot(4, 3, 12)
residuals = y_test - best['y_pred']
ax12.hist(residuals, bins=20, color='steelblue', edgecolor='black')
ax12.axvline(0, color='red', linestyle='--')
ax12.set_xlabel('Residual'); ax12.set_title('Residual Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('dry_matter_full_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# 6. SUMMARY
print("\n" + "=" * 70)
print("6. FINAL SUMMARY")
print("=" * 70)
results_sorted = sorted(results, key=lambda x: x['R2'], reverse=True)
print(f"\n{'Rank':>4} {'Model':<20} {'R²':>8} {'RMSE':>10} {'F1':>8} {'Acc':>8}")
print("-" * 62)
for i, r in enumerate(results_sorted, 1):
    print(f"{i:>4} {r['Model']:<20} {r['R2']:>8.4f} {r['RMSE']:>10.6f} {r['F1']:>8.4f} {r['Acc']:>8.4f}")

print(f"\nKey wavelengths: {top20.tolist()}")
print(f"KNN best params: {knn_search.best_params_}")
print(f"Saved: dry_matter_full_analysis.png")
