import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (mean_squared_error, r2_score, f1_score, confusion_matrix,
                             accuracy_score, classification_report, mean_absolute_error)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              VotingClassifier, ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv('apple1.csv')
wavelength_cols = [c for c in df.columns if c not in ['Apple', 'DryMatter']]
wavelengths = [int(c) for c in wavelength_cols]
X = df[wavelength_cols].values
y = df['DryMatter'].values

print("=" * 70)
print("COMPLETE ML PIPELINE: Dry Matter Prediction")
print("=" * 70)

# --- EDA ---
print("\n1. DATA EXPLORATORY ANALYSIS")
print("-" * 50)
print(f"Dataset: {df.shape[0]} samples, {df.shape[1]-2} wavelengths ({min(wavelengths)}-{max(wavelengths)} nm)")
print(f"DryMatter: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
print(f"Missing: {df.isnull().sum().sum()}")

correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
corr_df = pd.DataFrame({'nm': wavelengths, 'r': correlations}).sort_values('r', key=abs, ascending=False)
print(f"\nTop 10 Correlated Wavelengths:")
print(corr_df.head(10).to_string(index=False))

# --- Preprocessing ---
print("\n2. PREPROCESSING")
print("-" * 50)
def snv(s): return (s - s.mean()) / (s.std() + 1e-10)
X_snv = np.apply_along_axis(snv, 1, X)
X_all = np.hstack([X, X_snv])
print(f"Features: Raw({X.shape[1]}) + SNV({X_snv.shape[1]}) = {X_all.shape[1]}")

median_dm = np.median(y)
y_class = (y >= median_dm).astype(int)
print(f"Classification: median={median_dm:.4f}, Low={((y_class==0).sum())}, High={((y_class==1).sum())}")

X_tr_, X_te_, y_tr_, y_te_, yc_tr_, yc_te_ = train_test_split(
    X_all, y, y_class, test_size=0.2, random_state=42, stratify=y_class)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr_)
X_te = scaler.transform(X_te_)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- REGRESSION ---
print("\n3. REGRESSION MODELS (Predict DryMatter value)")
print("-" * 50)

reg_models = {
    'Ridge': Ridge(alpha=1.0),
    'PLSR (n=10)': PLSRegression(n_components=10),
    'PLSR (n=15)': PLSRegression(n_components=15),
    'PLSR (n=20)': PLSRegression(n_components=20),
    'KNN Reg': KNeighborsRegressor(n_neighbors=7, weights='distance', metric='manhattan'),
    'Random Forest Reg': RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3, n_jobs=-1, random_state=42),
    'Extra Trees Reg': ExtraTreesRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3, n_jobs=-1, random_state=42),
    'Gradient Boosting Reg': GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
    'LightGBM Reg': lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05, verbose=-1, seed=42),
}

print(f"\n{'Model':<25} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
print("=" * 55)

reg_results = []
for name, model in reg_models.items():
    model.fit(X_tr, y_tr_)
    yp = model.predict(X_te)
    r2 = r2_score(y_te_, yp)
    rmse = np.sqrt(mean_squared_error(y_te_, yp))
    mae = mean_absolute_error(y_te_, yp)
    reg_results.append((name, r2, rmse, mae, yp))
    print(f"{name:<25} {r2:>8.4f} {rmse:>10.6f} {mae:>10.6f}")

reg_sorted = sorted(reg_results, key=lambda x: x[1], reverse=True)
print(f"\nBest Regression: {reg_sorted[0][0]} (R²={reg_sorted[0][1]:.4f})")

# --- CLASSIFICATION ---
print("\n4. CLASSIFICATION MODELS (Predict High/Low)")
print("-" * 50)

clf_models = {
    'KNN Clf': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan'),
    'SVM Clf': SVC(C=50, gamma='scale', kernel='rbf', probability=True),
    'Random Forest Clf': RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=3, n_jobs=-1, random_state=42),
    'Extra Trees Clf': ExtraTreesClassifier(n_estimators=300, max_depth=20, min_samples_leaf=3, n_jobs=-1, random_state=42),
    'Gradient Boosting Clf': GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
    'LightGBM Clf': lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, verbose=-1, seed=42),
}

print(f"\n{'Model':<25} {'Accuracy':>10} {'F1':>8}")
print("=" * 45)

clf_results = []
for name, model in clf_models.items():
    model.fit(X_tr, yc_tr_)
    yp = model.predict(X_te)
    acc = accuracy_score(yc_te_, yp)
    f1 = f1_score(yc_te_, yp)
    clf_results.append((name, acc, f1, yp))
    print(f"{name:<25} {acc:>10.4f} {f1:>8.4f}")

# Voting
for vn, vt in [('Vote-Hard', 'hard'), ('Vote-Soft', 'soft')]:
    vc = VotingClassifier(estimators=list(clf_models.items()), voting=vt, n_jobs=-1)
    vc.fit(X_tr, yc_tr_)
    yp = vc.predict(X_te)
    acc = accuracy_score(yc_te_, yp)
    f1 = f1_score(yc_te_, yp)
    clf_results.append((vn, acc, f1, yp))
    print(f"{vn:<25} {acc:>10.4f} {f1:>8.4f}")

# PLSR as classifier
for nc in [10, 15, 20]:
    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_tr_)
    yp_c = (pls.predict(X_te).ravel() >= median_dm).astype(int)
    acc = accuracy_score(yc_te_, yp_c)
    f1 = f1_score(yc_te_, yp_c)
    clf_results.append((f'PLSR Clf(n={nc})', acc, f1, yp_c))
    print(f"{'PLSR Clf(n='+str(nc)+')':<25} {acc:>10.4f} {f1:>8.4f}")

clf_sorted = sorted(clf_results, key=lambda x: x[1], reverse=True)
print(f"\nBest Classification: {clf_sorted[0][0]} (Acc={clf_sorted[0][1]:.4f}, F1={clf_sorted[0][2]:.4f})")

# --- SUMMARY ---
print("\n" + "=" * 70)
print("5. COMPLETE RESULTS SUMMARY")
print("=" * 70)

print("\nREGRESSION (sorted by R²):")
print(f"{'Rank':>3} {'Model':<25} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
print("-" * 60)
for i, (n, r2, rmse, mae, _) in enumerate(reg_sorted, 1):
    print(f"{i:>3} {n:<25} {r2:>8.4f} {rmse:>10.6f} {mae:>10.6f}")

print(f"\nCLASSIFICATION (sorted by Accuracy):")
print(f"{'Rank':>3} {'Model':<25} {'Accuracy':>10} {'F1':>8}")
print("-" * 50)
for i, (n, acc, f1, _) in enumerate(clf_sorted, 1):
    print(f"{i:>3} {n:<25} {acc:>10.4f} {f1:>8.4f}")

# --- SAVE ---
print("\n6. SAVING MODELS")
print("-" * 50)

# RF regression
rf_reg = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3, n_jobs=-1, random_state=42)
rf_reg.fit(X_tr, y_tr_)
joblib.dump(rf_reg, 'random_forest_model.pkl')
print("Saved: random_forest_model.pkl")

# PLSR
pls = PLSRegression(n_components=15)
pls.fit(X_tr, y_tr_)
joblib.dump(pls, 'pls_regression_model.pkl')
print("Saved: pls_regression_model.pkl")

# Best RF classification
rf_clf = clf_models['Random Forest Clf']
joblib.dump(rf_clf, 'random_forest_classifier.pkl')
print("Saved: random_forest_classifier.pkl")

# Best GBM classification
gbm_clf = clf_models['Gradient Boosting Clf']
joblib.dump(gbm_clf, 'gradient_boosting_classifier.pkl')
print("Saved: gradient_boosting_classifier.pkl")

# Scaler
joblib.dump(scaler, 'scaler.pkl')
print("Saved: scaler.pkl")

# --- PREDICTION INSTRUCTIONS ---
print("\n7. HOW TO PREDICT (Prediction Code)")
print("-" * 50)
print("""
import numpy as np
import joblib

# Load
rf_model = joblib.load('random_forest_model.pkl')
pls_model = joblib.load('pls_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Input: 141 absorption values (430nm to 990nm, 4nm step)
my_apple = [0.12, 0.15, 0.18, ...]  # 141 values

# Preprocess
X_raw = np.array(my_apple).reshape(1, -1)
X_snv = (X_raw - X_raw.mean()) / (X_raw.std() + 1e-10)
X_combined = np.hstack([X_raw, X_snv])
X_scaled = scaler.transform(X_combined)

# Predict with Random Forest
rf_pred = rf_model.predict(X_scaled)
print(f'Random Forest DryMatter: {rf_pred[0]:.4f}')

# Predict with PLSR
pls_pred = pls_model.predict(X_scaled)
print(f'PLSR DryMatter: {pls_pred[0]:.4f}')
""")

# --- GRAPHS ---
print("8. GENERATING GRAPHS...")
fig = plt.figure(figsize=(22, 18))

# Row 1: EDA
ax = fig.add_subplot(3, 5, 1)
ax.plot(wavelengths, X.mean(0), color='steelblue')
ax.fill_between(wavelengths, X.mean(0)-X.std(0), X.mean(0)+X.std(0), alpha=0.3)
ax.set_title('Wavelength vs Absorbance', fontweight='bold', fontsize=9)
ax.set_xlabel('Wavelength (nm)', fontsize=8)

ax = fig.add_subplot(3, 5, 2)
ax.hist(y, bins=30, color='steelblue', edgecolor='black')
ax.axvline(median_dm, color='orange', linestyle='--')
ax.set_title('Dry Matter Distribution', fontweight='bold', fontsize=9)

ax = fig.add_subplot(3, 5, 3)
ax.plot(wavelengths, correlations, color='steelblue')
ax.axhline(0, color='gray', linestyle='--')
ax.set_title('Wavelength-DM Correlation', fontweight='bold', fontsize=9)

ax = fig.add_subplot(3, 5, 4)
for i in range(min(10, X.shape[0])):
    ax.plot(wavelengths, X[i], alpha=0.5, linewidth=0.8)
ax.set_title('Raw Sample Spectra', fontweight='bold', fontsize=9)

ax = fig.add_subplot(3, 5, 5)
for i in range(min(10, X_snv.shape[0])):
    ax.plot(wavelengths, X_snv[i], alpha=0.5, linewidth=0.8)
ax.set_title('SNV Spectra', fontweight='bold', fontsize=9)

# Row 2: Regression
ax = fig.add_subplot(3, 5, 6)
rn = [r[0] for r in reg_sorted]
rv = [r[1] for r in reg_sorted]
ax.barh(rn, rv, color='steelblue', edgecolor='black')
ax.set_xlabel('R²'); ax.set_title('Regression: R² Comparison', fontweight='bold', fontsize=9)
ax.tick_params(axis='y', labelsize=7)

# RF Actual vs Predicted
ax = fig.add_subplot(3, 5, 7)
yp_rf = reg_results[[r[0] for r in reg_results].index('Random Forest Reg')][4]
ax.scatter(y_te_, yp_rf, alpha=0.7, c='steelblue', edgecolors='k', s=40)
mn, mx = min(y_te_.min(), yp_rf.min()), max(y_te_.max(), yp_rf.max())
ax.plot([mn,mx],[mn,mx],'r--',lw=2)
ax.set_title(f'Random Forest: R²={r2_score(y_te_,yp_rf):.4f}', fontweight='bold', fontsize=9)
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')

# PLSR Actual vs Predicted
ax = fig.add_subplot(3, 5, 8)
yp_pls = reg_results[[r[0] for r in reg_results].index('PLSR (n=15)')][4]
ax.scatter(y_te_, yp_pls, alpha=0.7, c='coral', edgecolors='k', s=40)
ax.plot([mn,mx],[mn,mx],'r--',lw=2)
ax.set_title(f'PLSR: R²={r2_score(y_te_,yp_pls):.4f}', fontweight='bold', fontsize=9)
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')

# RF Feature Importance
ax = fig.add_subplot(3, 5, 9)
imp = rf_reg.feature_importances_[:141]
ax.plot(wavelengths, imp, color='steelblue')
ax.set_title('RF Feature Importance', fontweight='bold', fontsize=9)

# PLSR Coefficients
ax = fig.add_subplot(3, 5, 10)
coefs = pls.coef_[:141].ravel()
ax.plot(wavelengths, coefs, color='steelblue')
ax.axhline(0, color='gray', linestyle='--')
ax.set_title('PLSR Coefficients', fontweight='bold', fontsize=9)

# Row 3: Classification
ax = fig.add_subplot(3, 5, 11)
cn = [c[0] for c in clf_sorted]
ca = [c[1] for c in clf_sorted]
ax.barh(cn[:8], ca[:8], color='steelblue', edgecolor='black')
ax.axvline(0.95, color='green', linestyle='--', lw=2, label='95%')
ax.set_xlabel('Accuracy'); ax.set_title('Classification: Accuracy', fontweight='bold', fontsize=9)
ax.legend(); ax.tick_params(axis='y', labelsize=7)

ax = fig.add_subplot(3, 5, 12)
cm = confusion_matrix(yc_te_, clf_sorted[0][3])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Low','High'], yticklabels=['Low','High'])
ax.set_title(f'CM ({clf_sorted[0][0]})', fontweight='bold', fontsize=9)

ax = fig.add_subplot(3, 5, 13)
ax.barh(cn[:8], [c[2] for c in clf_sorted[:8]], color='steelblue', edgecolor='black')
ax.set_xlabel('F1 Score'); ax.set_title('Classification: F1 Score', fontweight='bold', fontsize=9)
ax.tick_params(axis='y', labelsize=7)

ax = fig.add_subplot(3, 5, 14)
res_rf = y_te_ - yp_rf
ax.hist(res_rf, bins=20, color='steelblue', edgecolor='black')
ax.axvline(0, color='red', linestyle='--')
ax.set_title('RF Residuals', fontweight='bold', fontsize=9)

ax = fig.add_subplot(3, 5, 15)
res_pls = y_te_ - yp_pls
ax.hist(res_pls, bins=20, color='coral', edgecolor='black')
ax.axvline(0, color='red', linestyle='--')
ax.set_title('PLSR Residuals', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('dry_matter_complete_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved: dry_matter_complete_analysis.png")
print("\n" + "=" * 70)
print("FILES CREATED:")
print("=" * 70)
print("  random_forest_model.pkl      - RF regression model")
print("  pls_regression_model.pkl     - PLSR regression model")
print("  random_forest_classifier.pkl - RF classification model")
print("  gradient_boosting_classifier.pkl - GBM classification model")
print("  scaler.pkl                   - StandardScaler for predictions")
print("  dry_matter_complete_analysis.png - All 15 graphs")
print("=" * 70)
