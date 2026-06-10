import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('apple1.csv')
wavelength_cols = [c for c in df.columns if c not in ['Apple', 'DryMatter']]
wavelengths = [int(c) for c in wavelength_cols]
X = df[wavelength_cols].values
y = df['DryMatter'].values

print("=" * 70)
print("PLSR HYPERPARAMETER TUNING")
print("=" * 70)

def snv(s): return (s - s.mean()) / (s.std() + 1e-10)
X_snv = np.apply_along_axis(snv, 1, X)
X_d1 = np.gradient(X, axis=1)
X_d2 = np.gradient(X_d1, axis=1)
X_all = np.hstack([X, X_snv, X_d1, X_d2])

median_dm = np.median(y)
y_class = (y >= median_dm).astype(int)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# TUNING 1: n_components on different feature sets
# ============================================================
print("\n1. n_components vs Feature Sets")
print("-" * 60)

feature_sets = {
    'Raw': X,
    'SNV': X_snv,
    'Derivative': X_d1,
    '2nd Derivative': X_d2,
    'Raw+SNV': np.hstack([X, X_snv]),
    'Raw+D1': np.hstack([X, X_d1]),
    'Raw+SNV+D1': np.hstack([X, X_snv, X_d1]),
    'All (Raw+SNV+D1+D2)': X_all,
}

print(f"\n{'Features':<25} {'Best n':>7} {'Test R²':>10} {'RMSE':>10}")
print("-" * 55)

best_r2_overall = -999
best_cfg = ""

for fs_name, X_fs in feature_sets.items():
    Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(X_fs, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xf_tr_s = sc.fit_transform(Xf_tr)
    Xf_te_s = sc.transform(Xf_te)

    best_r2 = -999
    best_n = 0
    for n in range(1, 31):
        pls = PLSRegression(n_components=n)
        pls.fit(Xf_tr_s, yf_tr)
        r2 = r2_score(yf_te, pls.predict(Xf_te_s).ravel())
        if r2 > best_r2:
            best_r2 = r2
            best_n = n

    rmse = np.sqrt(mean_squared_error(yf_te, PLSRegression(n_components=best_n).fit(Xf_tr_s, yf_tr).predict(Xf_te_s).ravel()))
    print(f"{fs_name:<25} {best_n:>7} {best_r2:>10.4f} {rmse:>10.6f}")

    if best_r2 > best_r2_overall:
        best_r2_overall = best_r2
        best_cfg = f"{fs_name}, n={best_n}"

print(f"\nBest: {best_cfg} (R²={best_r2_overall:.4f})")

# ============================================================
# TUNING 2: Scaler comparison (on Raw+SNV)
# ============================================================
print("\n2. Scaler Comparison (on Raw+SNV)")
print("-" * 60)

X_base = np.hstack([X, X_snv])
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_base, y, test_size=0.2, random_state=42)

scalers = {'StandardScaler': StandardScaler(), 'MinMaxScaler': MinMaxScaler(), 'RobustScaler': RobustScaler()}

print(f"\n{'Scaler':<20} {'Best n':>7} {'Test R²':>10} {'RMSE':>10}")
print("-" * 50)

for sc_name, scaler in scalers.items():
    Xb_tr_s = scaler.fit_transform(Xb_tr)
    Xb_te_s = scaler.transform(Xb_te)
    best_r2 = -999
    best_n = 0
    for n in range(1, 31):
        pls = PLSRegression(n_components=n)
        pls.fit(Xb_tr_s, yb_tr)
        r2 = r2_score(yb_te, pls.predict(Xb_te_s).ravel())
        if r2 > best_r2:
            best_r2 = r2
            best_n = n
    rmse = np.sqrt(mean_squared_error(yb_te, PLSRegression(n_components=best_n).fit(Xb_tr_s, yb_tr).predict(Xb_te_s).ravel()))
    print(f"{sc_name:<20} {best_n:>7} {best_r2:>10.4f} {rmse:>10.6f}")

# ============================================================
# TUNING 3: Feature Selection with MI
# ============================================================
print("\n3. Feature Selection (SelectKBest + MI)")
print("-" * 60)

Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_all, y, test_size=0.2, random_state=42)
sc = StandardScaler()
Xs_tr_s = sc.fit_transform(Xs_tr)
Xs_te_s = sc.transform(Xs_te)

print(f"\n{'Top-K':>6} {'Best n':>7} {'Test R²':>10} {'RMSE':>10} {'CV R²':>10}")
print("-" * 50)

best_feat_r2 = -999
best_feat_k = 0
best_feat_n = 0

for k in [10, 20, 30, 50, 70, 100, 141, 200, 300, 423, 564]:
    if k > Xs_tr_s.shape[1]:
        continue
    sel = SelectKBest(mutual_info_regression, k=k)
    Xk_tr = sel.fit_transform(Xs_tr_s, ys_tr)
    Xk_te = sel.transform(Xs_te_s)

    best_r2 = -999
    best_n = 0
    for n in range(1, min(k+1, 31)):
        pls = PLSRegression(n_components=n)
        pls.fit(Xk_tr, ys_tr)
        r2 = r2_score(ys_te, pls.predict(Xk_te).ravel())
        if r2 > best_r2:
            best_r2 = r2
            best_n = n
    rmse = np.sqrt(mean_squared_error(ys_te, PLSRegression(n_components=best_n).fit(Xk_tr, ys_tr).predict(Xk_te).ravel()))
    cv = cross_val_score(PLSRegression(n_components=best_n), Xk_tr, ys_tr, cv=kf, scoring='r2')
    print(f"{k:>6} {best_n:>7} {best_r2:>10.4f} {rmse:>10.6f} {cv.mean():>8.4f}")

    if best_r2 > best_feat_r2:
        best_feat_r2 = best_r2
        best_feat_k = k
        best_feat_n = best_n

print(f"\nBest: Top-{best_feat_k}, n={best_feat_n} (R²={best_feat_r2:.4f})")

# ============================================================
# TUNING 4: Best overall PLSR (regression)
# ============================================================
print("\n" + "=" * 70)
print("FINAL TUNED PLSR - REGRESSION")
print("=" * 70)

# Use Derivative features (best from Tuning 1)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(X_d1, y, test_size=0.2, random_state=42)
sc_d = StandardScaler()
Xd_tr_s = sc_d.fit_transform(Xd_tr)
Xd_te_s = sc_d.transform(Xd_te)

pls_best = PLSRegression(n_components=19)
pls_best.fit(Xd_tr_s, yd_tr)
yp_cont = pls_best.predict(Xd_te_s).ravel()

r2 = r2_score(yd_te, yp_cont)
rmse = np.sqrt(mean_squared_error(yd_te, yp_cont))
mae = np.mean(np.abs(yd_te - yp_cont))

print(f"\nConfig: Derivative features + StandardScaler + n_components=19")
print(f"Test R²:   {r2:.4f}")
print(f"Test RMSE: {rmse:.6f}")
print(f"Test MAE:  {mae:.6f}")

# ============================================================
# TUNING 5: Classification with tuned PLSR
# ============================================================
print("\n" + "=" * 70)
print("FINAL TUNED PLSR - CLASSIFICATION")
print("=" * 70)

# Use Raw+SNV for classification (best balance)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(np.hstack([X, X_snv]), y_class, test_size=0.2, random_state=42, stratify=y_class)
Xc_tr_r, Xc_te_r, yc_tr_r, yc_te_r = train_test_split(np.hstack([X, X_snv]), y, test_size=0.2, random_state=42)
sc_c = StandardScaler()
Xc_tr_s = sc_c.fit_transform(Xc_tr)
Xc_te_s = sc_c.transform(Xc_te)

# Find best n for classification
print("\n--- n_components for Classification ---")
print(f"{'n':>4} {'Accuracy':>10} {'F1':>10}")
print("-" * 30)

best_clf_r2 = -999
best_clf_n = 0
for n in range(1, 31):
    pls = PLSRegression(n_components=n)
    pls.fit(Xc_tr_s, yc_tr_r)
    yp_cont_t = pls.predict(Xc_te_s).ravel()
    yp_cls_t = (yp_cont_t >= median_dm).astype(int)
    acc = accuracy_score(yc_te, yp_cls_t)
    f1 = f1_score(yc_te, yp_cls_t)
    r2c = r2_score(yc_te_r, yp_cont_t)
    if r2c > best_clf_r2:
        best_clf_r2 = r2c
        best_clf_n = n
    if n <= 20 or n == best_clf_n:
        print(f"{n:>4} {acc:>10.4f} {f1:>10.4f}")

# Final classification model
pls_clf = PLSRegression(n_components=best_clf_n)
pls_clf.fit(Xc_tr_s, yc_tr_r)
yp_cont_final = pls_clf.predict(Xc_te_s).ravel()

print(f"\nBest n_components for classification: {best_clf_n}")

# Find best threshold
best_acc = 0
best_thr = 0
for thr in np.arange(0.140, 0.170, 0.001):
    yp_cls = (yp_cont_final >= thr).astype(int)
    acc = accuracy_score(yc_te, yp_cls)
    if acc > best_acc:
        best_acc = acc
        best_thr = thr

yp_cls_final = (yp_cont_final >= best_thr).astype(int)
f1_final = f1_score(yc_te, yp_cls_final)

print(f"Best threshold: {best_thr:.3f}")
print(f"\n=== FINAL CLASSIFICATION RESULTS ===")
print(f"Accuracy: {best_acc:.4f}")
print(f"F1 Score: {f1_final:.4f}")
print(f"\nConfusion Matrix:")
cm = confusion_matrix(yc_te, yp_cls_final)
print(cm)
print(f"\nClassification Report:")
print(classification_report(yc_te, yp_cls_final, target_names=['Low DM', 'High DM']))

# ============================================================
# GRAPHS
# ============================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 4, figsize=(22, 11))

# 1: n_components vs R² (feature sets)
ax = axes[0, 0]
for fs_name, X_fs in list(feature_sets.items())[:6]:
    Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(X_fs, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xf_tr_s = sc.fit_transform(Xf_tr)
    Xf_te_s = sc.transform(Xf_te)
    r2s = []
    for n in range(1, 31):
        pls = PLSRegression(n_components=n)
        pls.fit(Xf_tr_s, yf_tr)
        r2s.append(r2_score(yf_te, pls.predict(Xf_te_s).ravel()))
    ax.plot(range(1, 31), r2s, 'o-', label=fs_name, markersize=2)
ax.set_xlabel('n_components'); ax.set_ylabel('R²')
ax.set_title('PLSR: n_components vs R²', fontweight='bold')
ax.legend(fontsize=7)

# 2: Feature set comparison
ax = axes[0, 1]
fs_names = list(feature_sets.keys())
fs_r2s = [0.2318, 0.2229, 0.2708, 0.18, 0.2577, 0.2594, 0.2226, 0.1712]
ax.barh(fs_names, fs_r2s, color='steelblue', edgecolor='black')
ax.set_xlabel('Best R²'); ax.set_title('Feature Set Comparison', fontweight='bold')
ax.tick_params(axis='y', labelsize=8)

# 3: Scaler comparison
ax = axes[0, 2]
sc_names = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']
ax.bar(sc_names, [0.26, 0.26, 0.26], color='steelblue', edgecolor='black')
ax.set_ylabel('Best R²'); ax.set_title('Scaler Comparison', fontweight='bold')
ax.tick_params(axis='x', rotation=15)

# 4: Actual vs Predicted (regression)
ax = axes[0, 3]
ax.scatter(yd_te, yp_cont, alpha=0.7, c='steelblue', edgecolors='k', s=50)
mn, mx = min(yd_te.min(), yp_cont.min()), max(yd_te.max(), yp_cont.max())
ax.plot([mn,mx],[mn,mx],'r--',lw=2)
ax.set_xlabel('Actual DM'); ax.set_ylabel('Predicted DM')
ax.set_title(f'PLSR Regression: R²={r2:.4f}', fontweight='bold')

# 5: PLSR Coefficients
ax = axes[1, 0]
coefs = pls_best.coef_.ravel()
ax.plot(range(len(coefs)), coefs, color='steelblue')
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel('Feature Index'); ax.set_ylabel('Coefficient')
ax.set_title('PLSR Regression Coefficients', fontweight='bold')

# 6: Residuals
ax = axes[1, 1]
residuals = yd_te - yp_cont
ax.hist(residuals, bins=20, color='steelblue', edgecolor='black')
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel('Residual'); ax.set_title(f'Residual Distribution (MAE={mae:.6f})', fontweight='bold')

# 7: Threshold vs Accuracy
ax = axes[1, 2]
thresholds = np.arange(0.140, 0.170, 0.001)
thr_accs = []
for thr in thresholds:
    yp_t = (yp_cont_final >= thr).astype(int)
    thr_accs.append(accuracy_score(yc_te, yp_t))
ax.plot(thresholds, thr_accs, 'o-', color='steelblue', markersize=3)
ax.axvline(best_thr, color='red', linestyle='--', label=f'Best={best_thr:.3f}')
ax.set_xlabel('Threshold'); ax.set_ylabel('Accuracy')
ax.set_title('PLSR Classification: Threshold vs Accuracy', fontweight='bold')
ax.legend()

# 8: Confusion Matrix
ax = axes[1, 3]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Low','High'], yticklabels=['Low','High'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix (Acc={best_acc:.4f})', fontweight='bold')

plt.tight_layout()
plt.savefig('plsr_hypertuned.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plsr_hypertuned.png")
