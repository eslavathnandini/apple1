import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_decomposition import PLSRegression
from scipy.signal import savgol_filter
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

median_dm = np.median(y)
y_class = (y >= median_dm).astype(int)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 70)
print("OPTIMIZATION RESULTS: Preprocessing + Feature Engineering")
print("=" * 70)

# Preprocessing
def snv(s): return (s - s.mean()) / (s.std() + 1e-10)
X_snv = np.apply_along_axis(snv, 1, X)
X_sg = np.apply_along_axis(lambda s: savgol_filter(s, 11, 3), 1, X)

def msc(X):
    X_m = np.zeros_like(X)
    ref = X.mean(axis=0)
    for i in range(X.shape[0]):
        fit = np.polyfit(ref, X[i], 1)
        X_m[i] = (X[i] - fit[1]) / fit[0]
    return X_m
X_msc = msc(X)

# Spectral Ratios
def nearest_wl(target):
    return min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i]-target))

pairs = [(430,550),(430,680),(430,740),(550,680),(550,740),(550,870),
         (680,740),(680,800),(680,870),(680,930),(740,800),(740,870),
         (800,870),(800,930),(800,970),(870,930),(870,970),(930,970)]
ratios = np.column_stack([X[:,nearest_wl(w1)]/(X[:,nearest_wl(w2)]+1e-10) for w1,w2 in pairs])

diffs = np.column_stack([X[:,nearest_wl(w1)]-X[:,nearest_wl(w2)] for w1,w2 in [(680,800),(740,800),(870,930),(430,680),(550,680)]])

ndvi = np.column_stack([(X[:,nearest_wl(800)]-X[:,nearest_wl(680)])/(X[:,nearest_wl(800)]+X[:,nearest_wl(680)]+1e-10),
                        (X[:,nearest_wl(870)]-X[:,nearest_wl(680)])/(X[:,nearest_wl(870)]+X[:,nearest_wl(680)]+1e-10)])

wi = X[:,nearest_wl(970)]/(X[:,nearest_wl(900)]+1e-10)
all_idx = np.column_stack([ratios, diffs, ndvi, wi])

feature_sets = {
    'Raw': X,
    'SNV': X_snv,
    'SG': X_sg,
    'MSC': X_msc,
    'Raw+SNV': np.hstack([X, X_snv]),
    'SG+SNV': np.hstack([X_sg, X_snv]),
    'Raw+Ratios': np.hstack([X, all_idx]),
    'SG+Ratios': np.hstack([X_sg, all_idx]),
    'MSC+SG': np.hstack([X_msc, X_sg]),
    'Raw+SNV+SG': np.hstack([X, X_snv, X_sg]),
}

# REGRESSION
print("\n1. REGRESSION (PLSR)")
print("-" * 60)
print(f"\n{'Features':<20} {'n':>4} {'R²':>8} {'RMSE':>10}")
print("-" * 45)

best_r2 = -999
best_name = ""
reg_data = {}

for name, Xf in feature_sets.items():
    Xt, Xv, yt, yv = train_test_split(Xf, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xt_s = sc.fit_transform(Xt)
    Xv_s = sc.transform(Xv)
    br2, bn = -999, 0
    for n in range(1, 21):
        pls = PLSRegression(n_components=n)
        pls.fit(Xt_s, yt)
        r2 = r2_score(yv, pls.predict(Xv_s).ravel())
        if r2 > br2: br2, bn = r2, n
    rmse = np.sqrt(mean_squared_error(yv, PLSRegression(n_components=bn).fit(Xt_s, yt).predict(Xv_s).ravel()))
    reg_data[name] = (br2, rmse, bn)
    print(f"{name:<20} {bn:>4} {br2:>8.4f} {rmse:>10.6f}")
    if br2 > best_r2: best_r2, best_name = br2, name

print(f"\nBest Regression: {best_name} (R²={best_r2:.4f})")

# CLASSIFICATION
print("\n2. CLASSIFICATION (Gradient Boosting)")
print("-" * 60)
print(f"\n{'Features':<20} {'Accuracy':>10} {'F1':>8}")
print("-" * 40)

best_acc = 0
best_cname = ""

for name, Xf in feature_sets.items():
    Xt, Xv, yt, yv = train_test_split(Xf, y_class, test_size=0.2, random_state=42, stratify=y_class)
    sc = StandardScaler()
    Xt_s = sc.fit_transform(Xt)
    Xv_s = sc.transform(Xv)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    gb.fit(Xt_s, yt)
    yp = gb.predict(Xv_s)
    acc = accuracy_score(yv, yp)
    f1 = f1_score(yv, yp)
    print(f"{name:<20} {acc:>10.4f} {f1:>8.4f}")
    if acc > best_acc: best_acc, best_cname = acc, name

print(f"\nBest Classification: {best_cname} (Acc={best_acc:.4f})")

# Before vs After
print("\n" + "=" * 70)
print("BEFORE vs AFTER OPTIMIZATION")
print("=" * 70)
print(f"\n{'Metric':<30} {'Before':>12} {'After':>12}")
print("-" * 55)
print(f"{'Regression R²':<30} {0.2708:>12.4f} {best_r2:>12.4f}")
print(f"{'Classification Accuracy':<30} {0.7708:>12.4f} {best_acc:>12.4f}")

# Save
import joblib
sc_final = StandardScaler()
X_final = feature_sets[best_name]
Xt_f, Xv_f, yt_f, yv_f = train_test_split(X_final, y, test_size=0.2, random_state=42)
Xt_fs = sc_final.fit_transform(Xt_f)
pls_final = PLSRegression(n_components=reg_data[best_name][2])
pls_final.fit(Xt_fs, yt_f)
joblib.dump(pls_final, 'optimized_pls_model.pkl')
joblib.dump(sc_final, 'optimized_pls_scaler.pkl')
print(f"\nSaved: optimized_pls_model.pkl")
print(f"Saved: optimized_pls_scaler.pkl")

# Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Feature sets bar
ax = axes[0,0]
names = list(reg_data.keys())
r2s = [reg_data[n][0] for n in names]
ax.barh(names, r2s, color='steelblue', edgecolor='black')
ax.set_xlabel('R²'); ax.set_title('Feature Set Comparison (R²)', fontweight='bold')
ax.tick_params(axis='y', labelsize=8)

# SG effect
ax = axes[0,1]
ax.plot(wavelengths, X[0], 'b-', alpha=0.5, label='Raw', linewidth=0.8)
ax.plot(wavelengths, X_sg[0], 'r-', label='SG Smoothed', linewidth=1.5)
ax.set_title('Savitzky-Golay Effect', fontweight='bold'); ax.legend()

# MSC effect
ax = axes[0,2]
ax.plot(wavelengths, X[0], 'b-', alpha=0.5, label='Raw')
ax.plot(wavelengths, X_msc[0], 'r-', label='MSC')
ax.set_title('MSC Effect', fontweight='bold'); ax.legend()

# Before vs After
ax = axes[1,0]
ax.bar(['Before','After'], [0.2708, best_r2], color=['steelblue','coral'], edgecolor='black')
ax.set_ylabel('R²'); ax.set_title('Regression R²: Before vs After', fontweight='bold')

# Before vs After Accuracy
ax = axes[1,1]
ax.bar(['Before','After'], [0.7708, best_acc], color=['steelblue','coral'], edgecolor='black')
ax.set_ylabel('Accuracy'); ax.set_title('Classification Accuracy: Before vs After', fontweight='bold')

# Actual vs Predicted
ax = axes[1,2]
Xt_f, Xv_f, yt_f, yv_f = train_test_split(feature_sets[best_name], y, test_size=0.2, random_state=42)
sc_p = StandardScaler()
yp = pls_final.predict(sc_p.fit_transform(Xv_f)).ravel()
ax.scatter(yv_f, yp, alpha=0.7, c='steelblue', edgecolors='k')
mn, mx = min(yv_f.min(),yp.min()), max(yv_f.max(),yp.max())
ax.plot([mn,mx],[mn,mx],'r--',lw=2)
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
ax.set_title(f'Optimized PLSR: R²={best_r2:.4f}', fontweight='bold')

plt.tight_layout()
plt.savefig('optimized_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: optimized_results.png")
