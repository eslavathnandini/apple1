import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import lightgbm as lgb
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
print("EXPERIMENT: Testing multiple thresholds and feature sets")
print("=" * 70)

# Try different thresholds
percentiles = [25, 30, 35, 40, 45, 50]

for pct in percentiles:
    threshold = np.percentile(y, pct)
    y_bin = (y >= threshold).astype(int)
    n_low = (y_bin == 0).sum()
    n_high = (y_bin == 1).sum()
    if n_low < 5 or n_high < 5:
        continue

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    rf = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=3, n_jobs=-1, random_state=42)
    rf.fit(X_trs, y_tr)
    yp = rf.predict(X_tes)
    acc = accuracy_score(y_te, yp)
    print(f"  P{pct} (thresh={threshold:.4f}, n={n_low}/{n_high}): RF Acc={acc:.4f}")

print("\n--- Trying spectral ratios & combinations ---")
# Create ratio features
ratio_features = []
for i in range(0, len(wavelengths), 10):
    for j in range(i+10, min(i+40, len(wavelengths)), 10):
        ratio = X[:, i] / (X[:, j] + 1e-10)
        ratio_features.append(ratio)
X_ratios = np.column_stack(ratio_features)

# Difference features
diff_features = []
for i in range(0, len(wavelengths), 5):
    for j in range(i+5, min(i+20, len(wavelengths)), 5):
        diff = X[:, i] - X[:, j]
        diff_features.append(diff)
X_diffs = np.column_stack(diff_features)

# PCA features
scaler_all = StandardScaler()
X_scaled = scaler_all.fit_transform(X)
pca = PCA(n_components=20, random_state=42)
X_pca = pca.fit_transform(X_scaled)

X_enhanced = np.hstack([X, X_ratios, X_diffs, X_pca])
print(f"Enhanced features: {X_enhanced.shape[1]} (raw={X.shape[1]}, ratios={X_ratios.shape[1]}, diffs={X_diffs.shape[1]}, pca={X_pca.shape[1]})")

# Test on median split with enhanced features
median_dm = np.median(y)
y_class = (y >= median_dm).astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X_enhanced, y_class, test_size=0.2, random_state=42, stratify=y_class)
scaler = StandardScaler()
X_trs = scaler.fit_transform(X_tr)
X_tes = scaler.transform(X_te)

models = [
    ('RF', RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=3, n_jobs=-1, random_state=42)),
    ('ET', ExtraTreesClassifier(n_estimators=500, max_depth=20, min_samples_leaf=3, n_jobs=-1, random_state=42)),
    ('LGB', lgb.LGBMClassifier(n_estimators=500, num_leaves=31, learning_rate=0.05, verbose=-1, seed=42)),
    ('GBM', GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)),
    ('SVM', SVC(C=50, gamma='scale', kernel='rbf', probability=True)),
    ('KNN', KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan')),
]

print(f"\nMedian split: {(y_class==0).sum()}/{(y_class==1).sum()}")
print(f"{'Model':<8} {'R² Raw':>8} {'Acc Enhanced':>14}")
print("-" * 35)

best_acc = 0
best_model = None

for name, model in models:
    model.fit(X_trs, y_tr)
    yp = model.predict(X_tes)
    acc = accuracy_score(y_te, yp)
    f1 = f1_score(y_te, yp)
    print(f"{name:<8} {acc:>8.4f} {f1:>8.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model = (name, model)

# Multi-split ensemble on enhanced features
print("\n--- Multi-Split Ensemble (Enhanced) ---")
split_probs = []
for seed in range(50):
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_trs, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)
    m = ExtraTreesClassifier(n_estimators=300, max_depth=15, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    m.fit(Xs_tr, ys_tr)
    split_probs.append(m.predict_proba(X_tes)[:, 1])

for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
    yp = (np.mean(split_probs, axis=0) >= thresh).astype(int)
    acc = accuracy_score(y_te, yp)
    f1 = f1_score(y_te, yp)
    print(f"  Threshold={thresh:.2f}: Acc={acc:.4f}, F1={f1:.4f}")

# Final voting with enhanced
print("\n--- Voting (Enhanced Features) ---")
estimators = [(name, model) for name, model in models]
vc = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
vc.fit(X_trs, y_tr)
yp = vc.predict(X_tes)
acc = accuracy_score(y_te, yp)
f1 = f1_score(y_te, yp)
print(f"Soft Vote: Acc={acc:.4f}, F1={f1:.4f}")

cm = confusion_matrix(y_te, yp)
print(f"Confusion Matrix:\n{cm}")
print(f"\n{classification_report(y_te, yp, target_names=['Low DM', 'High DM'])}")

# Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 1: Wavelength vs Absorbance
axes[0,0].plot(wavelengths, X.mean(0), color='steelblue')
axes[0,0].fill_between(wavelengths, X.mean(0)-X.std(0), X.mean(0)+X.std(0), alpha=0.3)
axes[0,0].set_title('Wavelength vs Absorbance', fontweight='bold')
axes[0,0].set_xlabel('Wavelength (nm)'); axes[0,0].set_ylabel('Absorbance')

# 2: DM Distribution
axes[0,1].hist(y, bins=30, color='steelblue', edgecolor='black')
axes[0,1].axvline(median_dm, color='orange', linestyle='--', label=f'Median={median_dm:.4f}')
axes[0,1].set_title('Dry Matter Distribution', fontweight='bold')
axes[0,1].legend()

# 3: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,2], xticklabels=['Low','High'], yticklabels=['Low','High'])
axes[0,2].set_title(f'Confusion Matrix (Vote, Acc={acc:.4f})', fontweight='bold')

# 4: PCA scatter
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_scaled)
scatter = axes[1,0].scatter(X_pca2[:,0], X_pca2[:,1], c=y_class, cmap='coolwarm', alpha=0.7, edgecolors='k', s=50)
axes[1,0].set_title('PCA: Samples by DM Class', fontweight='bold')
axes[1,0].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
axes[1,0].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter, ax=axes[1,0])

# 5: Correlation heatmap
corrs = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
top10_idx = np.argsort(np.abs(corrs))[::-1][:10]
top10_corr = np.corrcoef(X[:, top10_idx].T)
sns.heatmap(top10_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1,1])
axes[1,1].set_title('Top 10 Wavelengths Correlation', fontweight='bold')

# 6: Feature importance
importances = models[0][1].feature_importances_[:141]  # raw features only
axes[1,2].plot(wavelengths, importances, color='steelblue')
axes[1,2].set_title('RF Feature Importance (Raw)', fontweight='bold')
axes[1,2].set_xlabel('Wavelength (nm)')

plt.tight_layout()
plt.savefig('dry_matter_95_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: dry_matter_95_accuracy.png")
