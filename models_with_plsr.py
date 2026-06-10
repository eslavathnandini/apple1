import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, ExtraTreesClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
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

def snv(s): return (s - s.mean()) / (s.std() + 1e-10)
X_snv = np.apply_along_axis(snv, 1, X)
X_all = np.hstack([X, X_snv])

median_dm = np.median(y)
y_class = (y >= median_dm).astype(int)

X_train, X_test, y_train, y_test, yc_train, yc_test = train_test_split(
    X_all, y, y_class, test_size=0.2, random_state=42, stratify=y_class)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 70)
print("7 MODELS: 6 Original + PLSR")
print("=" * 70)

results = []

# 1. KNN
print("\n[1/7] KNN...", end=" ", flush=True)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan')
knn.fit(X_tr, yc_train)
yp = knn.predict(X_te)
acc = accuracy_score(yc_test, yp); f1 = f1_score(yc_test, yp)
results.append(('KNN', acc, f1, yp))
print(f"Acc={acc:.4f}, F1={f1:.4f}")

# 2. SVM
print("[2/7] SVM...", end=" ", flush=True)
svm = SVC(C=50, gamma='scale', kernel='rbf', probability=True)
svm.fit(X_tr, yc_train)
yp = svm.predict(X_te)
acc = accuracy_score(yc_test, yp); f1 = f1_score(yc_test, yp)
results.append(('SVM', acc, f1, yp))
print(f"Acc={acc:.4f}, F1={f1:.4f}")

# 3. Random Forest
print("[3/7] Random Forest...", end=" ", flush=True)
rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=3, n_jobs=-1, random_state=42)
rf.fit(X_tr, yc_train)
yp = rf.predict(X_te)
acc = accuracy_score(yc_test, yp); f1 = f1_score(yc_test, yp)
results.append(('Random Forest', acc, f1, yp))
print(f"Acc={acc:.4f}, F1={f1:.4f}")

# 4. Extra Trees
print("[4/7] Extra Trees...", end=" ", flush=True)
et = ExtraTreesClassifier(n_estimators=300, max_depth=20, min_samples_leaf=3, n_jobs=-1, random_state=42)
et.fit(X_tr, yc_train)
yp = et.predict(X_te)
acc = accuracy_score(yc_test, yp); f1 = f1_score(yc_test, yp)
results.append(('Extra Trees', acc, f1, yp))
print(f"Acc={acc:.4f}, F1={f1:.4f}")

# 5. Gradient Boosting
print("[5/7] Gradient Boosting...", end=" ", flush=True)
gbm = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
gbm.fit(X_tr, yc_train)
yp = gbm.predict(X_te)
acc = accuracy_score(yc_test, yp); f1 = f1_score(yc_test, yp)
results.append(('Gradient Boosting', acc, f1, yp))
print(f"Acc={acc:.4f}, F1={f1:.4f}")

# 6. LightGBM
print("[6/7] LightGBM...", end=" ", flush=True)
lgbm = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, verbose=-1, seed=42)
lgbm.fit(X_tr, yc_train)
yp = lgbm.predict(X_te)
acc = accuracy_score(yc_test, yp); f1 = f1_score(yc_test, yp)
results.append(('LightGBM', acc, f1, yp))
print(f"Acc={acc:.4f}, F1={f1:.4f}")

# 7. PLSR (NEW)
print("[7/7] PLSR...", end=" ", flush=True)
pls = PLSRegression(n_components=15)
pls.fit(X_tr, y_train)
yp_cont = pls.predict(X_te).ravel()
yp_cls = (yp_cont >= median_dm).astype(int)
acc = accuracy_score(yc_test, yp_cls); f1 = f1_score(yc_test, yp_cls)
results.append(('PLSR', acc, f1, yp_cls))
print(f"Acc={acc:.4f}, F1={f1:.4f}")

# Voting
print("\n[BONUS] Vote-Soft (all 7)...", end=" ", flush=True)
estimators = [('KNN', knn), ('SVM', svm), ('RF', rf), ('ET', et), ('GBM', gbm), ('LGB', lgbm)]
vc = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
vc.fit(X_tr, yc_train)
yp = vc.predict(X_te)
acc = accuracy_score(yc_test, yp); f1 = f1_score(yc_test, yp)
results.append(('Vote-Soft', acc, f1, yp))
print(f"Acc={acc:.4f}, F1={f1:.4f}")

# Sort
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

print("\n" + "=" * 70)
print("FINAL RESULTS (sorted by Accuracy)")
print("=" * 70)
print(f"{'Rank':>4} {'Model':<25} {'Accuracy':>10} {'F1':>8}")
print("-" * 50)
for i, (n, acc, f1, _) in enumerate(results_sorted, 1):
    m = " <-- BEST" if i == 1 else ""
    print(f"{i:>4} {n:<25} {acc:>10.4f} {f1:>8.4f}{m}")

# Confusion Matrix for best
best = results_sorted[0]
print(f"\nConfusion Matrix ({best[0]}):")
print(confusion_matrix(yc_test, best[3]))
print(f"\nClassification Report ({best[0]}):")
print(classification_report(yc_test, best[3], target_names=['Low DM', 'High DM']))

# Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

names = [r[0] for r in results_sorted]
accs = [r[1] for r in results_sorted]
f1s = [r[2] for r in results_sorted]
colors = ['green' if a >= 0.90 else 'steelblue' for a in accs]

axes[0,0].barh(names, accs, color=colors, edgecolor='black')
axes[0,0].axvline(0.95, color='green', linestyle='--', lw=2, label='95%')
axes[0,0].set_xlabel('Accuracy'); axes[0,0].set_title('Accuracy Comparison', fontweight='bold')
axes[0,0].legend()

axes[0,1].barh(names, f1s, color=colors, edgecolor='black')
axes[0,1].set_xlabel('F1 Score'); axes[0,1].set_title('F1 Score Comparison', fontweight='bold')

cm = confusion_matrix(yc_test, best[3])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,2], xticklabels=['Low','High'], yticklabels=['Low','High'])
axes[0,2].set_title(f'Confusion Matrix ({best[0]})', fontweight='bold')

axes[1,0].plot(wavelengths, X.mean(0), color='steelblue')
axes[1,0].fill_between(wavelengths, X.mean(0)-X.std(0), X.mean(0)+X.std(0), alpha=0.3)
axes[1,0].set_title('Wavelength vs Absorbance', fontweight='bold')

axes[1,1].hist(y, bins=30, color='steelblue', edgecolor='black')
axes[1,1].axvline(median_dm, color='orange', linestyle='--')
axes[1,1].set_title('Dry Matter Distribution', fontweight='bold')

# PLSR: components vs R²
n_range = range(1, 31)
pls_r2s = []
for n in n_range:
    pls_t = PLSRegression(n_components=n)
    pls_t.fit(X_tr, y_train)
    pls_r2s.append(np.corrcoef(yc_test, (pls_t.predict(X_te).ravel() >= median_dm).astype(int))[0,1]**2)
axes[1,2].plot(list(n_range), pls_r2s, 'o-', color='steelblue')
axes[1,2].set_xlabel('Components'); axes[1,2].set_ylabel('R²')
axes[1,2].set_title('PLSR: Components vs Accuracy', fontweight='bold')

plt.tight_layout()
plt.savefig('models_with_plsr.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: models_with_plsr.png")
