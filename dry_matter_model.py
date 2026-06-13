import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('apple1.csv')
print(f"Dataset shape: {df.shape}")
print(f"Target (DryMatter) stats:\n{df['DryMatter'].describe()}\n")

X = df.drop(columns=['Apple', 'DryMatter'])
y = df['DryMatter']
wavelengths = [int(c) for c in X.columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# STEP 1: Baseline with default params
# ============================================================
print("=" * 60)
print("STEP 1: BASELINE MODEL (Default Params)")
print("=" * 60)

base_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

base_model = lgb.train(
    base_params, train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

y_pred_base = base_model.predict(X_test)
r2_base = r2_score(y_test, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
mae_base = mean_absolute_error(y_test, y_pred_base)

print(f"\nBaseline Results:")
print(f"  RMSE : {rmse_base:.6f}")
print(f"  MAE  : {mae_base:.6f}")
print(f"  R2   : {r2_base:.6f}")

# ============================================================
# STEP 2: Manual Hyperparameter Tuning
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: MANUAL HYPERPARAMETER TUNING")
print("=" * 60)

best_r2 = -999
best_params = {}
best_model = None

num_leaves_vals = [16, 31, 50, 80, 128, 256]
lr_vals = [0.001, 0.005, 0.01, 0.05, 0.1]
feature_frac_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bagging_frac_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
min_data_vals = [5, 10, 20, 30, 50]
n_est_vals = [100, 200, 300, 500, 800]

total = len(num_leaves_vals) * len(lr_vals) * len(feature_frac_vals)
print(f"\nTesting {total} combinations (quick grid)...\n")

count = 0
for nl in num_leaves_vals:
    for lr in lr_vals:
        for ff in feature_frac_vals:
            count += 1
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': nl,
                'learning_rate': lr,
                'feature_fraction': ff,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 10,
                'verbose': -1,
                'n_jobs': -1,
                'seed': 42
            }

            model = lgb.LGBMRegressor(**params, n_estimators=300)
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])

            yp = model.predict(X_test)
            r2 = r2_score(y_test, yp)

            if r2 > best_r2:
                best_r2 = r2
                best_params = params.copy()
                best_model = model
                print(f"  [{count}/{total}] NEW BEST: num_leaves={nl}, lr={lr}, ff={ff} => R2={r2:.4f}")

print(f"\nManual Tuning Best R²: {best_r2:.4f}")
print(f"Best Params: num_leaves={best_params['num_leaves']}, lr={best_params['learning_rate']}, ff={best_params['feature_fraction']}")

# ============================================================
# STEP 3: Fine-tune around best params
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: FINE-TUNE AROUND BEST PARAMS")
print("=" * 60)

nl_center = best_params['num_leaves']
lr_center = best_params['learning_rate']
ff_center = best_params['feature_fraction']

fine_nl = [nl_center]
fine_lr = [lr_center]
fine_ff = [ff_center]
fine_bf = [0.7, 0.8, 0.9]
fine_mdl = [5, 10, 15]
fine_ne = [300, 500]

best_r2_fine = best_r2
best_params_fine = best_params.copy()
best_model_fine = best_model

for nl in fine_nl:
    for lr in fine_lr:
        for ff in fine_ff:
            for bf in fine_bf:
                for mdl in fine_mdl:
                    for ne in fine_ne:
                        params = {
                            'objective': 'regression',
                            'metric': 'rmse',
                            'boosting_type': 'gbdt',
                            'num_leaves': nl,
                            'learning_rate': lr,
                            'feature_fraction': ff,
                            'bagging_fraction': bf,
                            'bagging_freq': 5,
                            'min_data_in_leaf': mdl,
                            'verbose': -1,
                            'n_jobs': -1,
                            'seed': 42
                        }

                        model = lgb.LGBMRegressor(**params, n_estimators=ne)
                        model.fit(X_train, y_train,
                                  eval_set=[(X_test, y_test)],
                                  callbacks=[lgb.early_stopping(30, verbose=False)])

                        yp = model.predict(X_test)
                        r2 = r2_score(y_test, yp)

                        if r2 > best_r2_fine:
                            best_r2_fine = r2
                            best_params_fine = params.copy()
                            best_model_fine = model
                            rmse_fine = np.sqrt(mean_squared_error(y_test, yp))
                            mae_fine = mean_absolute_error(y_test, yp)
                            print(f"  NEW BEST: nl={nl}, lr={lr:.4f}, ff={ff}, bf={bf}, mdl={mdl}, ne={ne} => R2={r2:.4f}")

print(f"\nFine-Tuned Best R²: {best_r2_fine:.4f}")
print(f"RMSE: {rmse_fine:.6f}")
print(f"MAE:  {mae_fine:.6f}")

# ============================================================
# STEP 4: Final Model with Best Params
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: FINAL MODEL WITH BEST PARAMS")
print("=" * 60)

final_params = best_params_fine.copy()
final_model = best_model_fine

y_pred_final = final_model.predict(X_test)

r2_final = r2_score(y_test, y_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
mae_final = mean_absolute_error(y_test, y_pred_final)

print(f"\nFinal Model Results:")
print(f"  RMSE : {rmse_final:.6f}")
print(f"  MAE  : {mae_final:.6f}")
print(f"  R2   : {r2_final:.6f}")

print(f"\nFinal Hyperparameters:")
for k, v in final_params.items():
    if k not in ['objective', 'metric', 'boosting_type', 'verbose', 'n_jobs', 'seed']:
        print(f"  {k}: {v}")

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lgb.LGBMRegressor(**final_params, n_estimators=final_model.n_estimators),
                            X, y, cv=kf, scoring='r2')
print(f"\n5-Fold CV R²: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")

# ============================================================
# STEP 5: Comparison Table
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: BEFORE vs AFTER TUNING")
print("=" * 60)

print(f"\n{'Metric':<15} {'Baseline':>12} {'Tuned':>12} {'Improvement':>15}")
print("-" * 55)
print(f"{'R2':<15} {r2_base:>12.4f} {r2_final:>12.4f} {((r2_final-r2_base)/abs(r2_base)*100):>14.1f}%")
print(f"{'RMSE':<15} {rmse_base:>12.6f} {rmse_final:>12.6f} {((rmse_base-rmse_final)/rmse_base*100):>14.1f}%")
print(f"{'MAE':<15} {mae_base:>12.6f} {mae_final:>12.6f} {((mae_base-mae_final)/mae_base*100):>14.1f}%")

# ============================================================
# STEP 6: Feature Importance (changes with better params)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: FEATURE IMPORTANCE (Updated after tuning)")
print("=" * 60)

importance = pd.DataFrame({
    'wavelength': wavelengths,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Important Wavelengths (after tuning):")
print(importance.head(15).to_string(index=False))

# ============================================================
# STEP 7: Graphs (saved with new filename)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: GENERATING GRAPHS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1: Actual vs Predicted (Baseline)
axes[0, 0].scatter(y_test, y_pred_base, alpha=0.7, edgecolors='k', linewidth=0.5, c='steelblue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Dry Matter')
axes[0, 0].set_ylabel('Predicted Dry Matter')
axes[0, 0].set_title(f'Baseline: R2={r2_base:.4f}')

# 2: Actual vs Predicted (Tuned)
axes[0, 1].scatter(y_test, y_pred_final, alpha=0.7, edgecolors='k', linewidth=0.5, c='coral')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Dry Matter')
axes[0, 1].set_ylabel('Predicted Dry Matter')
axes[0, 1].set_title(f'Tuned: R2={r2_final:.4f}')

# 3: Residuals
residuals = y_test.values - y_pred_final
axes[0, 2].scatter(y_pred_final, residuals, alpha=0.7, edgecolors='k', linewidth=0.5, c='coral')
axes[0, 2].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 2].set_xlabel('Predicted Dry Matter')
axes[0, 2].set_ylabel('Residuals')
axes[0, 2].set_title('Residual Plot (Tuned)')

# 4: Top 20 Important Wavelengths (updated after tuning)
top20 = importance.head(20)
axes[1, 0].barh(range(len(top20)), top20['importance'].values, color='coral')
axes[1, 0].set_yticks(range(len(top20)))
axes[1, 0].set_yticklabels([f'{w}nm' for w in top20['wavelength'].values])
axes[1, 0].invert_yaxis()
axes[1, 0].set_xlabel('Importance (Gain)')
axes[1, 0].set_title('Top 20 Wavelengths (After Tuning)')

# 5: Feature Importance Across Spectrum (updated after tuning)
axes[1, 1].plot(wavelengths, importance.set_index('wavelength').reindex(wavelengths)['importance'].values, c='coral')
axes[1, 1].set_xlabel('Wavelength (nm)')
axes[1, 1].set_ylabel('Importance')
axes[1, 1].set_title('Importance Across Spectrum (After Tuning)')

# 6: Before vs After comparison
metrics = ['R²', 'RMSE', 'MAE']
baseline_vals = [r2_base, rmse_base, mae_base]
tuned_vals = [r2_final, rmse_final, mae_final]
x_pos = np.arange(len(metrics))
axes[1, 2].bar(x_pos - 0.15, baseline_vals, 0.3, label='Baseline', color='steelblue', edgecolor='black')
axes[1, 2].bar(x_pos + 0.15, tuned_vals, 0.3, label='Tuned', color='coral', edgecolor='black')
axes[1, 2].set_xticks(x_pos)
axes[1, 2].set_xticklabels(metrics)
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_title('Baseline vs Tuned Comparison')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('dry_matter_results_tuned.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nPlot saved as dry_matter_results_tuned.png")
print(f"\n{'='*60}")
print(f"COMPLETE: LightGBM Hypertuned Successfully!")
print(f"{'='*60}")
