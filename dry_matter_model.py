import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
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

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
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

model = lgb.train(
    params, train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"TEST SET PERFORMANCE")
print(f"{'='*50}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"R²   : {r2:.6f}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lgb.LGBMRegressor(**params, n_estimators=500), X, y, cv=kf, scoring='r2')
print(f"\n5-Fold CV R²: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")

importance = pd.DataFrame({
    'wavelength': wavelengths,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print(f"\nTop 15 most important wavelengths:")
print(importance.head(15).to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(y_test, y_pred, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Dry Matter')
axes[0, 0].set_ylabel('Predicted Dry Matter')
axes[0, 0].set_title(f'Actual vs Predicted (R²={r2:.4f})')

residuals = y_test.values - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Dry Matter')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')

top20 = importance.head(20)
axes[1, 0].barh(range(len(top20)), top20['importance'].values)
axes[1, 0].set_yticks(range(len(top20)))
axes[1, 0].set_yticklabels([f'{w}nm' for w in top20['wavelength'].values])
axes[1, 0].invert_yaxis()
axes[1, 0].set_xlabel('Importance (Gain)')
axes[1, 0].set_title('Top 20 Important Wavelengths')

axes[1, 1].plot(wavelengths, importance.set_index('wavelength').reindex(wavelengths)['importance'].values)
axes[1, 1].set_xlabel('Wavelength (nm)')
axes[1, 1].set_ylabel('Importance')
axes[1, 1].set_title('Feature Importance Across Spectrum')

plt.tight_layout()
plt.savefig('dry_matter_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as dry_matter_results.png")
