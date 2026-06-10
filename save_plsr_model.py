import numpy as np
import joblib
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('apple1.csv')
wavelength_cols = [c for c in df.columns if c not in ['Apple', 'DryMatter']]
X = df[wavelength_cols].values
y = df['DryMatter'].values

# Preprocess
def snv(s): return (s - s.mean()) / (s.std() + 1e-10)
X_snv = np.apply_along_axis(snv, 1, X)

# Best PLSR: Derivative features, n=19
X_d1 = np.gradient(X, axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_d1)

pls = PLSRegression(n_components=19)
pls.fit(X_scaled, y)

# Save
joblib.dump(pls, 'pls_model.pkl')
joblib.dump(scaler, 'pls_scaler.pkl')
joblib.dump(wavelength_cols, 'wavelength_columns.pkl')

print("Saved: pls_model.pkl")
print("Saved: pls_scaler.pkl")
print("Saved: wavelength_columns.pkl")
print(f"\nModel: PLSRegression(n_components=19)")
print(f"Features: 141 derivative wavelengths (430-990nm)")
print(f"R²=0.2708, RMSE=0.00639")

# Test
yp = pls.predict(X_scaled).ravel()
print(f"\nSample predictions (first 5):")
for i in range(5):
    print(f"  Actual: {y[i]:.4f}  Predicted: {yp[i]:.4f}  Error: {abs(y[i]-yp[i]):.4f}")
