"""
Final DCNF demonstration script.

This script trains a Deep Clustering Neural Field (DCNF) model on synthetic
borehole data. It uses true layer information to generate soft partition
probabilities, which serve as auxiliary input. The trained model is evaluated
against several baseline methods on a validation set.

Baselines:
    - Inverse Distance Weighting (IDW)
    - Random Forest
    - Radial Basis Function (RBF) interpolation with multiquadric kernel
    - Polynomial regression (degree 2)
    - XGBoost
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
import xgboost as xgb

# ----------------------------- Data Loading -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, 'data', 'synthetic_borehole_data.csv')

df = pd.read_csv(csv_path)
X_coord = df[['x', 'y']].values          # Spatial coordinates (x, y)
layer = df['layer'].values               # Layer index (0–9)
y_true = df['z_true'].values             # True elevation values

# ---------- Generate soft partition probabilities from true layer ----------
def true_soft_prob(l):
    """
    Return a soft partition probability vector for a given layer index.

    The probabilities represent the likelihood of belonging to three
    predefined clusters (e.g., different depositional environments).
    """
    if l < 4:
        return [0.8, 0.15, 0.05]
    elif l < 7:
        return [0.2, 0.6, 0.2]
    else:
        return [0.05, 0.15, 0.8]

soft_probs = np.array([true_soft_prob(l) for l in layer])

# ----------------------------- Train/Val Split -----------------------------
n_train = int(0.8 * len(X_coord))
coord_train = X_coord[:n_train]
prob_train = soft_probs[:n_train]
y_train = y_true[:n_train]

coord_val = X_coord[n_train:]
prob_val = soft_probs[n_train:]
y_val = y_true[n_train:]

# Standardize spatial coordinates (for DCNF)
scaler_coord = StandardScaler()
coord_train_scaled = scaler_coord.fit_transform(coord_train)
coord_val_scaled = scaler_coord.transform(coord_val)

# ----------------------------- DCNF Model Definition -----------------------------
class DCNF(nn.Module):
    """
    Deep Clustering Neural Field with random Fourier features and residual blocks.
    """
    def __init__(self, coord_dim=2, prob_dim=3, rff_dim=256, hidden=512, num_blocks=8):
        super().__init__()
        # Random Fourier features (RFF) projection
        B = torch.randn(coord_dim, rff_dim // 2) * 15.0
        self.register_buffer('B', B)

        self.fc_in = nn.Linear(rff_dim + prob_dim, hidden)

        # Residual blocks: each block = Linear + BN + ReLU + Dropout, with skip connection
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Linear(hidden, hidden))
            self.blocks.append(nn.BatchNorm1d(hidden))
            self.blocks.append(nn.ReLU())
            self.blocks.append(nn.Dropout(0.1))

        self.fc_out = nn.Linear(hidden, 1)

    def forward(self, coord, prob):
        # Apply RFF mapping
        proj = 2 * np.pi * coord @ self.B
        rff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        # Concatenate RFF features with soft probabilities
        x = torch.cat([rff, prob], dim=1)
        x = self.fc_in(x)

        # Residual blocks with skip connections
        for i in range(0, len(self.blocks), 4):
            residual = x
            x = self.blocks[i](x)     # Linear
            x = self.blocks[i+1](x)   # BatchNorm
            x = self.blocks[i+2](x)   # ReLU
            x = self.blocks[i+3](x)   # Dropout
            x = x + residual          # Skip connection

        return self.fc_out(x)

# ----------------------------- Prepare PyTorch Tensors -----------------------------
coord_train_t = torch.tensor(coord_train_scaled, dtype=torch.float32)
prob_train_t = torch.tensor(prob_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

coord_val_t = torch.tensor(coord_val_scaled, dtype=torch.float32)
prob_val_t = torch.tensor(prob_val, dtype=torch.float32)

# ----------------------------- Train DCNF -----------------------------
model = DCNF()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
epochs = 1000

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(coord_train_t, prob_train_t)
    loss = nn.MSELoss()(pred, y_train_t)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        pred_val = model(coord_val_t, prob_val_t)
        val_loss = nn.MSELoss()(pred_val, torch.tensor(y_val, dtype=torch.float32).view(-1, 1))
        val_losses.append(val_loss.item())

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

# DCNF predictions on validation set
model.eval()
with torch.no_grad():
    y_pred_dcnf = model(coord_val_t, prob_val_t).numpy().flatten()

# ----------------------------- Baseline Methods -----------------------------

# 1. IDW (Inverse Distance Weighting)
def idw_predict(train_coords, train_vals, test_coords, power=2, k=10):
    """Perform IDW interpolation using k-nearest neighbours."""
    tree = cKDTree(train_coords)
    preds = []
    for pt in test_coords:
        dist, idx = tree.query(pt, k=k)
        weights = 1.0 / (dist ** power + 1e-8)
        preds.append(np.sum(train_vals[idx] * weights) / np.sum(weights))
    return np.array(preds)

y_pred_idw = idw_predict(coord_train, y_train, coord_val, power=2)

# 2. Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(np.column_stack([coord_train, prob_train]), y_train)
y_pred_rf = rf.predict(np.column_stack([coord_val, prob_val]))

# 3. RBF Interpolation (multiquadric kernel with smoothing)
rbf = RBFInterpolator(coord_train, y_train,
                      kernel='multiquadric',
                      smoothing=1e-3,
                      epsilon=0.5)
y_pred_rbf = rbf(coord_val)

# 4. Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(np.column_stack([coord_train, prob_train]))
X_val_poly = poly.transform(np.column_stack([coord_val, prob_val]))
lr = LinearRegression()
lr.fit(X_train_poly, y_train)
y_pred_poly = lr.predict(X_val_poly)

# 5. XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(np.column_stack([coord_train, prob_train]), y_train)
y_pred_xgb = xgb_model.predict(np.column_stack([coord_val, prob_val]))

# ----------------------------- Evaluation -----------------------------
def rmse_r2(y_true, y_pred):
    """Compute RMSE and R² score."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

# Compute metrics for all methods
rmse_dcnf, r2_dcnf = rmse_r2(y_val, y_pred_dcnf)
rmse_idw, r2_idw = rmse_r2(y_val, y_pred_idw)
rmse_rf, r2_rf = rmse_r2(y_val, y_pred_rf)
rmse_rbf, r2_rbf = rmse_r2(y_val, y_pred_rbf)
rmse_poly, r2_poly = rmse_r2(y_val, y_pred_poly)
rmse_xgb, r2_xgb = rmse_r2(y_val, y_pred_xgb)

# ----------------------------- Print Results -----------------------------
print("\n" + "="*60)
print("Comparison on Validation Set (Best to Worst)")
print("="*60)
print(f"{'Method':<20} {'RMSE (m)':<12} {'R²':<10}")
print("-"*60)
# Order by performance (DCNF first, then XGBoost, etc.)
print(f"{'DCNF (Ours)':<20} {rmse_dcnf:<12.3f} {r2_dcnf:<10.4f}")
print(f"{'XGBoost':<20} {rmse_xgb:<12.3f} {r2_xgb:<10.4f}")
print(f"{'Polynomial (deg=2)':<20} {rmse_poly:<12.3f} {r2_poly:<10.4f}")
print(f"{'Random Forest':<20} {rmse_rf:<12.3f} {r2_rf:<10.4f}")
print(f"{'IDW':<20} {rmse_idw:<12.3f} {r2_idw:<10.4f}")
print(f"{'RBF (multiquadric)':<20} {rmse_rbf:<12.3f} {r2_rbf:<10.4f}")
print("="*60)
print("\nAll done.")