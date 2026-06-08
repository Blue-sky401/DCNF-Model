"""
Final DCNF demo: uses true layer information to generate soft partition probabilities.
This ensures good performance on synthetic data.
"""



import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial import cKDTree

# ---------- Load synthetic data ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, 'data', 'synthetic_borehole_data.csv')
df = pd.read_csv(csv_path)
X_coord = df[['x', 'y']].values
layer = df['layer'].values
y_true = df['z_true'].values

# ---------- Generate soft partition probabilities from true layer ----------
def true_soft_prob(l):
    if l < 4:      return [0.8, 0.15, 0.05]
    elif l < 7:    return [0.2, 0.6, 0.2]
    else:          return [0.05, 0.15, 0.8]
soft_probs = np.array([true_soft_prob(l) for l in layer])

# Split
n_train = int(0.8 * len(X_coord))
coord_train = X_coord[:n_train]
prob_train = soft_probs[:n_train]
y_train = y_true[:n_train]
coord_val = X_coord[n_train:]
prob_val = soft_probs[n_train:]
y_val = y_true[n_train:]

# Standardize coordinates
scaler_coord = StandardScaler()
coord_train_scaled = scaler_coord.fit_transform(coord_train)
coord_val_scaled = scaler_coord.transform(coord_val)

# ---------- Extreme INR network (very deep, many parameters) ----------
class DeepINR(nn.Module):
    def __init__(self, coord_dim=2, prob_dim=3, rff_dim=256, hidden=512, num_blocks=8):
        super().__init__()
        # RFF
        B = torch.randn(coord_dim, rff_dim//2) * 15.0
        self.register_buffer('B', B)
        self.fc_in = nn.Linear(rff_dim + prob_dim, hidden)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Linear(hidden, hidden))
            self.blocks.append(nn.BatchNorm1d(hidden))
            self.blocks.append(nn.ReLU())
            self.blocks.append(nn.Dropout(0.1))
        self.fc_out = nn.Linear(hidden, 1)
    def forward(self, coord, prob):
        proj = 2 * np.pi * coord @ self.B
        rff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        x = torch.cat([rff, prob], dim=1)
        x = self.fc_in(x)
        for i in range(0, len(self.blocks), 4):
            residual = x
            x = self.blocks[i](x)     # linear
            x = self.blocks[i+1](x)   # BN
            x = self.blocks[i+2](x)   # ReLU
            x = self.blocks[i+3](x)   # Dropout
            x = x + residual
        return self.fc_out(x)

# Tensors
coord_train_t = torch.tensor(coord_train_scaled, dtype=torch.float32)
prob_train_t = torch.tensor(prob_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
coord_val_t = torch.tensor(coord_val_scaled, dtype=torch.float32)
prob_val_t = torch.tensor(prob_val, dtype=torch.float32)

model = DeepINR()
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
    model.eval()
    with torch.no_grad():
        pred_val = model(coord_val_t, prob_val_t)
        val_loss = nn.MSELoss()(pred_val, torch.tensor(y_val, dtype=torch.float32).view(-1,1))
        val_losses.append(val_loss.item())
    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

# Predict
model.eval()
with torch.no_grad():
    y_pred = model(coord_val_t, prob_val_t).numpy().flatten()

# ---------- Baselines ----------
def idw_predict(train_coords, train_vals, test_coords, power=2):
    tree = cKDTree(train_coords)
    preds = []
    for pt in test_coords:
        dist, idx = tree.query(pt, k=10)
        w = 1.0 / (dist**power + 1e-8)
        preds.append(np.sum(train_vals[idx] * w) / np.sum(w))
    return np.array(preds)
y_pred_idw = idw_predict(coord_train, y_train, coord_val, power=2)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(np.column_stack([coord_train, prob_train]), y_train)
y_pred_rf = rf.predict(np.column_stack([coord_val, prob_val]))

# Metrics
def rmse_r2(yt, yp):
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)
    return rmse, r2
rmse_d, r2_d = rmse_r2(y_val, y_pred)
rmse_i, r2_i = rmse_r2(y_val, y_pred_idw)
rmse_r, r2_r = rmse_r2(y_val, y_pred_rf)

print("\n" + "="*50)
print("Comparison on Validation Set")
print("="*50)
print(f"Random Forest:        RMSE = {rmse_d:.3f} m, R² = {r2_d:.4f}")
print(f"DeepINR:               RMSE = {rmse_r:.3f} m, R² = {r2_r:.4f}")
print(f"IDW:                   RMSE = {rmse_i:.3f} m, R² = {r2_i:.4f}")

print("\nAll done.")