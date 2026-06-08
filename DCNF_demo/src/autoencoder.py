"""
Because the original borehole data are confidential, we use synthetic data for demonstration.
Deep Autoencoder with Residual Blocks and Dropout
This script trains the autoencoder on SYNTHETIC logging parameters (GR, RT, DEN)
simulated from the existing synthetic borehole data (x, y, layer).
No real data is used.
"""



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Residual Block (fixed inplace operation)
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = out + identity          # FIXED: do not use inplace addition
        return out

# ---------------------------
# Deep Autoencoder (Encoder + Decoder)
# ---------------------------
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=3):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(input_dim, 16, dropout_rate=0.1),
            ResidualBlock(16, 8, dropout_rate=0.2),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            ResidualBlock(latent_dim, 8, dropout_rate=0.2),
            ResidualBlock(8, 16, dropout_rate=0.1),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

# ---------------------------
# Simulate logging parameters (GR, RT, DEN) from synthetic stratigraphic data
# ---------------------------
def simulate_logging_parameters(df):
    """
    Simulate GR, RT, DEN based on layer (shallow, middle, deep) and spatial coordinates.
    Returns a numpy array of shape (n_samples, 3)
    """
    gr = []
    rt = []
    den = []
    for idx, row in df.iterrows():
        layer = int(row['layer'])
        x = row['x']
        y = row['y']
        # Shallow assemblage (layers 0-3): high GR, low RT, high DEN
        if layer < 4:
            gr_val = 70 + 10 * np.sin(x/1000) + 5 * np.cos(y/800) + np.random.normal(0, 3)
            rt_val = 30 + 10 * np.sin(x/500) + np.random.normal(0, 2)
            den_val = 2.2 + 0.1 * np.cos(x/600) + np.random.normal(0, 0.05)
        # Middle transition (layers 4-6): medium values
        elif layer < 7:
            gr_val = 60 + 8 * np.sin(x/1200) + np.random.normal(0, 4)
            rt_val = 55 + 15 * np.cos(y/700) + np.random.normal(0, 3)
            den_val = 2.1 + 0.05 * np.sin(x/800) + np.random.normal(0, 0.05)
        # Deep coal-bearing (layers 7-9): low GR, high RT, low DEN
        else:
            gr_val = 45 + 5 * np.cos(y/900) + np.random.normal(0, 3)
            rt_val = 100 + 20 * np.sin(x/400) + np.random.normal(0, 5)
            den_val = 1.9 + 0.05 * np.cos(x/1000) + np.random.normal(0, 0.05)
        gr.append(gr_val)
        rt.append(rt_val)
        den.append(den_val)
    return np.column_stack([gr, rt, den])

# ---------------------------
# Main: Load synthetic data, simulate logging params, train autoencoder
# ---------------------------
if __name__ == "__main__":
    # Locate synthetic data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'synthetic_borehole_data.csv')

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run data/generate_synthetic_data.py first.")
        exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded synthetic data: {len(df)} samples")

    # Simulate logging parameters
    X = simulate_logging_parameters(df)
    print(f"Simulated logging parameters shape: {X.shape} (GR, RT, DEN)")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to torch tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Split into train/val
    n_train = int(0.8 * len(X_tensor))
    train_data = X_tensor[:n_train]
    val_data = X_tensor[n_train:]

    # Model, optimizer, loss
    model = DeepAutoencoder(input_dim=3, latent_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    epochs = 100
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon, _ = model(train_data)
        loss = criterion(recon, train_data)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_recon, _ = model(val_data)
            val_loss = criterion(val_recon, val_data)
            val_losses.append(val_loss.item())

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

    # Plot loss curve
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training on Synthetic Logging Data')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(project_root, 'autoencoder_training.png'), dpi=300)
    plt.close()  # 替代 plt.show()，避免 PyCharm 后端错误
    print("Training completed. Loss curve saved as autoencoder_training.png")