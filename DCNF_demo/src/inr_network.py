"""
Implicit Neural Representation (INR) Network with RFF and Residual Connections
This script defines a small INR network for stratigraphic elevation prediction.
It includes Random Fourier Feature (RFF) encoding and a residual MLP.
"""

import torch
import torch.nn as nn
import numpy as np

# ---------------------------
# Random Fourier Feature (RFF) encoding
# ---------------------------
class RFF(nn.Module):
    def __init__(self, input_dim=3, encoding_dim=64, scale=10.0):
        """
        input_dim: dimension of input coordinates (e.g., 3 for x,y,z)
        encoding_dim: output dimension of RFF (should be even)
        scale: bandwidth parameter for random frequencies
        """
        super(RFF, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        # Random projection matrix (frozen)
        B = torch.randn(input_dim, encoding_dim // 2) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        # x shape: (batch, input_dim)
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# ---------------------------
# Residual Block for MLP
# ---------------------------
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super(ResidualMLPBlock, self).__init__()
        self.linear = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out + identity
        return out

# ---------------------------
# INR Network (MLP + RFF + Residuals)
# ---------------------------
class INRNetwork(nn.Module):
    def __init__(self, coord_dim=3, prob_dim=3, rff_dim=64, hidden_dim=128, num_res_blocks=4):
        """
        coord_dim: dimension of spatial coordinates (x,y,z)
        prob_dim: dimension of soft partition probabilities (q1,q2,q3)
        rff_dim: output dimension of RFF encoding
        hidden_dim: hidden layer size
        num_res_blocks: number of residual blocks after RFF
        """
        super(INRNetwork, self).__init__()
        # RFF encoding for coordinates
        self.rff = RFF(input_dim=coord_dim, encoding_dim=rff_dim, scale=10.0)
        # Input dimension after concatenation: rff_dim + prob_dim
        input_dim = rff_dim + prob_dim
        # Initial linear layer
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim) for _ in range(num_res_blocks)])
        # Output layer
        self.output_linear = nn.Linear(hidden_dim, 1)

    def forward(self, x_coord, x_prob):
        """
        x_coord: (batch, 3) spatial coordinates
        x_prob: (batch, 3) soft partition probabilities
        """
        # RFF encoding for coordinates
        feat = self.rff(x_coord)                # (batch, rff_dim)
        # Concatenate with probabilities
        feat = torch.cat([feat, x_prob], dim=-1) # (batch, rff_dim + 3)
        # Initial projection
        feat = self.input_linear(feat)
        feat = self.bn_in(feat)
        feat = self.relu(feat)
        # Residual blocks
        feat = self.res_blocks(feat)
        # Output elevation
        out = self.output_linear(feat)          # (batch, 1)
        return out

# ---------------------------
# Quick test (run this script directly)
# ---------------------------
if __name__ == "__main__":
    # Create random input
    batch_size = 4
    coord = torch.randn(batch_size, 3)
    prob = torch.rand(batch_size, 3)
    prob = prob / prob.sum(dim=1, keepdim=True)  # normalize to sum=1

    model = INRNetwork()
    pred = model(coord, prob)

    print("INR Network test passed!")
    print(f"Input coordinates shape: {coord.shape}")
    print(f"Soft probabilities shape: {prob.shape}")
    print(f"Predicted elevation shape: {pred.shape}")
    print(f"Example prediction:\n{pred}")