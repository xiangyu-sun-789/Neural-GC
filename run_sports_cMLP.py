import torch
import numpy as np
import matplotlib.pyplot as plt
from synthetic import simulate_var
from models.cmlp import cMLP, cMLPSparse, train_model_ista, train_unregularized

# For GPU acceleration
# device = torch.device('cuda')
device = torch.device('cpu')

# Simulate data
# X_np, beta, GC = simulate_var(p=10, T=1000, lag=3)
X_np, _, _ = simulate_var(p=10, T=1000, lag=3)

assert X_np.shape == (1000, 10)

X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)

# assert X shape: (1, number of time steps, number of variables)
assert X.shape == (1, 1000, 10)

number_of_lags = 3

# Set up model
cmlp = cMLP(X.shape[-1], lag=number_of_lags, hidden=[100])

# Train with ISTA
train_loss_list = train_model_ista(cmlp, X, lam=0.002, lam_ridge=1e-2, lr=5e-2, penalty='H',
                                   max_iter=50000,
                                   check_every=100)

# (p x p x lag) matrix. Entry (i, j, k) indicates whether variable j is Granger causal of variable i at lag k.
# column_{t-k} -> row_t
GC_est = cmlp.GC(ignore_lag=False).cpu().data.numpy()
print(GC_est)
print(GC_est.shape)
