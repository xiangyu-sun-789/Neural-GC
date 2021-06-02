import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from synthetic import simulate_var
from models.cmlp import cMLP, cMLPSparse, train_model_ista, train_unregularized
from utils import save_adjacency_matrix_in_csv, draw_DAGs_using_LINGAM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_sports_data(number_of_lags):
    data_file = "/Users/shawnxys/Development/Data/preprocessed_causal_sports_data_by_games/17071/features_shots_rewards.csv"

    features_shots_rewards_df = pd.read_csv(data_file)
    # rename column name
    features_shots_rewards_df = features_shots_rewards_df.rename(columns={'reward': 'goal'})

    X = features_shots_rewards_df.to_numpy()  # (number of time steps, number of variables)

    # data standardization
    scaler = preprocessing.StandardScaler().fit(X)
    normalized_X = scaler.transform(X)  # (number of time steps, number of variables)

    print('feature std after standardization: ', normalized_X.std(axis=0))
    assert (normalized_X.std(axis=0).round(
        decimals=3) == 1).all()  # make sure all the variances are (very close to) 1

    T, N = normalized_X.shape  # (number of time steps, number of variables)

    assert T == 4021 and N == 12

    # ..., t-2, t-1, t
    variable_names = []
    for k in range(number_of_lags, 0, -1):
        variable_names += [s + "_t-{}".format(k) for s in features_shots_rewards_df.columns]

    variable_names += [s + "_t" for s in features_shots_rewards_df.columns]

    return normalized_X, variable_names


number_of_lags = 1

normalized_X, variable_names = load_sports_data(number_of_lags)
print(variable_names)

assert normalized_X.shape == (4021, 12)

d = normalized_X.shape[1]

X = torch.tensor(normalized_X[np.newaxis], dtype=torch.float32, device=device)

# assert X shape: (1, number of time steps, number of variables)
assert X.shape == (1, 4021, 12)

# Set up model
cmlp = cMLP(X.shape[-1], lag=number_of_lags, hidden=[100])

# Train with ISTA
train_loss_list = train_model_ista(cmlp, X, lam=0.002, lam_ridge=1e-2, lr=5e-2, penalty='H',
                                   max_iter=50000,
                                   check_every=100)

# (p x p x lag) matrix: Entry (i, j, k) indicates whether variable j is Granger causal of variable i at lag k.
# column_{t-k} -> row_t
# t-1, t-2, ...
GC_est = cmlp.GC(ignore_lag=False).cpu().data.numpy()
print(GC_est)
print(GC_est.shape)

# ..., t-2, t-1, t
W_est_full = np.zeros((len(variable_names), len(variable_names)))

for k in range(number_of_lags, 0, -1):
    current_lag_W = GC_est[:, :, k - 1]

    # column_{t-k} -> row_t
    W_est_full[-1 * d:, -1 * d - k * d:0 - k * d] = current_lag_W

print(W_est_full)
print(W_est_full.shape)

file_name = './estimated_DAG'
save_adjacency_matrix_in_csv(file_name, W_est_full, variable_names)
draw_DAGs_using_LINGAM(file_name, W_est_full, variable_names)
