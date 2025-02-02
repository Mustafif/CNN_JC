import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def coral(source, target):
    """
    Perform CORAL loss computation between source and target feature distributions.
    """
    d = source.size(1)
    source_covar = torch.mm((source - source.mean(0)).T, (source - source.mean(0))) / (len(source) - 1)
    target_covar = torch.mm((target - target.mean(0)).T, (target - target.mean(0))) / (len(target) - 1)
    loss = torch.norm(source_covar - target_covar, p='fro') / (4 * d * d)
    return loss

scaler = MinMaxScaler(feature_range=(0, 1))

class OptionDataset(Dataset):
    def __init__(self, dataframe, is_train=False, target_scaler=None):
        self.data = dataframe
        self.is_train = is_train
        self.base_features = ["S0", "m", "r", "T", "corp",
                            "alpha", "beta", "omega", "gamma", "lambda"]
        self.target_scaler = scaler
        self.target_scaler.fit(self.data[["V"]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        base_features = torch.tensor(
            self.data[self.base_features].iloc[idx].values,
            dtype=torch.float32
        )

        S0, m, r, T, corp, alpha, beta, omega, gamma, lambda_ = base_features

        # Core financial ratios
        moneyness = S0 / (m + 1e-6)
        time_value = r * T
        volatility_measure = torch.sqrt(gamma * lambda_ + 1e-6)

        # Non-linear transformations
        log_gamma = torch.log(gamma + 1e-6)
        sqrt_omega = torch.sqrt(omega + 1e-6)
        inv_T = 1 / (T + 1e-6)

        # Interaction terms
        alpha_beta = alpha * beta
        risk_adjusted = (corp * omega) / (gamma + 1e-6)

        # Time-based features
        time_decay = torch.exp(-0.05 * T)

        # Combine all engineered features
        # engineered_features = torch.stack([
        #     moneyness,
        #     time_value,
        #     volatility_measure,
        #     log_gamma,
        #     sqrt_omega,
        #     inv_T,
        #     alpha_beta,
        #     risk_adjusted,
        #     time_decay
        # ])
        # engineered_features = []
        # X = torch.cat([base_features, engineered_features])
        X = base_features
        target_value = self.data["V"].iloc[idx]
        target_df = pd.DataFrame([[target_value]], columns=["V"])
        scaled_target = self.target_scaler.transform(target_df).flatten()
        Y = torch.tensor(scaled_target, dtype=torch.float32)

        return X, Y

def dataset_file(filename):
    return pd.read_csv(filename)

def cleandataset(data):
    return data[data['V'] > 1].reset_index(drop=True)

# Load and prepare datasets
data_train = cleandataset(dataset_file('train_dataset.csv'))
data_test = cleandataset(dataset_file('../data_gen/test_dataset.csv'))
dataset_train = OptionDataset(data_train, is_train=True)
dataset_test = OptionDataset(data_test, is_train=False,
                           target_scaler=dataset_train.target_scaler)

# # Extract raw option prices (not scaled) for CORAL loss
# source_prices = torch.tensor(data_train["V"].values, dtype=torch.float32).view(-1, 1)
# target_prices = torch.tensor(data_test["V"].values, dtype=torch.float32).view(-1, 1)

# coral_loss = coral(source_prices, target_prices)
# print("CORAL Loss (on option prices):", coral_loss.item())
