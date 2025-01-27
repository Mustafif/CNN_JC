import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import numpy as np

class OptionDataset(Dataset):
    def __init__(self, dataframe, is_train=False, target_scaler=None):
        self.data = dataframe
        self.is_train = is_train
        self.base_features = ["S0", "m", "r", "T", "corp",
                            "alpha", "beta", "omega", "gamma", "lambda"]

        # Initialize or use existing target scaler
        if is_train:
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler.fit(self.data[["V"]])
        else:
            self.target_scaler = target_scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract base features as tensor
        base_features = torch.tensor(
            self.data[self.base_features].iloc[idx].values,
            dtype=torch.float32
        )

        # Calculate financial ratios and transformed features
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
        time_decay = torch.exp(-0.05 * T)  # Simplified discount factor

        # Combine all engineered features
        engineered_features = torch.stack([
            moneyness,
            time_value,
            volatility_measure,
            log_gamma,
            sqrt_omega,
            inv_T,
            alpha_beta,
            risk_adjusted,
            time_decay
        ])

        # Full feature vector
        X = torch.cat([base_features, engineered_features])

        # Target processing with DataFrame wrapper
        target_value = self.data["V"].iloc[idx]
        target_df = pd.DataFrame([[target_value]], columns=["V"])
        scaled_target = self.target_scaler.transform(target_df).flatten()
        Y = torch.tensor(scaled_target, dtype=torch.float32)

        return X, Y

# class OptionDataset(Dataset):
#     def __init__(self, dataframe, is_train=False, target_scaler=None):
#         self.data = dataframe
#         self.is_train = is_train
#         self.feature_columns = ["S0", "m", "r", "T", "corp", "alpha", "beta", "omega", "gamma", "lambda"]

#         # Initialize or use existing target scaler
#         if is_train:
#             self.target_scaler = MinMaxScaler(feature_range=(0, 1))
#             # Fit scaler on training data using DataFrame with column name
#             self.target_scaler.fit(self.data[["V"]])  # <- Already correct
#         else:
#             self.target_scaler = target_scaler

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # Extract features (X) - unchanged
#         X = torch.tensor(
#             self.data[self.feature_columns].iloc[idx].values,
#             dtype=torch.float32
#         )

#         # Fixed target scaling with proper DataFrame input
#         target_value = self.data["V"].iloc[idx]
#         # Create DataFrame with column name to match training format
#         target_df = pd.DataFrame([[target_value]], columns=["V"])
#         scaled_target = self.target_scaler.transform(target_df).flatten()
#         Y = torch.tensor(scaled_target, dtype=torch.float32)

#         return X, Y

def dataset_file(filename):
    return pd.read_csv(filename)

def cleandataset(data):
    # Drops rows where "V" is < 1
    return data[data['V'] > 1].reset_index(drop=True)

# Load and prepare datasets
data_train = cleandataset(dataset_file('train_dataset.csv'))
data_test = cleandataset(dataset_file('test_dataset.csv'))
dataset_train = OptionDataset(data_train, is_train=True)
dataset_test = OptionDataset(data_test, is_train=False,
                           target_scaler=dataset_train.target_scaler)
