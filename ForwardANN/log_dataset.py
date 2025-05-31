import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection as sklearn
scaler = MinMaxScaler(feature_range=(0, 1))
import numpy as np

class OptionDataset(Dataset):
    def __init__(self, dataframe, is_train=False, target_scaler=None, log_target=False):
        self.data = dataframe
        self.is_train = is_train
        self.log_target = log_target
        self.base_features = ["S0", "m", "r", "T", "corp",
                              "alpha", "beta", "omega", "gamma", "lambda", "V"]

        # Precompute useful transformations
        self.epsilon = 1e-6
        self.data["strike"] = self.data["S0"] * self.data["m"]
        self.data["log_gamma"] = torch.log(torch.tensor(self.data["gamma"].values) + self.epsilon)
        self.data["log_m"] = torch.log(torch.tensor(self.data["m"].values))
        self.data["sqrt_omega"] = torch.sqrt(torch.tensor(self.data["omega"].values) + self.epsilon)
        self.data["inv_T"] = 1 / (torch.tensor(self.data["T"].values) + self.epsilon)
        self.data["alpha_beta"] = torch.tensor(self.data["alpha"].values) * torch.tensor(self.data["beta"].values)
        self.data["time_decay"] = torch.exp(-0.05 * torch.tensor(self.data["T"].values))

        # Transform target
        self.target_raw = self.data["impl"].values
        if self.log_target:
            self.target_raw = np.log(np.clip(self.target_raw, a_min=self.epsilon, a_max=None))

        # Fit or use scaler
        if target_scaler is not None:
            self.target_scaler = target_scaler
        else:
            self.target_scaler = MinMaxScaler()
            self.target_scaler.fit(self.target_raw.reshape(-1, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        base_features = torch.tensor(row[self.base_features].values, dtype=torch.float32)

        engineered_features = torch.tensor([
            row["strike"],
            row["log_gamma"],
            row["sqrt_omega"],
            row["inv_T"],
            row["alpha_beta"],
            row["time_decay"],
        ], dtype=torch.float32)

        X = torch.cat([base_features, engineered_features])

        target_value = self.target_raw[idx]
        scaled_target = self.target_scaler.transform([[target_value]]).flatten()
        Y = torch.tensor(scaled_target, dtype=torch.float32)

        return X, Y
