import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import numpy as np

class OptionDataset(Dataset):
    def __init__(self, dataframe, is_train=False, target_scaler=None):
        self.data = dataframe
        self.is_train = is_train
        self.feature_columns = ["S0", "m", "r", "T", "corp", "alpha", "beta", "omega", "gamma", "lambda"]

        # Initialize or use existing target scaler
        if is_train:
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
            # Fit scaler on training data using DataFrame with column name
            self.target_scaler.fit(self.data[["V"]])  # <- Already correct
        else:
            self.target_scaler = target_scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract features (X) - unchanged
        X = torch.tensor(
            self.data[self.feature_columns].iloc[idx].values,
            dtype=torch.float32
        )

        # Fixed target scaling with proper DataFrame input
        target_value = self.data["V"].iloc[idx]
        # Create DataFrame with column name to match training format
        target_df = pd.DataFrame([[target_value]], columns=["V"])
        scaled_target = self.target_scaler.transform(target_df).flatten()
        Y = torch.tensor(scaled_target, dtype=torch.float32)

        return X, Y

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
