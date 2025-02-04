import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sklearn.model_selection as sklearn

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
        self.base_features = ["S0", "m", "r", "T", "corp", "alpha", "beta", "omega", "gamma", "lambda"]
        self.target_scaler = scaler
        if is_train:
            self.target_scaler.fit(self.data[["V"]])

        # Precompute constant features for faster access later
        self.epsilon = 1e-6  # To avoid division by zero in calculations
        self.data["strike"] = self.data["S0"] * self.data["m"]
        # self.data["time_value"] = self.data["r"] * self.data["T"]

        # # Correct volatility measure using standard deviation of returns
        # self.data["returns"] = self.data["S0"].pct_change()  # Simple return calculation (percentage change)

        # # Convert to numpy ndarray and then to tensor
        # rolling_std = self.data["returns"].rolling(window=30).std().values  # numpy ndarray
        # annualized_volatility = torch.tensor(rolling_std) * torch.sqrt(torch.tensor(252.0))  # Annualized volatility (assuming 252 trading days)
        # self.data["volatility_measure"] = annualized_volatility

        # # Apply epsilon to avoid division by zero
        # self.data["volatility_measure"] = self.data["volatility_measure"].fillna(self.epsilon)

        # Convert Series to Tensor for operations like log and sqrt
        self.data["log_gamma"] = torch.log(torch.tensor(self.data["gamma"].values) + self.epsilon)
        self.data["sqrt_omega"] = torch.sqrt(torch.tensor(self.data["omega"].values) + self.epsilon)
        self.data["inv_T"] = 1 / (torch.tensor(self.data["T"].values) + self.epsilon)
        self.data["alpha_beta"] = torch.tensor(self.data["alpha"].values) * torch.tensor(self.data["beta"].values)
        # self.data["risk_adjusted"] = (torch.tensor(self.data["corp"].values) * torch.tensor(self.data["omega"].values)) / (torch.tensor(self.data["gamma"].values) + self.epsilon)
        self.data["time_decay"] = torch.exp(-0.05 * torch.tensor(self.data["T"].values))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Directly access precomputed features for this sample
        row = self.data.iloc[idx]
        base_features = torch.tensor(row[self.base_features].values, dtype=torch.float32)

        # Extract precomputed engineered features
        engineered_features = torch.tensor([
            row["strike"],
            # row["time_value"],
            # row["volatility_measure"],  # Using the corrected volatility measure
            row["log_gamma"],
            row["sqrt_omega"],
            row["inv_T"],
            row["alpha_beta"],
            # row["risk_adjusted"],
            row["time_decay"]
        ], dtype=torch.float32)

        # Concatenate base features with engineered features
        X = torch.cat([base_features, engineered_features])

        # Scale target variable
        target_value = row["V"]
        target_df = pd.DataFrame([[target_value]], columns=["V"])
        scaled_target = self.target_scaler.transform(target_df).flatten()
        Y = torch.tensor(scaled_target, dtype=torch.float32)

        return X, Y


def train_test_split(data, test_size=0.3, random_state=42):
    """
    Split the dataset into training and validation sets.

    Args:
        data (pd.DataFrame): Input DataFrame containing the dataset
        test_size (float): Proportion of the dataset to include in the validation split (0 to 1)
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (train_dataset, val_dataset) containing OptionDataset objects
    """
    # Split the data using sklearn's train_test_split
    train_data, val_data = sklearn.train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # Create OptionDataset objects
    train_dataset = OptionDataset(train_data, is_train=True)
    val_dataset = OptionDataset(val_data, is_train=False,
                               target_scaler=train_dataset.target_scaler)

    return train_dataset, val_dataset


def dataset_file(filename):
    return pd.read_csv(filename)

def cleandataset(data):
    return data[data['V'] > 0.5].reset_index(drop=True)

# Load and prepare datasets
# data_train = cleandataset(dataset_file('train_dataset.csv'))
# data_test = cleandataset(dataset_file('test_dataset.csv'))

# dataset_train = OptionDataset(data_train, is_train=True)
# dataset_test = OptionDataset(data_test, is_train=False,
#                            target_scaler=dataset_train.target_scaler)

dataset_train, dataset_test = train_test_split(cleandataset(dataset_file('stage3.csv')))

# # Extract raw option prices (not scaled) for CORAL loss
# source_prices = torch.tensor(data_train["V"].values, dtype=torch.float32).view(-1, 1)
# target_prices = torch.tensor(data_test["V"].values, dtype=torch.float32).view(-1, 1)

# coral_loss = coral(source_prices, target_prices)
# print("CORAL Loss (on option prices):", coral_loss.item())
