import pandas as pd
import torch
from torch.utils.data import Dataset

class OptionDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Extract features (X) and target (Y)
        X = torch.tensor([
            self.data["S0"][idx],
            self.data["m"][idx],
            self.data["r"][idx],
            self.data["T"][idx],
            self.data["corp"][idx],
            self.data["alpha"][idx],
            self.data["beta"][idx],
            self.data["omega"][idx],
            self.data["gamma"][idx],
            self.data["lambda"][idx]
        ], dtype=torch.float32)
        Y = torch.tensor(self.data["V"][idx], dtype=torch.float32)

        return X, Y

def dataset_file(filename):
    return pd.read_csv(filename)

def cleandataset(data):
    # Drops rows where "V" is < 0.5
    return data[data['V'] > 0.5].reset_index(drop=True)

data_train = cleandataset(dataset_file('train_dataset.csv'))
data_test = cleandataset(dataset_file('test_dataset.csv'))
dataset_train = OptionDataset(data_train)
dataset_test = OptionDataset(data_test)
