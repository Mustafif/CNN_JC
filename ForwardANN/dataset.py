import pandas as pd
import torch
from torch.utils.data import Dataset

class OptionDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        S0 = 0
        K = 1
        r = 2
        T = 3
        corp = 4
        alpha = 5
        beta = 6
        omega = 7
        gamma = 8
        lambda_ = 9
        V = 10
        # Extract features (X) and target (Y)
        X = torch.tensor([
            self.data.iloc[S0][idx],
            self.data.iloc[K][idx],
            self.data.iloc[r][idx],
            self.data.iloc[T][idx],
            self.data.iloc[corp][idx],
            self.data.iloc[alpha][idx],
            self.data.iloc[beta][idx],
            self.data.iloc[omega][idx],
            self.data.iloc[gamma][idx],
            self.data.iloc[lambda_][idx]
        ], dtype=torch.float32)
        Y = torch.tensor(self.data.iloc[V][idx], dtype=torch.float32)

        return X, Y

def dataset_file(filename):
    data = pd.read_csv(filename, header=None)
    result = {row[0]: row[1:].values.tolist() for _, row in data.iterrows()}
    df = pd.DataFrame.from_dict(result, orient='index')
    return df

data_train = dataset_file('train_dataset.csv')
data_test = dataset_file('test_dataset.csv')
dataset_train = OptionDataset(data_train)
dataset_test = OptionDataset(data_test)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
