import torch
import torch.nn as nn

class CaNNModel(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(CaNNModel, self).__init__()
        input_features = 10
        neurons = 100
        self.input_layer = nn.Linear(input_features, neurons)
        self.hl1 = nn.Linear(neurons, neurons)
        self.hl2 = nn.Linear(neurons, neurons)
        self.hl3 = nn.Linear(neurons, neurons)
        self.hl4 = nn.Linear(neurons, neurons)
        self.hl5 = nn.Linear(neurons, neurons)
        self.output_layer = nn.Linear(neurons, 1) # just need 1 price

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.dropout5 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hl1(x))
        x = self.dropout1(x)
        x = torch.relu(self.hl2(x))
        x = self.dropout2(x)
        x = torch.relu(self.hl3(x))
        x = self.dropout3(x)
        x = torch.relu(self.hl4(x))
        x = self.dropout4(x)
        x = torch.relu(self.hl5(x))
        x = self.dropout5(x)
        x = self.output_layer(x)
        return x
