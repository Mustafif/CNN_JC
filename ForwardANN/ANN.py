import historical
from optdata import OptionData
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
import os

class ParamFeatures:
    def __init__(self, day_num,opt_file=None, alpha=1.33e-6, beta=0.8, omega=1e-6, gamma=100, lambda_=0.5, r=0.03, corp=1):
        # historical asset parameters
        self.historical = historical.historical()
        # market parameters
        self.r: float = 0.03
        self.S0: float = self.historical[-5+(day_num-1)]
        self.T = np.array([5, 10, 21, 42, 63, 126]) - (day_num - 1) # one number :)
        self.K= np.linspace(0.8, 1.2, 9)*self.S0 # one number :)
        self.corp: int = corp
        # GARCH parameters
        self.alpha: float = alpha
        self.beta: float = beta
        self.omega: float = omega
        self.gamma: float = gamma
        self.lambda_: float = lambda_

        # Option Prices given maturity and Day Number
        opt_data = OptionData(day_num=day_num, opt_file=opt_file)
        self.call = opt_data.call()
        self.put = opt_data.put()

    def get_features(self):
       market_params = tensor(np.array([self.r, self.S0, self.corp]), dtype=torch.float64)
       T = tensor(np.array(self.T), dtype=torch.float64)
       K = tensor(np.array(self.K), dtype=torch.float64)
       garch  = tensor(np.array([self.alpha, self.beta, self.omega, self.gamma, self.lambda_]), dtype=torch.float64)
       return torch.cat([market_params, T, K, garch], dim=0)

class CaNNModel(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(CaNNModel, self).__init__()
        input_features = 3+6+9+5
        neurons = 200
        self.input_layer = nn.Linear(input_features, neurons)
        self.hl1 = nn.Linear(neurons, neurons)
        self.hl2 = nn.Linear(neurons, neurons)
        self.hl3 = nn.Linear(neurons, neurons)
        self.hl4 = nn.Linear(neurons, neurons)
        self.output_layer = nn.Linear(neurons, 9) # just need 1 price 

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)

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
        x = self.output_layer(x)
        return x


def train_and_predict(model, X, Y, num_epochs=1000, learning_rate=0.01):
    # Define the loss function and optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Ensure X and Y are of type Float
    X = X.float()
    Y = Y.float()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation mode
    model.eval()
    with torch.no_grad():
        predicted_prices = model(X)
        return predicted_prices


def run_trials(n_trials, day_num, corp, num_epochs=1000, learning_rate=0.01, opt_file=None):
    all_predicted = []
    all_true = []

    for trial in range(n_trials):
        # Create a folder for each trial
        trial_folder = f"results/trial_{trial + 1}"
        os.makedirs(trial_folder, exist_ok=True)

        input_features = ParamFeatures(day_num=day_num, corp=corp, opt_file=opt_file)
        X = input_features.get_features().float()

        Y = np.array(input_features.put if corp == -1 else input_features.call)
        Y = torch.tensor(Y, dtype=torch.float64).float()

        model = CaNNModel().float()
        criterion = nn.HuberLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        predicted = []
        for i in range(Y.shape[0]):
            predicted_prices = train_and_predict(model, X, Y[i], num_epochs=num_epochs, learning_rate=learning_rate)
            predicted_prices = predicted_prices.numpy()
            predicted.append(predicted_prices)

        all_predicted.append(predicted)
        all_true.append(Y.numpy())

        # Save the predicted data to a CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{trial_folder}/{'call' if corp == 1 else 'put'}{trial + 1}_Day{day_num}_{timestamp}.csv"
        pd.DataFrame(predicted).to_csv(filename, index=False)

    # Flatten the lists for plotting
    all_predicted = np.array(all_predicted).flatten()
    all_true = np.array(all_true).flatten()

    plt.figure(figsize=(10, 6))
    plt.scatter(all_true, all_predicted, color='blue', label='Predicted vs True')
    plt.plot([all_true.min(), all_true.max()], [all_true.min(), all_true.max()], 'k--', lw=2, label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    title = f"Day Number: {day_num}, Trials: {n_trials} ({'Call' if corp == 1 else 'Put'})"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/Graphs/{'Call' if corp == 1 else 'Put'}{n_trials}_Day{day_num}.png")

if __name__ == '__main__':
    run_trials(n_trials=25, day_num=1, corp=-1, opt_file=None)
