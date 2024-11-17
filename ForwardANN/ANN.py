import historical
from optdata import OptionData
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ParamFeatures:
    def __init__(self, T, day_num, alpha=1.33e-6, beta=0.8, omega=1e-6, gamma=100, lambda_=0.5, r=0.03, corp=1):
        # historical asset parameters
        self.historical = historical.historical()
        # market parameters
        self.r: float = 0.03
        self.S0: float = self.historical[-5+(day_num-1)]
        self.T: int = T # change to all maturities
        self.K= np.linspace(0.8, 1.2, 9)*self.S0
        # GARCH parameters
        self.alpha: float = alpha
        self.beta: float = beta
        self.omega: float = omega
        self.gamma: float = gamma
        self.lambda_: float = lambda_
        self.corp: int = corp
        # Option Prices given maturity and Day Number
        opt_data = OptionData(day_num=day_num)
        self.call = opt_data.call()
        self.put = opt_data.put()

    def get_features(self):
        # Convert scalars to tensor
        market_params = torch.tensor([
            self.r, self.S0, self.T, self.corp
        ], dtype=torch.float32)

        # K is now handled separately as an array
        strike_prices = torch.tensor(self.K, dtype=torch.float32)

        garch_params = torch.tensor([
            self.alpha, self.beta, self.omega,
            self.gamma, self.lambda_
        ], dtype=torch.float32)

        # Convert arrays to tensors
        historical = torch.tensor(self.historical, dtype=torch.float32)
        call_prices = torch.tensor(self.call, dtype=torch.float32)
        put_prices = torch.tensor(self.put, dtype=torch.float32)
        # need to have 1 for call and -1 for put for option param
        return {
            'market_params': market_params,
            'strike_prices': strike_prices,
            'garch_params': garch_params,
            'call_prices': call_prices,
            'put_prices': put_prices
        }

class CaNNModel(nn.Module):
    def __init__(self, market_dim=4, strike_dim=9, garch_dim=5):
        super(CaNNModel, self).__init__()
        # Separate processing branches
        self.market_branch = nn.Sequential(
            nn.Linear(self.market_dim, 8),
            nn.ReLU()
        )

        self.strike_branch = nn.Sequential(
            nn.Linear(self.strike_dim, 12),
            nn.ReLU()
        )

        self.garch_branch = nn.Sequential(
            nn.Linear(self.garch_dim, 8),
            nn.ReLU()
        )

        # Combined processing
        combined_dim = 8 + 12 + 8
        self.combined_layers = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        market_params = x[:, :4]
        strike_prices = x[:, 4:13]
        garch_params = x[:, 13:18]

        market_features = self.market_branch(market_params)
        strike_features = self.strike_branch(strike_prices)
        garch_features = self.garch_branch(garch_params)

        combined = torch.cat([market_features, strike_features, garch_features], dim=1)
        return self.combined_layers(combined)


if __name__ == '__main__':
    corp = 1
    input = ParamFeatures(5, 1, corp=corp)
    X = input.get_features()

    Y = []
    if corp == 1:
        Y = np.array(input.call)
    else:
        Y = np.array(input.put)
    #X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y)

    model = CaNNModel()
    model.forward(X["market_params"] + X["strike_prices"] + X["garch_params"])

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model()
        # handling put and call prices
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        predicted_prices = model(X_tensor)
        print(predicted_prices.numpy())  # Predicted option prices
