import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

import historical
import lsm
import price as p

# Step 1: Prepare your data
# Assuming 'stock_prices' is a 1D NumPy array of historical stock prices
# and you have defined your constants
K = 0.8  # strike price moneyness
r = 0.03  # risk-free rate
T = 5  # time to maturity in weeks
stock_prices = historical.historical()
# Create feature and label arrays
X = []  # Features
y = []  # Labels (option prices)

for price in stock_prices:
    feature = [price, K * price, T]  # current price, strike price, time to maturity
    # instead of price have S0, K*price = S_t and T
    option_price = max(price - (K * price), 0)  # we want to be a func to calculate opt price 
    X.append(feature)
    y.append(option_price)

X = np.array(X)
y = np.array(y)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).view(-1, 1)  # Reshape for PyTorch

# Step 2: Define the Neural Network
class OptionPricingModel(nn.Module):
    def __init__(self):
        super(OptionPricingModel, self).__init__()
        # input we have option features, prices and model features to get ANN to learn to price itself 
        self.fc1 = nn.Linear(3, 100)  # 3 input features, 100 hidden units
        self.fc2 = nn.Linear(100, 100)  # 100 hidden units
        self.fc3 = nn.Linear(100, 1)   # 1 output (option price)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = OptionPricingModel()

# Step 3: Train the Model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(X_tensor)  # Forward pass
    loss = criterion(outputs, y_tensor)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 4: Make Predictions
model.eval()
with torch.no_grad():
    predicted_prices = model(X_tensor)
    print(predicted_prices.numpy())  # Predicted option prices
