class OptionPricingNN(nn.Module):
    def __init__(self, hist_size, num_strikes):
        super().__init__()

        # Input dimensions
        self.market_dim = 3      # r, S0, T
        self.strike_dim = num_strikes  # K array
        self.garch_dim = 5       # alpha, beta, omega, gamma, lambda
        self.hist_size = hist_size

        # Processing branches
        self.market_branch = nn.Sequential(
            nn.Linear(self.market_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        self.strike_branch = nn.Sequential(
            nn.Linear(self.strike_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.garch_branch = nn.Sequential(
            nn.Linear(self.garch_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        self.historical_branch = nn.Sequential(
            nn.Linear(self.hist_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Combined processing
        combined_dim = 32 + 64 + 32 + 64  # Sum of output dims from all branches
        self.combined_layers = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_strikes * 2)  # Output both call and put prices for each strike
        )

    def forward(self, market_params, strike_prices, garch_params, historical):
        # Process each branch
        market_features = self.market_branch(market_params)
        strike_features = self.strike_branch(strike_prices)
        garch_features = self.garch_branch(garch_params)
        hist_features = self.historical_branch(historical)

        # Concatenate all features
        combined = torch.cat([
            market_features,
            strike_features,
            garch_features,
            hist_features
        ], dim=1)

        # Process combined features
        output = self.combined_layers(combined)

        # Split output into call and put prices
        num_strikes = strike_prices.shape[1]
        call_prices = output[:, :num_strikes]
        put_prices = output[:, num_strikes:]

        return call_prices, put_prices

def get_features(self):
    # Convert inputs to tensors
    market_params = torch.tensor([
        self.r, self.S0, self.T
    ], dtype=torch.float32)

    strike_prices = torch.tensor(self.K, dtype=torch.float32)

    garch_params = torch.tensor([
        self.alpha, self.beta, self.omega,
        self.gamma, self.lambda_
    ], dtype=torch.float32)

    historical = torch.tensor(self.historical, dtype=torch.float32)

    return {
        'market_params': market_params,
        'strike_prices': strike_prices,
        'garch_params': garch_params,
        'historical': historical
    }

# Training loop
num_strikes = len(K)  # number of strike prices
model = OptionPricingNN(hist_size=100, num_strikes=num_strikes)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1000

# Get your features
X = get_features()
market_params = X['market_params']
strike_prices = X['strike_prices']
garch_params = X['garch_params']
historical = X['historical']

# Actual call/put prices for validation
actual_call = torch.tensor(call_prices, dtype=torch.float32)
actual_put = torch.tensor(put_prices, dtype=torch.float32)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    pred_call, pred_put = model(
        market_params,
        strike_prices,
        garch_params,
        historical
    )

    # Calculate loss using both call and put prices
    call_loss = criterion(pred_call, actual_call)
    put_loss = criterion(pred_put, actual_put)
    total_loss = call_loss + put_loss

    total_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Call Loss: {call_loss.item():.4f}')
        print(f'Put Loss: {put_loss.item():.4f}')
        print(f'Total Loss: {total_loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    pred_call, pred_put = model(
        market_params,
        strike_prices,
        garch_params,
        historical
    )
    # Compare predictions with actual prices
    print("Predicted Call Prices:", pred_call.numpy())
    print("Actual Call Prices:", actual_call.numpy())
    print("Predicted Put Prices:", pred_put.numpy())
    print("Actual Put Prices:", actual_put.numpy())
