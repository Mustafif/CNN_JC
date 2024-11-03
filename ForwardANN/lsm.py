import torch
import torch.nn as nn
import numpy as np
class LSMOptionPricer:
    def __init__(self, historical_prices, strike_price, risk_free_rate, time_to_maturity, is_call=True):
        """
        Initialize the LSM Option Pricer using PyTorch

        Parameters:
        historical_prices (array-like): Array of historical stock prices
        strike_price (float): Strike price of the option
        risk_free_rate (float): Annual risk-free rate (decimal)
        time_to_maturity (float): Time to maturity in years
        is_call (bool): True for call option, False for put option
        device (str): Device to run computations on ('cuda' or 'cpu')
        """
        self.S = torch.tensor(historical_prices, dtype=torch.float32)
        self.K = strike_price
        self.r = risk_free_rate
        self.T = time_to_maturity
        self.is_call = is_call

        # Calculate daily returns and volatility
        self.daily_returns = torch.log(self.S[1:] / self.S[:-1])
        self.volatility = torch.std(self.daily_returns) * torch.sqrt(torch.tensor(252.0))

    def generate_paths(self, n_paths, n_steps):
        """
        Generate Monte Carlo price paths using historical volatility

        Parameters:
        n_paths (int): Number of price paths to simulate
        n_steps (int): Number of time steps

        Returns:
        torch.Tensor: Matrix of simulated price paths
        """
        dt = self.T / n_steps
        S0 = self.S[-1]  # Use most recent price as starting point

        # Generate random normal variables
        Z = torch.randn((n_paths, n_steps))

        # Initialize price paths tensor
        paths = torch.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        # Generate paths using geometric Brownian motion
        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t-1] * torch.exp(
                (self.r - 0.5 * self.volatility**2) * dt +
                self.volatility * torch.sqrt(torch.tensor(dt)) * Z[:, t-1]
            )

        return paths

    def payoff(self, stock_price):
        """Calculate the option payoff"""
        if self.is_call:
            return torch.maximum(stock_price - self.K, torch.tensor(0.0))
        else:
            return torch.maximum(self.K - stock_price, torch.tensor(0.0))

    class RegressionModel(nn.Module):
        """Neural network for regression in the LSM algorithm"""
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x):
            return self.net(x)

    def price_option(self, n_paths=10000, n_steps=50, batch_size=1000):
        """
        Price the American option using LSM method with batched computation

        Parameters:
        n_paths (int): Number of Monte Carlo paths
        n_steps (int): Number of time steps
        batch_size (int): Batch size for path generation and computation

        Returns:
        dict: Pricing statistics including price, confidence interval, and early exercise premium
        """
        paths = self.generate_paths(n_paths, n_steps)
        dt = self.T / n_steps

        # Initialize cash flow tensor
        cash_flows = torch.zeros_like(paths)
        cash_flows[:, -1] = self.payoff(paths[:, -1])

        # Backward induction through time steps
        for t in range(n_steps-1, 0, -1):
            # Find paths where option is in-the-money
            if self.is_call:
                itm = paths[:, t] > self.K
            else:
                itm = paths[:, t] < self.K

            if torch.sum(itm) > 0:
                # Select in-the-money paths
                S_itm = paths[itm, t].reshape(-1, 1)

                # Calculate discounted future cash flows
                discount_factor = np.exp(-self.r * dt)
                discounted_cashflows = discount_factor * cash_flows[itm, t+1:]
                Y = torch.sum(discounted_cashflows, dim=1).reshape(-1, 1)

                # Create and train regression model
                model = self.RegressionModel()
                optimizer = torch.optim.Adam(model.parameters())

                # Mini-batch training
                n_batches = (len(S_itm) + batch_size - 1) // batch_size
                for _ in range(50):  # Training epochs
                    for i in range(n_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, len(S_itm))

                        X_batch = S_itm[start_idx:end_idx]
                        y_batch = Y[start_idx:end_idx]

                        optimizer.zero_grad()
                        pred = model(X_batch)
                        loss = nn.MSELoss()(pred, y_batch)
                        loss.backward()
                        optimizer.step()

                # Calculate continuation value
                with torch.no_grad():
                    continuation_value = model(S_itm)

                # Compare immediate exercise with continuation value
                immediate_exercise = self.payoff(S_itm.squeeze())

                # Update cash flows tensor
                exercise = immediate_exercise > continuation_value.squeeze()
                cash_flows[itm, t] = torch.where(
                    exercise,
                    immediate_exercise,
                    torch.tensor(0.0)
                )

                # Update future cash flows for paths where we don't exercise
                cash_flows[itm, t+1:][exercise] = 0

        # Calculate option price and statistics
        discount_factors = torch.exp(-self.r * torch.arange(n_steps+1) * dt)
        path_values = torch.sum(cash_flows * discount_factors, dim=1)
        option_price = torch.mean(path_values)
        std_error = torch.std(path_values) / torch.sqrt(torch.tensor(n_paths))

        # Move results to CPU for return
        stats = {
            'price': option_price.item(),
            'std_error': std_error.item(),
            'confidence_95': [
                (option_price - 1.96 * std_error).item(),
                (option_price + 1.96 * std_error).item()
            ],
            'early_exercise_premium': (option_price - self.price_european(n_paths, n_steps)).item()
        }

        return stats

    def price_european(self, n_paths=10000, n_steps=50):
        """Calculate equivalent European option price for comparison"""
        paths = self.generate_paths(n_paths, n_steps)
        payoffs = self.payoff(paths[:, -1])
        discount_factor = np.exp(-self.r * self.T)
        return torch.mean(payoffs) * discount_factor
