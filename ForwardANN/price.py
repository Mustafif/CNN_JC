import torch
import numpy as np

def price_american_option(S0, K, T, r, sigma, is_call=True, n_paths=10000, n_steps=50):
    """
    Price American option using a simplified LSM (Longstaff-Schwartz) method

    Parameters:
    S0 (float): Initial stock price
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free rate
    sigma (float): Volatility
    is_call (bool): True for call option, False for put option
    n_paths (int): Number of simulation paths
    n_steps (int): Number of time steps

    Returns:
    float: Option price
    """
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Time step
    dt = T/n_steps

    # Generate stock price paths
    Z = torch.randn((n_paths, n_steps), device=device)
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Initialize stock price paths
    S = torch.zeros((n_paths, n_steps+1), device=device)
    S[:, 0] = S0

    # Simulate paths
    for t in range(1, n_steps+1):
        S[:, t] = S[:, t-1] * torch.exp(drift + diffusion * Z[:, t-1])

    # Initialize exercise values
    if is_call:
        exercise_values = torch.maximum(S - K, torch.tensor(0.0, device=device))
    else:
        exercise_values = torch.maximum(K - S, torch.tensor(0.0, device=device))

    # Initialize value matrix with terminal values
    values = torch.zeros_like(exercise_values)
    values[:, -1] = exercise_values[:, -1]

    # Backward induction
    for t in range(n_steps-1, 0, -1):
        # Only consider in-the-money paths
        if is_call:
            itm = S[:, t] > K
        else:
            itm = S[:, t] < K

        if torch.sum(itm) > 0:
            # Select in-the-money paths
            S_itm = S[itm, t]

            # Create basis functions (1, S, S^2)
            basis = torch.stack([
                torch.ones_like(S_itm),
                S_itm,
                S_itm**2
            ]).T

            # Future cashflows
            discount_factor = torch.exp(-r * dt)
            future_values = values[itm, t+1:] * torch.exp(-r * dt * torch.arange(1, n_steps-t+1, device=device))
            discounted_values = torch.sum(future_values, dim=1)

            # Regression
            regression = torch.linalg.lstsq(basis, discounted_values).solution
            continuation_values = torch.matmul(basis, regression)

            # Exercise decision
            immediate_exercise = exercise_values[itm, t]
            exercise = immediate_exercise > continuation_values

            # Update values
            values[itm, t] = torch.where(exercise, immediate_exercise, 0)
            future_mask = exercise.unsqueeze(1).expand(-1, n_steps-t)
            values[itm, t+1:][future_mask] = 0

    # Discount all values
    discount_factors = torch.exp(-r * dt * torch.arange(n_steps+1, device=device))
    path_values = torch.sum(values * discount_factors, dim=1)

    # Calculate price and confidence interval
    option_price = torch.mean(path_values).item()
    return option_price
