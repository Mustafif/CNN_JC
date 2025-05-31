import numpy as np

def american_option_price(nodes, P, q, r, T, S0, K, index):
    """
    Price an American option using the Willow Tree method.

    Parameters:
        nodes (ndarray): M x N matrix of willow tree nodes
        P (ndarray): M x M x (N-1) tensor of transition matrices
        q (ndarray): M-length vector of initial probabilities
        r (float): risk-free interest rate
        T (float): time to maturity
        S0 (float): initial stock price
        K (float): strike price
        index (int): +1 for call, -1 for put

    Returns:
        price (float): computed option price
        V (ndarray): M x N value matrix of the option
    """
    M, N = nodes.shape
    V = np.zeros((M, N))

    # Payoff at maturity
    V[:, -1] = np.maximum(index * (nodes[:, -1] - K), 0)

    # Backward induction
    for i in reversed(range(N - 1)):
        continuation = np.exp(-r * T / N) * P[:, :, i] @ V[:, i + 1]
        exercise = np.maximum(index * (nodes[:, i] - K), 0)
        V[:, i] = np.maximum(exercise, continuation)

    # Price at time 0
    price = max(index * (S0 - K), 0)
    expected = np.exp(-r * T / N) * q @ V[:, 0]
    price = max(price, expected)

    return price, V
