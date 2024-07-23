import numpy as np

def American(nodes, P, q, r, T, S0, K, index):
    """
    Calculate the price of an American option given the willow tree including 
    tree-nodes matrix and transition probability matrices.

    Parameters:
    nodes (MxN matrix): Willow tree-nodes matrix
    P (MxMx(N-1) matrix): Transition probability matrices
    q (M array): Transition probability from initial rate
    r (float): Risk-free rate
    T (float): Expiration time
    S0 (float): Initial stock price
    K (float): Exercise price
    index (int): +1 for call and -1 for put

    Returns:
    price (float): The computed price of the option
    V (MxN matrix): Option values at each node

    Implemented by G.Wang 2016.12
    """
    M, N = nodes.shape
    
    # Initialize V with the payoff at maturity date
    V = np.zeros((M, N))
    V[:, -1] = np.maximum(index * (nodes[:, -1] - K), 0)

    # Backward induction
    for i in range(N-2, -1, -1):
        intrinsic_value = np.maximum(index * (nodes[:, i] - K), 0)
        continuation_value = np.exp(-r * T / N) * np.dot(P[:, :, i], V[:, i+1])
        V[:, i] = np.maximum(intrinsic_value, continuation_value)

    # Calculate the option price
    intrinsic_value_s0 = max(index * (S0 - K), 0)
    continuation_value_s0 = np.exp(-r * T / N) * np.dot(q, V[:, 0])
    price = max(intrinsic_value_s0, continuation_value_s0)

    return price, V