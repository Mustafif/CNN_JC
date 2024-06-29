import numpy as np
from multiprocessing import Pool

def calculate_V(i, nodes, P, V, r, T, K, index):
    return i, np.maximum(np.maximum(index * (nodes[:, i] - K), 0), np.exp(-r * T / N) * np.dot(P[:, :, i], V[:, i + 1]))

def American(nodes, P, q, r, T, S0, K, index, parallel=False):
    """
Given the willow tree including tree-nodes matrix and transition
probability matrices, this function calculates the price of an
American put option.

Input:
    nodes (M*N matrix): willow tree-nodes matrix
    P (M*M*(N-1) matrix): P[i, j, k] represents the transition probability from i-th node
                          at time t_k to j-th node at t_{k+1}.
    q (M*1 vector): q[i] represents the transition probability from initial rate
    r (a scalar): risk free rate
    T (a scalar): expiration
    S0 (a scalar): initial stock price
    K (a scalar): exercise price
    index (a scalar): +1 for call and -1 for put

Output:
    price: the computed price of the option
    V: the option values at each node and time step
    """
    N = nodes.shape[1]
    V = np.zeros_like(nodes)    
    V[:, N - 1] = np.maximum(index * (nodes[:, N - 1] - K), 0)

    if parallel:
        with Pool() as pool:
            results = pool.starmap(calculate_V, [(i, nodes, P, V, r, T, K, index) for i in range(N - 2, -1, -1)])

        for i, result in results:
            V[:, i] = result
    else:
        for i in range(N - 2, -1, -1):
            V[:, i] = np.maximum(np.maximum(index * (nodes[:, i] - K), 0), np.exp(-r * T / N) * np.dot(P[:, :, i], V[:, i + 1]))

    price = np.maximum(np.maximum(index * (S0 - K), 0), np.exp(-r * T / N) * np.dot(q, V[:, 0]))

    return price, V