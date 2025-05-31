# import numpy as np

# def american_option_price(nodes, P, q, r, T, S0, K, index):
#     """
#     Price an American option using the Willow Tree method.

#     Parameters:
#         nodes (ndarray): M x N matrix of willow tree nodes
#         P (ndarray): M x M x (N-1) tensor of transition matrices
#         q (ndarray): M-length vector of initial probabilities
#         r (float): risk-free interest rate
#         T (float): time to maturity
#         S0 (float): initial stock price
#         K (float): strike price
#         index (int): +1 for call, -1 for put

#     Returns:
#         price (float): computed option price
#         V (ndarray): M x N value matrix of the option
#     """
#     M, N = nodes.shape
#     V = np.zeros((M, N))

#     # Payoff at maturity
#     # V[:, -1] = np.maximum(index * (nodes[:, -1] - K), 0)
#     if index == 1:
#         V[:, -1] = np.maximum(nodes[:, -1] - K, 0)
#     elif index == -1:
#         V[:, -1] = np.maximum(K - nodes[:, -1], 0)
#     dt = T / N
#     # Backward induction
#     for i in reversed(range(N - 1)):
#         cont_values = np.zeros(M)
#         for j in range(M):
#             for k in range(M):
#                 cont_values[j] = cont_values[j] + P[j, k, i] * V[k, i + 1]

#         cont_values = np.exp(-r * dt) * cont_values
#         exercise = np.maximum(index * (nodes[:, i] - K), 0)
#         V[:, i] = np.maximum(exercise, cont_values)

#     # Price at time 0
#     price = np.maximum(index * (S0 - K), 0)
#     expected = np.exp(-r * dt) * q * V[:, 0]
#     price = np.maximum(price, expected)

#     return price, V

import numpy as np

def American(nodes, P, q, r, T, S0, K, index):
    #############################################
    #
    # Introduction
    # Given the willow tree including tree-nodes matrix and transition
    # probability matrices, this function calculates the price of an
    # American option (put or call).
    #
    # Input
    # nodes (M*N matrix) willow tree-nodes matrix of stock prices
    # P (M*M*(N-1) matrix) P[i,j,k] represents the transition probability from i-th node
    #                      at time t_k to j-th node at t_{k+1}
    # q (M vector) q[i] represents the initial transition probability
    # r (scalar) risk-free rate
    # T (scalar) time to expiration
    # S0 (scalar) initial stock price
    # K (scalar) strike price
    # index (scalar) +1 for call and -1 for put
    #
    # Output
    # price: the computed price of the option
    # V: option values at each node
    #
    #############################################

    # Get dimensions of the tree
    M, N = nodes.shape

    # Initialize option values matrix
    V = np.zeros((M, N))

    # Terminal condition at maturity
    if index == 1:  # Call option
        V[:, N-1] = np.maximum(nodes[:, N-1] - K, 0)
    else:  # Put option
        V[:, N-1] = np.maximum(K - nodes[:, N-1], 0)

    # Backward induction
    dt = T / N
    for i in range(N-2, -1, -1):
        # Calculate continuation values explicitly
        cont_values = np.zeros(M)
        for j in range(M):
            for k in range(M):
                cont_values[j] += P[j, k, i] * V[k, i+1]
        cont_values = np.exp(-r * dt) * cont_values

        # Calculate exercise values
        if index == 1:  # Call option
            exercise_values = np.maximum(nodes[:, i] - K, 0)
        else:  # Put option
            exercise_values = np.maximum(K - nodes[:, i], 0)

        # Take maximum
        V[:, i] = np.maximum(exercise_values, cont_values)

    # Calculate final price
    if index == 1:  # Call option
        immediate_exercise = max(S0 - K, 0)
    else:  # Put option
        immediate_exercise = max(K - S0, 0)

    continuation_value = np.exp(-r * dt) * np.dot(q, V[:, 0])
    price = max(immediate_exercise, continuation_value)

    return price, V
