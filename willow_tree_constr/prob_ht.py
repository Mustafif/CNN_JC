import numpy as np
from prob import Prob

def Prob_ht(nodes_ht, h0, alpha, beta, gamma, omega):
    """
    This function computes the transition probabilities and probability of h_t of the Heston-Nadi GARCH model
    and constructs all tree nodes of h_t.

    Parameters:
    nodes_ht (numpy.ndarray): Tree nodes of ht
    h0 (float): Initial value of ht
    alpha (float): Parameter for Heston-Nadi GARCH
    beta (float): Parameter for Heston-Nadi GARCH
    gamma (float): Parameter for Heston-Nadi GARCH
    omega (float): Parameter for Heston-Nadi GARCH

    Returns:
    P_ht_N (numpy.ndarray): Transition probability matrix of ht, 3-d array
    P_ht (numpy.ndarray): Probability of ht given h0
    """

    M = len(nodes_ht)
    N = len(nodes_ht[0])

    P_ht = np.zeros((M, N))  # transition probability matrix between two time steps
    P_ht_N = np.zeros((M, M, N - 1))  # probability for hd->h2d

    curr_h = h0
    next_h = nodes_ht[0]
    p = Prob(curr_h, next_h, alpha, beta, gamma, omega)
    P_ht[:, 0] = p.reshape(-1)

    for n in range(1, N):
        next_h = nodes_ht[:, n]
        for i in range(M):
            curr_h = nodes_ht[i, n - 1]
            p = Prob(curr_h, next_h, alpha, beta, gamma, omega)
            P_ht_N[i, :, n - 1] = p.reshape(-1)
        P_ht[:, n] = np.dot(P_ht[:, n - 1].reshape(1, -1), P_ht_N[:, :, n - 1]).reshape(-1)

    return P_ht_N, P_ht
