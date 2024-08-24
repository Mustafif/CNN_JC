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

    m = len(nodes_ht)
    n = len(nodes_ht[0])

    p_ht = np.zeros((m, n))
    p_ht_n = np.zeros((m, m, n - 1))
    
    curr_h = h0
    next_h = nodes_ht[0]
    p = Prob(curr_h, next_h, alpha, beta, gamma, omega)
    p_ht[:, 0] = p.flatten()
    
    for i in range(1, n):
        next_h = nodes_ht[:, i]
        for j in range(m):
            curr_h = nodes_ht[j, i - 1]
            p = Prob(curr_h, next_h, alpha, beta, gamma, omega)
            p_ht_n[j, :, i - 1] = p.flatten()
        p_ht[:, i] = (p_ht[:, i - 1].T @ p_ht_n[:, :, i - 1]).T
    
    return p_ht_n, p_ht


