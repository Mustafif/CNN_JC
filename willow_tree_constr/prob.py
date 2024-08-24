import numpy as np
from scipy.stats import norm

def Prob(current_ht, next_ht, alpha, beta, gamma, omega):
    """
    This function calculates the transition probability from current_ht to next_ht.

    Parameters:
    current_ht (numpy.ndarray): Current values of ht
    next_ht (numpy.ndarray): Next values of ht
    alpha (float): Parameter for Heston-Nadi GARCH
    beta (float): Parameter for Heston-Nadi GARCH
    gamma (float): Parameter for Heston-Nadi GARCH
    omega (float): Parameter for Heston-Nadi GARCH

    Returns:
    p (numpy.ndarray): Transition probability from current_ht to next_ht
    """
    next_ht = np.array(next_ht)
    int_points = (next_ht[:-1] + next_ht[1:]) / 2
    print(f"int points {int_points}")
    num_ht = 1
    up_bound = np.nan * np.ones(len(int_points))
    low_bound = np.nan * np.ones(len(int_points))
    
    for i in range(num_ht):
        now_ht = current_ht
       # now_ht = np.full_like(int_points, now_ht)  # Create an array with the same shape as int_points
        up_bound[i] = float(gamma) * np.sqrt(now_ht) + np.sqrt((float(int_points - beta) * now_ht - omega) / alpha)
        low_bound[i] = float(gamma) * np.sqrt(now_ht) - np.sqrt((float(int_points - beta) * now_ht - omega) / alpha)
    
    
    if alpha > 0:
        prob = norm.cdf(up_bound) - norm.cdf(low_bound)
        prob = np.vstack((np.zeros((1, num_ht)), prob, np.ones((1, num_ht))))
        p = np.diff(prob, axis=0).T
    elif alpha < 0:
        prob = 1 - (norm.cdf(up_bound) - norm.cdf(low_bound))
        prob = np.vstack((np.ones((1, num_ht)), prob, np.zeros((1, num_ht))))
        p = (prob[:-1, :] - prob[1:, :]).T

    return p
