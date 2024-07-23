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

    intPoints = (next_ht[:-1] + next_ht[1:]) / 2
    numHt = len(current_ht)
    upBound = np.zeros((len(intPoints), numHt))
    lowBound = np.zeros((len(intPoints), numHt))

    for i in range(numHt):
        nowHt = current_ht[i]
        upBound[:, i] = gamma * np.sqrt(nowHt) + np.sqrt((intPoints - beta * nowHt - omega) / alpha)
        lowBound[:, i] = gamma * np.sqrt(nowHt) - np.sqrt((intPoints - beta * nowHt - omega) / alpha)

    if alpha > 0:
        prob = norm.cdf(np.real(upBound)) - norm.cdf(np.real(lowBound))
        prob = np.vstack((np.zeros(numHt), prob, np.ones(numHt)))
        p = np.diff(prob, axis=0).T
    elif alpha < 0:
        prob = 1 - (norm.cdf(np.real(upBound)) - norm.cdf(np.real(lowBound)))
        prob = np.vstack((np.ones(numHt), prob, np.zeros(numHt)))
        p = (prob[:-1, :] - prob[1:, :]).T

    return p
