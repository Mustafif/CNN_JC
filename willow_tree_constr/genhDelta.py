import numpy as np
from zq import zq
from scipy.stats import norm

def genhDelta(h0, beta, alpha, gamma, omega, mh, gamma_h):
    """
    This function generates nodes of a standard normal distribution and calculates 
    the probability distribution.

    Parameters:
    h0 (float): Initial value of h
    beta (float): Coefficient of the linear term in the equation for hd
    alpha (float): Coefficient of the quadratic term in the equation for hd
    gamma (float): Coefficient of the square root term in the equation for hd
    omega (float): Constant term in the equation for hd
    mh (int): Number of nodes to generate for the standard normal distribution
    gamma_h (float): Standard deviation of the standard normal distribution

    Returns:
    hd (numpy.ndarray): Sorted array of nodes generated from the standard normal distribution
    q (numpy.ndarray): Array of probabilities calculated from the nodes
    """
    # generate nodes of standard normal distribution
    z, q, _, _ = zq(mh, gamma_h)
    z = z.reshape(-1, 1)
    hd = omega + beta * h0 + alpha * (z - gamma * np.sqrt(h0)) ** 2
    hd = np.sort(hd, axis=0)
    intPoints = (hd[:-1] + hd[1:]) / 2

    numHt = len(hd)

    upBound = gamma * np.sqrt(h0) + np.sqrt((intPoints - omega - beta * h0) / alpha)
    lowBound = gamma * np.sqrt(h0) - np.sqrt((intPoints - omega - beta * h0) / alpha)

    if alpha > 0:
        prob = norm.cdf(np.real(upBound)) - norm.cdf(np.real(lowBound))
        prob = prob.flat
        print(prob)
        prob = np.concatenate(([0], prob, [1]))
        q = np.diff(prob)
    elif alpha < 0:
        prob = 1 - (norm.cdf(np.real(upBound)) - norm.cdf(np.real(lowBound)))
        prob = np.concatenate((np.ones((1, numHt)), prob, np.zeros((1, numHt))))
        q = prob[:-1] - prob[1:]
    return hd, q