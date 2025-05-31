import numpy as np
from scipy.stats import norm

def zq(M, gamma):
    """
    Given an even number M and a gamma in (0,1), this function generates the discrete
    density distribution function {z, q}.

    Parameters:
        M (int): Number of spatial nodes at each time step (must be even and positive).
        gamma (float): Adjustable factor between 0 and 1.

    Returns:
        z (np.ndarray): Vector with M entities, the values of z in the distribution.
        q (np.ndarray): Vector with M entities, the probabilities corresponding to z.
        vzq (float): Variance of the distribution {z, q}.
        kzq (float): Kurtosis of the distribution {z, q}.
    """

    # if M % 2 != 0 or M <= 0:
    #     raise ValueError("M should be positive and even")

    I = np.arange(1, M // 2 + 1)
    q = np.zeros(M)
    q[:M // 2] = ((I - 0.5) ** gamma) / M
    qsum = np.sum(q[:M // 2])
    q[:M // 2] = q[:M // 2] / qsum / 2
    q[M // 2:] = q[:M // 2][::-1]

    z0 = np.zeros(M)
    z0[0] = norm.ppf(q[0] / 2)
    for i in range(1, M):
        qsum_i = np.sum(q[:i + 1])
        qavg = (qsum_i - np.sum(q[:i])) / 2
        z0[i] = norm.ppf(qsum_i - qavg)

    z = z0.copy()
    a = q[0] + q[-1]
    b = 2 * (q[-1] * z0[-1] - q[0] * z0[0])
    c = np.dot(q, z0 ** 2) - 1
    x = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    z[0] = z0[0] - x
    z[1:M - 1] = z0[1:M - 1]
    z[-1] = z0[-1] + x

    tmp = 1.5 - np.sum(q[:M // 2] * z[:M // 2] ** 4)
    a_corr = tmp / ((z[0] ** 2 - z[1] ** 2) * (z[0] ** 2 - z[M // 2 - 1] ** 2))
    b_corr = tmp / ((z[1] ** 2 - z[0] ** 2) * (z[1] ** 2 - z[M // 2 - 1] ** 2))

    q[0] += a_corr
    q[1] += b_corr
    q[M // 2 - 1] -= a_corr + b_corr
    q[M // 2:] = q[:M // 2][::-1]

    vzq = np.dot(q, z ** 2)
    kzq = np.dot(q, z ** 4)

    return z, q, vzq, kzq
