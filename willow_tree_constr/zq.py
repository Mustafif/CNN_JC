import numpy as np
from scipy.stats import norm

def zq(M, gamma):
    """
    [z, q, vzq, kzq] = zq(M, gamma)

    Introduction:
        Given an even number M and a gamma belonging to (0,1), 
        this function generates the discrete density distribution function {z,q}.

    Input:
        M (an even scalar): number of spatial nodes at each time step;
        gamma (a scalar): an adjustable factor between 0 and 1.

    Output:
        z (1*N vector): a row vector with M entities, z of the function {z,q};
        q (1*N vector): also a row vector with M entities, the probabilities of z;
        vzq (a scalar): the variance of {z,q};
        kzq (a scalar): the kurtosis of {z,q}.

    References:
        1. W.Xu, Z.Hong, and C.Qin, A new sampling strategy willow tree method with application
           to path-dependent option pricing, 2013.

    Implemented by:
        G.Wang  2016.12.15.
    """
    # Check if M is positive and even
    if M % 2 != 0 or M < 0:
        raise ValueError("M should be positive and even")
    # Check if gamma is between 0 and 1
    if gamma < 0 or gamma > 1:
        raise ValueError("gamma should be a number between 0 and 1")

    # Create an array I of integers from 1 to M/2
    I = np.arange(1, M // 2 + 1)
    # Initialize an array q of zeros with length M
    q = np.zeros(M)
    # Calculate the first half of q using the formula (I-0.5)^gamma / M
    q[:M // 2] = (I - 0.5) ** gamma / M
    # Calculate the sum of the first half of q
    qsum = np.sum(q[:M // 2])
    # Normalize the first half of q by dividing by qsum and 2
    q[:M // 2] = q[:M // 2] / qsum / 2
    # Copy the first half of q to the second half of q in reverse order
    q[M // 2:] = q[:M // 2][::-1]

    # Initialize an array z0 of zeros with length M
    z0 = np.zeros(M)
    # Calculate the first element of z0 using the inverse CDF of the normal distribution
    z0[0] = norm.ppf(q[0] / 2)
    # Calculate the remaining elements of z0 using the inverse CDF of the normal distribution
    for i in range(1, M):
        z0[i] = norm.ppf(np.sum(q[:i + 1]) - (np.sum(q[:i + 1]) - np.sum(q[:i])) / 2)

    # Copy z0 to z
    z = z0.copy()
    # Calculate the coefficients a, b, and c
    a = q[0] + q[-1]
    b = 2 * (q[-1] * z0[-1] - q[0] * z0[0])
    c = np.sum(q * z0 ** 2) - 1
    # Calculate the value of x
    x = (-b + np.sqrt(b ** 2 - (4 * a * c))) / (2*a)
    # Adjust the first and last elements of z using the value of x
    z[0] = z0[0] - x
    z[1:-1] = z0[1:-1]
    z[-1] = z0[-1] + x

    # Calculate the coefficients a and b
    tmp = 1.5 - np.sum(q[:M // 2] * z[:M // 2] ** 4)
    a = tmp / (z[0] ** 2 - z[1] ** 2) / (z[0] ** 2 - z[M // 2] ** 2)
    b = tmp / (z[1] ** 2 - z[0] ** 2) / (z[1] ** 2 - z[M // 2] ** 2)

    # Adjust the first, second, and middle elements of q using the values of a and b
    q[0] += a
    q[1] += b
    q[M // 2] -= a + b
    # Copy the first half of q (excluding the middle element) to the second half of q in reverse order
    q[M // 2 + 1:] = q[:M // 2 - 1][::-1]

    # Calculate the variance and kurtosis of z
    vzq = np.sum(q * z ** 2)
    kzq = np.sum(q * z ** 4)

    # Return the values of z, q, vzq, and kzq
    return z, q, vzq, kzq
