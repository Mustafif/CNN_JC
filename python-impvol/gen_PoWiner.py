import numpy as np
from scipy.stats import norm

def gen_powiener(T, N, z):
    """
    Generate the transition probabilities of a standard Wiener process.

    Parameters:
        T (float): Maturity
        N (int): Number of time steps
        z (np.ndarray): m x 1 array of sorted normal quantile points

    Returns:
        P (np.ndarray): m x m x (N-1) array of transition matrices from t1 to tN
        q (np.ndarray): m-element vector of transition probabilities from t0 to t1
    """
    z = np.asarray(z).flatten()
    m = len(z)
    dt = T / N
    tt = np.linspace(dt, T, N)
    Yt = np.outer(z, np.sqrt(tt))  # shape (m, N)

    P = np.zeros((m, m, N - 1))
    sigma = np.sqrt(dt)

    # Compute q (transition from t0 to t1)
    C = np.zeros(m + 1)
    C[1:m] = (Yt[0:m-1, 0] + Yt[1:m, 0]) / 2
    C[0] = -np.inf
    C[m] = np.inf
    NF = norm.cdf(C, loc=0, scale=sigma)
    q = NF[1:] - NF[:-1]

    # Compute P for transitions t1 -> t2, ..., t(N-1) -> tN
    for n in range(N - 1):
        # Update C for time step n+1
        C[1:m] = (Yt[0:m-1, n+1] + Yt[1:m, n+1]) / 2
        C[0] = -np.inf
        C[m] = np.inf

        for i in range(m):
            mu = Yt[i, n]
            NF = norm.cdf(C, loc=mu, scale=sigma)
            probs = NF[1:] - NF[:-1]
            probs = probcali(probs, Yt[:, n+1], mu, sigma)
            P[i, :, n] = probs / np.sum(probs)

    return P, q

def probcali(p, y, mu, sigma):
    """
    Calibrate a transition probability vector to match the target mean (mu) and variance (sigma^2).

    Parameters:
        p (np.ndarray): Original transition probabilities, shape (N,)
        y (np.ndarray): Variable values (states), shape (N,)
        mu (float): Target mean
        sigma (float): Target standard deviation

    Returns:
        pc (np.ndarray): Calibrated transition probabilities, shape (N,)
    """
    p = np.asarray(p).flatten()
    y = np.asarray(y).flatten()
    m = len(p)

    a = np.dot(p, y) - mu
    b = np.dot(p, y ** 2) - mu ** 2 - sigma ** 2

    # Find the top 3 largest probabilities
    pind = np.argsort(p)  # ascending
    wm, w2, w1 = pind[-1], pind[-2], pind[-3]

    ym = y[wm]
    y2 = y[w2]
    y1 = y[w1]

    x = np.zeros(m)
    denominator = ym**2 - y2**2 - (y1 + y2)*(ym - y2)
    if denominator == 0:
        return p  # Avoid division by zero, return unchanged

    x[wm] = (-b + a * (y1 + y2)) / denominator
    x[w1] = (-a - x[wm] * (ym - y2)) / (y1 - y2)
    x[w2] = - (x[wm] + x[w1])

    pc = p + x
    pc = np.maximum(pc, 0)
    pc /= np.sum(pc)

    return pc
