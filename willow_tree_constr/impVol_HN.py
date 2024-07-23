import numpy as np
from scipy.stats import norm
from Treenodes_logSt_HN import Treenodes_logSt_HN
from prob_Xt import Prob_Xt
from American import American
from genhDelta import genhDelta
from Treenodes_ht_HN import Treenodes_ht_HN
from zq import zq


def impvol(S0, K, T, r, V, index, N, m, gamma, tol, itmax):
    """
    Compute the implied volatility of the American option by willow tree
    through the bisection method.

    Input:
        S0 -- initial value of stock price
        K -- strike prices, matrix
        T -- maturities, vector
        r -- interest rate
        V -- American option prices
        N -- # of time steps of willow tree
        m -- # of tree nodes ate each time step
        gamma -- gamma for sampling z

    Output:
        imp_vol -- implied volatilities of the American options
    """
    n = len(T)
    k = K.shape[0]
    z = zq(m, gamma)
    imp_vol = np.zeros((k, n))

    for i in range(n):  # each maturity
        P, q = gen_PoWiner(T[i], N, z)
        for j in range(k):
            V0 = 10000
            it = 0
            a = 0
            b = 1
            sigma = (a + b) / 2
            while abs(V0 - V[j, i]) > tol and it < itmax:
                Xnodes = nodes_wiener(T, N, z, r, sigma)
                nodes = S0 * np.exp(Xnodes)
                V0 = American(nodes, P, q, r, T[i], S0, K[j, i], index[j, i])
                if V0 > V[j, i]:
                    b = sigma
                else:
                    a = sigma
                sigma = (a + b) / 2
                it += 1
            imp_vol[j, i] = sigma

    return imp_vol, S0, it


def impVol_HN(r, lambda_, w, beta, alpha, gamma, h0, S0, K, T, N, m_h, m_x, CorP):
    """
    Compute implied volatility of American options with underlying in HN-GARCH
    """
    c = gamma + lambda_ + 0.5
    gamma_h = 0.6
    gamma_x = 0.8
    tol = 1e-4
    itmax = 60

    hd, qhd = genhDelta(h0, beta, alpha, c, w, m_h, gamma_h)
    nodes_ht = Treenodes_ht_HN(m_h, hd, qhd, gamma_h, alpha, beta, c, w, N + 1)
    nodes_Xt, _, _, _, _ = Treenodes_logSt_HN(
        m_x, gamma_x, r, hd, qhd, S0, alpha, beta, c, w, N
    )
    q_Xt, P_Xt, _ = Prob_Xt(nodes_ht, qhd, nodes_Xt, S0, r, alpha, beta, c, w)
    nodes_S = np.exp(nodes_Xt)
    V = American(nodes_S, P_Xt, q_Xt, r, T, S0, K, CorP)
    sig, V0, it = impvol(S0, K, T, r, V[0, 0], CorP, N, m_x, gamma_x, tol, itmax)

    return sig, V, V0


def gen_PoWiner(T, N, z):
    """
    Generate the transition probabilities of a standard Winner process.

    Input:
        T -- maturity
        N -- number of time steps
        z -- m*1-vector of normal distribution points [z1, z2, ..., zm], z1 < z2 < ... < zm

    Output:
        P -- transition probability matrices from t1 to tN, m*m*N
        q -- transition probability vector from t0 to t1
    """
    m = len(z)
    dt = T / N
    tt = np.linspace(dt, T, N)
    Yt = z * np.sqrt(tt)  # (n))*z(i); Yt size m*N
    P = np.zeros((m, m, N - 1))

    sigma = np.sqrt(dt)
    C = np.zeros(m + 1)

    for i in range(1, m):
        C[i] = (Yt[i - 1, 0] + Yt[i, 0]) / 2

    C[0] = -np.inf
    C[-1] = np.inf
    NF = norm.cdf(C, 0, sigma)
    q = NF[1:] - NF[:-1]

    for n in range(N - 1):
        for i in range(1, m):
            C[i] = (Yt[i - 1, n + 1] + Yt[i, n + 1]) / 2

        for i in range(m):
            mu = Yt[i, n]
            NF = norm.cdf(C, mu, sigma)
            P[i, :, n] = NF[1:] - NF[:-1]
            P[i, :, n] = probcali(P[i, :, n], Yt[:, n + 1], Yt[i, n], sigma)
            P[i, :, n] /= np.sum(P[i, :, n])

    return P, q


def probcali(p, y, mu, sigma):
    """
    Introduction
           Given a row vector and a column vector, two scalars, i.e., mean and variance,
            this function calibrate transition probability to keep mean and variance right

    Input
         p (1*N vector)  : original transition probability, row vector
         y (N*1 vector)  : variables, column vector
        mu (a scalar)     : mean
       sigma (a scalar) : standard volatility

    Output
          pc (1*N vector) : probability vector after calibration

    References
     1. W.Xu, L.Lu,two-factor willow tree method for convertibel bond pricing
        with stochastic interest rate and default risk, 2016.

    Implemented by
           L.Lu 2016.12.15
    """
    m = len(p)
    a = np.dot(p, y) - mu
    b = np.dot(p, y**2) - mu**2 - sigma**2
    pind = np.argsort(p)  # from to min to max
    x = np.zeros(m)
    wm = pind[-1]  # max
    w2 = pind[-2]
    w1 = pind[-3]
    y1 = y[w1]  # min
    y2 = y[w2]  # middle
    ym = y[wm]  # max probability
    x[wm] = (-b + a * (y1 + y2)) / (ym**2 - y2**2 - (y1 + y2) * (ym - y2))
    x[w1] = (-a - x[wm] * (ym - y2)) / (y1 - y2)
    x[w2] = -(x[wm] + x[w1])

    pc = p + x
    pc = np.maximum(pc, 0)
    pc = pc / np.sum(pc)

    return pc


def nodes_wiener(T, N, z, r, sigma):
    """
    Construct a willow tree for standard Brownian motion with maturity T in N time steps.
    Generate the transition probabilities of a standard Wiener process.

    Parameters:
    T (float): Maturity
    N (int): Number of time steps
    z (numpy array): m*1-vector of normal distribution points [z1,z2,....,zm], z1<z2...<zm
    r (float): Interest rate
    sigma (float): Volatility of stock price

    Returns:
    nodes (numpy array): Tree nodes of the standard Brownian motion (m x N)

    Note: The function name has been corrected from 'Winer' to 'Wiener'.
    """
    m = len(z)
    dt = T / N
    tt = np.linspace(dt, T, N)
    z = z.reshape(-1, 1)
    t = np.tile(tt, (m, 1))
    nodes = (r - sigma**2 / 2) * t + sigma * np.sqrt(t) * np.tile(z, (1, N))

    return nodes
