import numpy as np
from zq import zq
from gen_PoWiner import gen_powiener as gen_PoWiner 
from nodes_Winer import nodes_wiener as nodes_Winer 
from American import american_option_price

def impvol(S0, K, T, r, V, index, N, m, gamma, tol=1e-5, itmax=100):
    """
    Compute implied volatilities of American options via Willow Tree and bisection.

    Parameters:
        S0 : float - Initial stock price
        K  : ndarray - Strike price matrix (k x n)
        T  : ndarray - Maturities (length n)
        r  : float - Risk-free interest rate
        V  : ndarray - Observed American option prices (k x n)
        index : ndarray - +1 for call, -1 for put (k x n)
        N  : int - Number of time steps in the tree
        m  : int - Number of nodes per time step
        gamma : float - Sampling parameter for z
        tol : float - Tolerance for bisection
        itmax : int - Maximum iterations

    Returns:
        imp_vol : ndarray - Implied volatilities (k x n)
    """
    n = len(T)
    k = K.shape[0]
    z, _, _, _ = zq(m, gamma)
    imp_vol = np.zeros((k, n))

    for i in range(n):  # for each maturity
        P, q = gen_PoWiner(T[i], N, z)

        for j in range(k):
            V0 = 1e10
            it = 0
            a, b = 0.0, 1.0
            sigma = 0.5 * (a + b)

            while abs(V0 - V[j, i]) > tol and it < itmax:
                Xnodes = nodes_Winer(T[i], N, z, r, sigma)
                nodes = S0 * np.exp(Xnodes)
                V0, _ = american_option_price(nodes, P, q, r, T[i], S0, K[j, i], index[j, i])

                if V0 > V[j, i]:
                    b = sigma
                else:
                    a = sigma

                sigma = 0.5 * (a + b)
                it += 1

            imp_vol[j, i] = sigma

    return imp_vol

# ==== TEST CASE ====
S0 = 100
K = np.array([[90], [100], [110]])  # Strike matrix (3x1)
T = np.array([1.0])                 # One maturity
r = 0.05
index = np.full((3, 1), -1)         # Put options
true_sigma = 0.25
N = 50
m = 20
gamma = 3.0

# Generate true option prices
z, _, _, _ = zq(m, gamma)
P, q = gen_PoWiner(T[0], N, z)
Xnodes = nodes_Winer(T[0], N, z, r, true_sigma)
nodes = S0 * np.exp(Xnodes)
V = np.zeros((3, 1))
for i in range(3):
    V[i, 0], _ = american_option_price(nodes, P, q, r, T[0], S0, K[i, 0], index[i, 0])

# Run implied vol solver
implied_vols = impvol(S0, K, T, r, V, index, N, m, gamma)
print("Implied volatilities:")
print(implied_vols)