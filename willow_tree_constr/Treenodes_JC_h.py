import numpy as np
from f_hhh import f_hhh
from zq import zq

def Treenodes_JC_h(G, N, M, gamma):
    """
    Generates a evolution matrix approximating the stochastic process
    of the underlying asset by Johnson Curve when the distribution
    followed by the asset is not normal and lognormal.

    Parameters:
    G (4xN matrix): The first four moments matrix of the underlying asset
    N (int): Total time steps
    M (int): Number of spatial nodes at each time step under the willow tree structure
    gamma (float): An adjustable factor between 0 and 1

    Returns:
    nodes (MxN matrix): Underlying asset matrix
    q (array): Probabilities associated with the nodes
    """
    z, q, _, _ = zq(M, gamma)  # Generate discrete sampling values
    nodes = np.zeros((M, N))  # Initialization
    itype = np.zeros(N, dtype=int)

    for i in range(N-1):
        mu = G[0, i]
        sd = np.sqrt(G[1, i])
        ka3 = G[2, i]
        ka4 = G[3, i]
        a, b, d, c, itype[i], _ = f_hhh(mu, sd, ka3, ka4)

        if itype[i] == 1:  # Type 1 lognormal
            u = (z - a) / b
            gi = np.exp(u)
            x = c + d * gi
        elif itype[i] == 2:  # Type 2 unbounded
            u = (z - a) / b
            gi = (np.exp(u) - np.exp(-u)) / 2
            x = c + d * gi
        elif itype[i] == 3:  # Type 3 bounded
            u = (z - a) / b
            gi = 1 / (1 + np.exp(-u))
            x = c + d * gi
        elif itype[i] == 4:  # Type 4 normal
            u = (z - a) / b
            gi = u
            x = c + d * gi
        else:
            x = np.linspace(mu - 4*sd, mu + 4*sd, M)

        if np.any(x < 0):
            x = np.linspace(nodes[0, i-1]*0.9, nodes[-1, i-1]*1.1, M)

        nodes[:, i] = np.sort(x)  # Generate the i-th column of the underlying asset matrix

    return nodes, q