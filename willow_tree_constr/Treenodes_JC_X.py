import numpy as np
from scipy.stats import norm
from f_hhh import f_hhh
from zq import zq

def Treenodes_JC_X(G, N, M, gamma):
    """
    This function generates an evolution matrix approximating the stochastic process
    of the underlying asset by Johnson Curve when the distribution
    followed by the asset is not normal and lognormal.

    Input:
         G (4*N numpy array): the first four moments matrix of the underlying asset,
                             i.e., G[0,:]: the expectation of the asset at each time step;
                                  G[1,:]: the variance of the asset at each time step;
                                  G[2,:]: the skewness of the asset at each time step;
                                  G[3,:]: the kurtosis of the asset at each time step;
         N (int): total time steps
         M (int): the number of spatial nodes at each time step under the willow tree structure
         gamma (float): an adjustable factor between 0 and 1

    Output:
           nodes (M*N numpy array): underlying asset matrix

    Dependency files:
          zq(M, gamma): generate discrete sampling values of standard normal distribution
          f_hhh(mu, sd, ka3, ka4): generate a required distribution, X, by its first four moments
                                  matrix and a standard normal distribution, Z, by X=c+d*g^{-1}((Z-a)/b),
                                  where a, b, c, d are Johnson Curve parameters, g is a function.
    """
    z, _, _, _ = zq(M, gamma)  # generate discrete sampling values
    nodes = np.zeros((M, N))  # Initialization
    itype = np.zeros(N, dtype=int)

    # Determine the type of Johnson Curve parameters, a, b, c, d and the function, g
    # by f_hhh process
    for i in range(N):
        mu = G[0, i]
        sd = np.sqrt(G[1, i])
        ka3 = G[2, i]
        ka4 = G[3, i]
        a, b, d, c, itype[i], _ = f_hhh(mu, sd)

        # Transform the discrete values of standard normal
        # distribution variable into our required underlying asset
        # by Johnson formula under different cases.
        if itype[i] == 1:  # type 1 lognormal
            u = (z - a) / b
            gi = np.exp(u)
            x = c + d * gi
        elif itype[i] == 2:  # type 2 unbounded
            u = (z - a) / b
            gi = (np.exp(u) - np.exp(-u)) / 2
            x = c + d * gi
        elif itype[i] == 3:  # type 3 bounded
            u = (z - a) / b
            gi = 1 / (1 + np.exp(-u))
            x = c + d * gi
        elif itype[i] == 4:  # type 4 normal
            u = (z - a) / b
            gi = u
            x = c + d * gi
        else:
            x = np.linspace(mu - 4 * sd, mu + 4 * sd, M)
            x = x.reshape(-1)
        nodes[:, i] = np.sort(x)  # generate the i-th column of the underlying asset matrix

    return nodes

