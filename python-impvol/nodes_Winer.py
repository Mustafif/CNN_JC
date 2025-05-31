import numpy as np

def nodes_wiener(T, N, z, r, sigma):
    """
    Construct a willow tree for standard Brownian motion with maturity T over N time steps.

    Parameters:
        T (float): Maturity
        N (int): Number of time steps
        z (np.ndarray): m x 1 array of standard normal quantile points (must be sorted ascending)
        r (float): Interest rate
        sigma (float): Volatility of the stock

    Returns:
        nodes (np.ndarray): m x N array of tree nodes for Brownian motion
    """
    z = np.asarray(z).reshape(-1, 1)  # Ensure column vector
    m = z.shape[0]
    dt = T / N
    tt = np.linspace(dt, T, N)  # shape (N,)

    drift = (r - 0.5 * sigma ** 2) * tt
    diffusion = sigma * np.sqrt(tt)

    nodes = drift + z @ diffusion[np.newaxis, :]  # outer product to get m x N

    return nodes