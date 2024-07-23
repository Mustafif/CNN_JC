import numpy as np
from Treenodes_JC_X import Treenodes_JC_X


def Treenodes_logSt_HN(m_x, gamma_x, r, hd, qht, S0, alpha, beta, gamma, omega, N):
    """
    Compute the first four moments of X_t of the Heston-Nadi GARCH model and construct all tree nodes of X_t

    Parameters:
    m_x (int): number of tree nodes
    gamma_x (float): parameter for generating z_t
    r (float): interest rate
    hd (list or np.array): discrete values of h_1
    qht (list or np.array): corresponding probabilities of hd
    S0 (float): initial stock price
    alpha (float): parameter for Heston-Nadi GARCH
    beta (float): parameter for Heston-Nadi GARCH
    gamma (float): parameter for Heston-Nadi GARCH
    omega (float): parameter for Heston-Nadi GARCH
    N (int): number of time steps

    Returns:
    nodes_Xt (np.array): tree nodes of X_t
    mu (float): mean of X_t
    var (float): variance of X_t
    k3 (float): third cumulant of X_t
    k4 (float): fourth cumulant of X_t
    """
    numPoint = N + 1
    u = 0

    # Initialize arrays for derivatives of B
    diffB0 = np.zeros(numPoint)
    diffB1 = np.zeros(numPoint)
    diffB2 = np.zeros(numPoint)
    diffB3 = np.zeros(numPoint)
    diffB4 = np.zeros(numPoint)

    # Compute diffB0
    for row in range(1, numPoint):
        diffB0[row] = (
            -0.5 * u
            + beta * diffB0[row - 1]
            + u**2 / (2 * (1 - 2 * alpha * diffB0[row - 1]))
            + (
                alpha * gamma**2 * diffB0[row - 1]
                - 2 * u * alpha * gamma * diffB0[row - 1]
            )
            / (1 - 2 * alpha * diffB0[row - 1])
        )

    # Compute diffB1
    for row in range(1, numPoint):
        diffB1[row] = (
            -0.5
            + beta * diffB1[row - 1]
            + (
                u**2 * alpha * diffB1[row - 1]
                + 2 * alpha**2 * gamma**2 * diffB0[row - 1] * diffB1[row - 1]
                - 4 * u * alpha**2 * gamma * diffB0[row - 1] * diffB1[row - 1]
            )
            / (1 - 2 * alpha * diffB0[row - 1]) ** 2
            + (
                alpha * gamma**2 * diffB1[row - 1]
                - 2 * alpha * gamma * diffB0[row - 1]
                - 2 * u * alpha * gamma * diffB1[row - 1]
                + u
            )
            / (1 - 2 * alpha * diffB0[row - 1])
        )

    # Compute diffB2
    for row in range(1, numPoint):
        diffB2[row] = (
            beta * diffB2[row - 1]
            + (
                4 * u**2 * alpha**2 * diffB1[row - 1] ** 2
                + 8 * alpha**3 * gamma**2 * diffB0[row - 1] * diffB1[row - 1] ** 2
                - 16 * u * alpha**3 * gamma * diffB0[row - 1] * diffB1[row - 1] ** 2
            )
            / (1 - 2 * alpha * diffB0[row - 1]) ** 3
            + (
                2 * u * alpha * diffB1[row - 1]
                + u**2 * alpha * diffB2[row - 1]
                + 4 * alpha**2 * gamma**2 * diffB1[row - 1] ** 2
                + 2 * alpha**2 * gamma**2 * diffB0[row - 1] * diffB2[row - 1]
                - 8 * alpha**2 * gamma * diffB0[row - 1] * diffB1[row - 1]
            )
            / (1 - 2 * alpha * diffB0[row - 1])
        )

        # Compute diffB3
    for row in range(1, numPoint):
        diffB3[row] = (
            beta * diffB3[row - 1]
            + (
                24 * u * alpha**3 * diffB1[row - 1] ** 3
                + 48 * alpha**4 * gamma**2 * diffB0[row - 1] * diffB1[row - 1] ** 3
                - 96 * u * alpha**4 * gamma * diffB0[row - 1] * diffB1[row - 1] ** 3
            )
            / (1 - 2 * alpha * diffB0[row - 1]) ** 4
            + (
                16 * u * alpha**2 * diffB1[row - 1] ** 2
                + 12 * u**2 * alpha**2 * diffB1[row - 1] * diffB2[row - 1]
                + 24 * alpha**3 * gamma**2 * diffB1[row - 1] ** 3
                - 48 * alpha**3 * gamma * diffB0[row - 1] * diffB1[row - 1] ** 2
                - 48 * u * alpha**3 * gamma * diffB1[row - 1] ** 3
                - 48
                * u
                * alpha**3
                * gamma
                * diffB0[row - 1]
                * diffB1[row - 1]
                * diffB2[row - 1]
            )
            / (1 - 2 * alpha * diffB0[row - 1]) ** 3
            + (
                4 * alpha * diffB1[row - 1]
                + 6 * u * alpha * diffB2[row - 1]
                + u**2 * alpha * diffB3[row - 1]
                + 12 * alpha**2 * gamma**2 * diffB1[row - 1] * diffB2[row - 1]
                + 2 * alpha**2 * gamma**2 * diffB3[row - 1] * diffB0[row - 1]
                - 24 * alpha**2 * gamma * diffB1[row - 1] ** 2
                - 12 * alpha**2 * gamma * diffB0[row - 1] * diffB2[row - 1]
                - 24 * u * alpha**2 * gamma * diffB1[row - 1] * diffB2[row - 1]
                - 4 * u * alpha**2 * gamma * diffB0[row - 1] * diffB3[row - 1]
            )
            / (1 - 2 * alpha * diffB0[row - 1]) ** 2
            + (
                alpha * gamma**2 * diffB3[row - 1]
                - 6 * alpha * gamma * diffB2[row - 1]
                - 2 * u * alpha * gamma * diffB3[row - 1]
            )
            / (1 - 2 * alpha * diffB0[row - 1])
        )

    for row in range(1, numPoint):
        diffB4[row] = (
            beta * diffB4[row - 1]
            + (
                192 * u**2 * alpha**4 * diffB1[row - 1] ** 4
                + 384 * alpha**5 * gamma**2 * diffB0[row - 1] * diffB1[row - 1] ** 4
                - 768 * u * alpha**5 * gamma * diffB0[row - 1] * diffB1[row - 1] ** 4
            )
            / (1 - 2 * alpha * diffB0[row - 1]) ** 5
            + (
                144 * u * alpha**3 * diffB1[row - 1] ** 3
                + 144 * u**2 * alpha**3 * diffB1[row - 1] ** 2 * diffB2[row - 1]
                + 192 * alpha**4 * gamma**2 * diffB1[row - 1] ** 4
                + 288
                * alpha**4
                * gamma**2
                * diffB0[row - 1]
                * diffB1[row - 1] ** 2
                * diffB2[row - 1]
                - 384 * alpha**4 * gamma * diffB0[row - 1] * diffB1[row - 1] ** 3
                - 96 * u * alpha**4 * gamma * diffB1[row - 1] ** 4
                - 288
                * u
                * alpha**4
                * gamma
                * diffB0[row - 1]
                * diffB1[row - 1] ** 2
                * diffB2[row - 1]
            )
            / (1 - 2 * alpha * diffB0[row - 1]) ** 4
            + (
                32 * alpha**2 * diffB1[row - 1] ** 2
                + 80 * u * alpha**2 * diffB1[row - 1] * diffB2[row - 1]
                + 12 * u**2 * alpha**2 * diffB2[row - 1] ** 2
                + 16 * u**2 * alpha**2 * diffB1[row - 1] * diffB3[row - 1]
                + 144 * alpha**3 * gamma**2 * diffB1[row - 1] ** 2 * diffB2[row - 1]
                + 24 * alpha**3 * gamma**2 * diffB0[row - 1] * diffB2[row - 1] ** 2
                + 32
                * alpha**3
                * gamma**2
                * diffB0[row - 1]
                * diffB1[row - 1]
                * diffB3[row - 1]
                - 192 * alpha**3 * gamma * diffB1[row - 1] ** 3
                - 192
                * alpha**3
                * gamma
                * diffB0[row - 1]
                * diffB1[row - 1]
                * diffB2[row - 1]
                - 192 * u * alpha**3 * gamma * diffB1[row - 1] ** 2 * diffB2[row - 1]
                - 48 * u * alpha**3 * gamma * diffB0[row - 1] * diffB2[row - 1] ** 2
                - 48
                * u
                * alpha**3
                * gamma
                * diffB0[row - 1]
                * diffB1[row - 1]
                * diffB3[row - 1]
            )
            / (1 - 2 * alpha * diffB0[row - 1]) ** 3
            + (
                10 * alpha * diffB2[row - 1]
                + 6 * u * alpha * diffB3[row - 1]
                + 12 * alpha**2 * gamma**2 * diffB2[row - 1] ** 2
                + 16 * alpha**2 * gamma**2 * diffB1[row - 1] * diffB3[row - 1]
                + 2 * alpha**2 * gamma**2 * diffB0[row - 1] * diffB4[row - 1]
                - 96 * alpha**2 * gamma * diffB1[row - 1] * diffB2[row - 1]
                - 16 * alpha**2 * gamma * diffB0[row - 1] * diffB3[row - 1]
                - 24 * u * alpha**2 * gamma * diffB2[row - 1] ** 2
                - 32 * u * alpha**2 * gamma * diffB1[row - 1] * diffB3[row - 1]
                - 4 * u * alpha**2 * gamma * diffB0[row - 1] * diffB4[row - 1]
                + 2 * u * alpha * diffB3[row - 1]
                + u**2 * alpha * diffB4[row - 1]
            )
            / (1 - 2 * alpha * diffB0[row - 1]) ** 2
            + (
                alpha * gamma**2 * diffB4[row - 1]
                - 8 * alpha * gamma * diffB3[row - 1]
                - 2 * u * alpha * gamma * diffB4[row - 1]
            )
            / (1 - 2 * alpha * diffB0[row - 1])
        )

    # Recursively compute derivatives of A at u=0
    # 0th order
    diffA0 = np.zeros(numPoint)
    for row in range(1, numPoint):
        diffA0[row] = (
            diffA0[row - 1]
            + u * r
            + diffB0[row - 1] * omega
            - 0.5 * np.log(1 - 2 * diffB0[row - 1] * alpha)
        )

    # 1st order
    diffA1 = np.zeros(numPoint)
    for row in range(1, numPoint):
        diffA1[row] = (
            r
            + diffA1[row - 1]
            + (omega + (alpha) / (1 - 2 * alpha * diffB0[row - 1])) * diffB1[row - 1]
        )

    # 2nd order
    diffA2 = np.zeros(numPoint)
    for row in range(1, numPoint):
        diffA2[row] = (
            (2.0 * alpha**2 * diffB1[row - 1] ** 2)
            / (1 - 2 * alpha * diffB0[row - 1]) ** 2
            + diffA2[row - 1]
            + (omega + (1.0 * alpha) / (1 - 2 * alpha * diffB0[row - 1]))
            * diffB2[row - 1]
        )

    # 3rd order
    diffA3 = np.zeros(numPoint)
    for row in range(1, numPoint):
        diffA3[row] = (
            (8.0 * alpha**3 * diffB1[row - 1] ** 3)
            / (1 - 2 * alpha * diffB0[row - 1]) ** 3
            + (6.0 * alpha**2 * diffB1[row - 1] * diffB2[row - 1])
            / (1 - 2 * alpha * diffB0[row - 1]) ** 2
            + diffA3[row - 1]
            + (omega + (1.0 * alpha) / (1 - 2 * alpha * diffB0[row - 1]))
            * diffB3[row - 1]
        )

    # 4th order
    diffA4 = np.zeros(numPoint)
    for row in range(1, numPoint):
        diffA4[row] = (
            diffA4[row - 1]
            + omega * diffB4[row - 1]
            + (48 * alpha**4 * diffB1[row - 1] ** 4)
            / (1 - 2 * alpha * diffB0[row - 1]) ** 4
            + (48 * alpha**3 * diffB1[row - 1] ** 2 * diffB2[row - 1])
            / (1 - 2 * alpha * diffB0[row - 1]) ** 3
            + (
                6 * alpha**2 * diffB2[row - 1] ** 2
                + 8 * alpha**2 * diffB1[row - 1] * diffB3[row - 1]
            )
            / (1 - 2 * alpha * diffB0[row - 1]) ** 2
            + alpha * diffB4[row - 1] / (1 - 2 * alpha * diffB0[row - 1])
        )

    # Recursively compute derivatives of m.g.f at u=0.
    # 0th order
    # diffmgf0 = exp(diffA0 + hd.*diffB0).*S0.^u;
    m_h = len(hd)
    tmp1 = np.ones((m_h, 1)) * diffA0.T + hd * diffB0.T
    tmp2 = np.ones((m_h, 1)) * diffA1.T + hd * diffB1.T
    tmp3 = np.ones((m_h, 1)) * diffA2.T + hd * diffB2.T
    tmp4 = np.ones((m_h, 1)) * diffA3.T + hd * diffB3.T

    # 1st order
    diffmgf1 = np.exp(tmp1) * S0**u * (np.log(S0) + tmp2)

    # 2nd order
    diffmgf2 = (
        np.exp(tmp1)
        * S0**u
        * (np.log(S0) ** 2 + 2 * (tmp2) * np.log(S0) + (tmp2) ** 2 + tmp3)
    )

    # 3rd order
    diffmgf3 = (
        np.exp(tmp1)
        * S0**u
        * ((np.log(S0) + tmp2) ** 3 + 3 * (np.log(S0) + tmp2) * tmp3 + tmp4)
    )

    # 4th order
    diffmgf4 = (
        np.exp(tmp1)
        * S0**u
        * (
            (np.log(S0) + tmp2) ** 4
            + 6 * (np.log(S0) + tmp2) ** 2 * tmp3
            + 3 * tmp3**2
            + 4 * (np.log(S0) + tmp2) * tmp4
            + np.ones((m_h, 1)) * diffA4.T
            + hd * diffB4.T
        )
    )

    mom1 = qht.T @ diffmgf1[:, 1:]
    mom2 = qht.T @ diffmgf2[:, 1:]
    mom3 = qht.T @ diffmgf3[:, 1:]
    mom4 = qht.T @ diffmgf4[:, 1:]

    # generate nodes by the fourth order moments
    mu = mom1
    var = mom2 - mu**2
    temp = np.sqrt(var)
    k3 = 1 / (temp**3) * (mom3 - 3 * mu * mom2 + 2 * mu**3)
    k4 = 1 / (temp**4) * (mom4 - 4 * mu * mom3 + 6 * (mu**2) * mom2 - 3 * mu**4)
    tmp3 = k3[:, 0]
    tmp4 = k4[:, 0]
    k33 = np.repeat(tmp3[:, None], N, axis=1)
    k44 = np.repeat(tmp4[:, None], N, axis=1)

    G = np.vstack((mu, var, k33, k44))
    nodes_Xt = Treenodes_JC_X(G, N, m_x, gamma_x)

    return nodes_Xt, mu, var, k3, k4


# # Example usage:
# m_x = 5
# gamma_x = 0.1
# r = 0.05
# hd = np.array([0.1, 0.2, 0.3])
# qht = np.array([0.3, 0.4, 0.3])
# S0 = 100
# alpha = 0.1
# beta = 0.85
# gamma = 0.1
# omega = 0.05
# N = 10

# nodes_Xt, mu, var, k3, k4 = tree_nodes_logSt_HN_New(m_x, gamma_x, r, hd, qht, S0, alpha, beta, gamma, omega, N)
# print(nodes_Xt, mu, var, k3, k4)
