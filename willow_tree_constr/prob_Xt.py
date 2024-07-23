import numpy as np
from scipy.stats import norm

def Prob_Xt(nodes_ht, q_ht, nodes_Xt, S0, r, alpha, beta, gamma, omega):
    """
    This function computes the transition probabilities X_t of the Heston-Nadi GARCH model and
    constructs all tree nodes of X_t and h_t.

    Parameters:
    nodes_ht (numpy.ndarray): Tree nodes of ht
    q_ht (numpy.ndarray): Probabilities of h_1 given h0
    nodes_Xt (numpy.ndarray): Tree nodes of Xt
    S0 (float): Initial value of St
    r (float): Interest rate
    alpha (float): Parameter for Heston-Nadi GARCH
    beta (float): Parameter for Heston-Nadi GARCH
    gamma (float): Parameter for Heston-Nadi GARCH
    omega (float): Parameter for Heston-Nadi GARCH

    Returns:
    q_Xt (numpy.ndarray): Transition probabilities from X1 to X0
    P_Xt_N (numpy.ndarray): Transition probability matrix of Xt, 3-d array
    tmpHt (numpy.ndarray): Temporary values of ht
    """

    X0 = np.log(S0)
    m_h = nodes_ht.shape[0]
    X_len, N = nodes_Xt.shape

    # Compute q_Xt for t=1
    Xt = nodes_Xt[:, 0]
    m_x = X_len
    cur_ht = nodes_ht[:, 0]

    q_Xt = np.zeros(m_x)
    mu = r - 0.5 * cur_ht
    std = np.sqrt(cur_ht)
    dx = Xt - X0
    intX = np.concatenate(([-np.inf], (dx[:-1] + dx[1:]) / 2, [np.inf]))
    tmpP = np.zeros((m_h, m_x))
    Ph = np.zeros((m_h, m_x, N))
    Ph_XXh_h = np.zeros((m_h, m_x, m_x))
    next_ht = nodes_ht[:, 1]

    for i in range(m_h):
        # compute P(X_i^1|X^0)
        p = norm.cdf(intX, mu[i], std[i])
        p = np.diff(p)
        tmpP[i, :] = p

        # compute P(h_j^2|X_i^1,X^0)
        ht = cur_ht[i]
        z = np.concatenate(([-np.inf], (intX[1:-1] - r + 0.5 * ht) / np.sqrt(ht), [np.inf]))
        intH = np.concatenate(([omega + beta * ht + 1e-16], (next_ht[:-1] + next_ht[1:]) / 2, [1]))
        zhup1 = np.real(-np.sqrt(intH[:-1] - omega - beta * ht) / np.sqrt(alpha) + gamma * np.sqrt(ht))
        zhlow1 = np.real(-np.sqrt(intH[1:] - omega - beta * ht) / np.sqrt(alpha) + gamma * np.sqrt(ht))
        zhup2 = np.real(np.sqrt(intH[1:] - omega - beta * ht) / np.sqrt(alpha) + gamma * np.sqrt(ht))
        zhlow2 = np.real(np.sqrt(intH[:-1] - omega - beta * ht) / np.sqrt(alpha) + gamma * np.sqrt(ht))

        for j in range(m_x):
            tmplow1 = np.maximum(zhlow1, z[j])
            tmplow2 = np.maximum(zhlow2, z[j])
            tmpup1 = np.minimum(zhup1, z[j + 1])
            tmpup2 = np.minimum(zhup2, z[j + 1])

            # compute P(h_k^2|X_j^1, X^0, h_i^1)
            ph1 = np.maximum(norm.cdf(tmpup1) - norm.cdf(tmplow1), 0) + np.maximum(norm.cdf(tmpup2) - norm.cdf(tmplow2), 0)
            ph1 = ph1 / tmpP[i, j]
            Ph_XXh_h[:, j, i] = ph1

    q_Xt = np.dot(q_ht.T, tmpP)
    q_Xt = q_Xt.reshape(-1)

    for i in range(m_h):
        Ph_YY = tmpP[i, :] * q_ht[i] / q_Xt
        Ph[:, :, 0] += Ph_XXh_h[:, :, i] * np.tile(Ph_YY, (m_h, 1))

    for i in range(m_x):
        Ph[:, i, 0] /= np.sum(Ph[:, i, 0])

    P_Xt = np.zeros((m_x, N))
    P_Xt[:, 0] = q_Xt
    tmpHt = np.zeros((m_h, N))
    tmpHt[:, 0] = np.dot(Ph[:, :, 0], P_Xt[:, 0])

    # compute transition probability matrices [p_ij]^n
    P_Xt_N = np.zeros((m_x, m_x, N - 1))

    for n in range(N - 1):
        next_ht = nodes_ht[:, n + 2]
        cur_ht = nodes_ht[:, n + 1]
        Xt = nodes_Xt[:, n + 1]
        mu = r - 0.5 * cur_ht
        std = np.sqrt(cur_ht)
        Ph_XXX = np.zeros((m_h, m_x, m_x, m_h))
        tmpP = np.zeros((m_h, m_x, m_x))

        for i in range(m_x):  # X_i^n
            cur_Xt = nodes_Xt[i, n]
            dx = Xt - cur_Xt
            intX = np.concatenate(([-1000], (dx[:-1] + dx[1:]) / 2, [1000]))

            for j in range(m_h):  # h_j^n+1
                # compute P(X^n+1|X_i^n, h_j^n+1)
                p = norm.cdf(intX, mu[j], std[j])
                p = np.diff(p)
                tmpP[j, :, i] = p

                # compute P(h^n+2)|X^n+1, X_i^n)
                ht = cur_ht[j]
                z = np.concatenate(([-1000], (intX[1:-1] - r + 0.5 * ht) / np.sqrt(ht), [1000]))
                intH = np.concatenate(([omega + beta * ht + 1e-16], (next_ht[:-1] + next_ht[1:]) / 2, [1]))
                zhup1 = np.real(-np.sqrt(intH[:-1] - omega - beta * ht) / np.sqrt(alpha) + gamma * np.sqrt(ht))
                zhlow1 = np.real(-np.sqrt(intH[1:] - omega - beta * ht) / np.sqrt(alpha) + gamma * np.sqrt(ht))
                zhup2 = np.real(np.sqrt(intH[1:] - omega - beta * ht) / np.sqrt(alpha) + gamma * np.sqrt(ht))
                zhlow2 = np.real(np.sqrt(intH[:-1] - omega - beta * ht) / np.sqrt(alpha) + gamma * np.sqrt(ht))

                for k in range(m_x):  # X_j^n+1
                    tmplow1 = np.maximum(zhlow1, z[k])
                    tmplow2 = np.maximum(zhlow2, z[k])
                    tmpup1 = np.minimum(zhup1, z[k + 1])
                    tmpup2 = np.minimum(zhup2, z[k + 1])

                    # compute P(h^n+2|X_k^n+1, X_i^n, h_j^n+1)
                    id1 = tmplow1 < tmpup1
                    id2 = tmplow2 < tmpup2

                    if tmpP[j, k, i] < 1e-4:  # tmpP(j,k) == 0
                        num = np.sum(id1) + np.sum(id2)
                        Ph_XXX[id1, k, i, j] = 1 / num
                        Ph_XXX[id2, k, i, j] = 1 / num
                    else:
                        ph1 = np.zeros(m_h)
                        ph1[id1] = norm.cdf(tmpup1[id1]) - norm.cdf(tmplow1[id1])
                        ph1[id2] += norm.cdf(tmpup2[id2]) - norm.cdf(tmplow2[id2])
                        tmp = ph1 / tmpP[j, k, i]
                        Ph_XXX[:, k, i, j] = tmp / np.sum(tmp)

            tmp = np.dot(Ph[:, i, n].T, tmpP[:, :, i])
            P_Xt_N[i, :, n] = tmp / np.sum(tmp)

        P_Xt[:, n + 1] = np.dot(P_Xt[:, n].T, P_Xt_N[:, :, n])

        Ph_XXh_h = np.zeros((m_h, m_x, m_x))
        for e in range(m_x):
            tmpP[:, :, e] = tmpP[:, :, e] * np.tile(Ph[:, e, n], (m_h, 1))

        for e in range(m_h):
            Ph_XXh_h += Ph_XXX[:, :, :, e] * np.tile(tmpP[e, :, :], (m_h, 1, 1))

        for d in range(m_x):
            tmp = P_Xt[:, n] / P_Xt[d, n + 1]
            sumtmp = np.sum(Ph_XXh_h[:, d, :] * tmp, axis=1)
            Ph[:, d, n + 1] = sumtmp

        tmpHt[:, n + 1] = np.dot(Ph[:, :, n + 1], P_Xt[:, n + 1])

    return q_Xt, P_Xt_N, tmpHt
