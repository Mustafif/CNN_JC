import numpy as np
from Treenodes_JC_h import Treenodes_JC_h

def Treenodes_ht_HN(m_h, hd, qht, gamma_h, alpha, beta, gamma, omega, N):
    numPoint = N + 1
    
    # Initialize arrays for derivatives of B
    diffB0 = np.zeros(numPoint)
    diffB1 = np.zeros(numPoint)
    diffB2 = np.zeros(numPoint)
    diffB3 = np.zeros(numPoint)
    diffB4 = np.zeros(numPoint)
    
    # Initialize arrays for derivatives of A
    diffA0 = np.zeros(numPoint)
    diffA1 = np.zeros(numPoint)
    diffA2 = np.zeros(numPoint)
    diffA3 = np.zeros(numPoint)
    diffA4 = np.zeros(numPoint)
    
    # Compute derivatives of B
    diffB1[0] = 1
    for row in range(1, numPoint):
        diffB0[row] = beta * diffB0[row-1] + (alpha * gamma**2 * diffB0[row-1]) / (1 - 2 * alpha * diffB0[row-1])
        diffB1[row] = ((beta + alpha * gamma**2 + 4 * alpha * beta * diffB0[row-1] * (-1 + alpha * diffB0[row-1])) * diffB1[row-1]) / (1 - 2 * alpha * diffB0[row-1])**2
        diffB2[row] = beta * diffB2[row-1] + (8 * alpha**3 * gamma**2 * diffB0[row-1] * diffB1[row-1]**2) / (1 - 2 * alpha * diffB0[row-1])**3 + \
                      (2 * alpha**2 * gamma**2 * (2 * diffB1[row-1]**2 + diffB0[row-1] * diffB2[row-1])) / (1 - 2 * alpha * diffB0[row-1])**2 + \
                      alpha * gamma**2 * diffB2[row-1] / (1 - 2 * alpha * diffB0[row-1])
        diffB3[row] = beta * diffB3[row-1] + 48 * alpha**4 * gamma**2 * diffB0[row-1] * diffB1[row-1]**3 / (1 - 2 * alpha * diffB0[row-1])**4 + \
                      (24 * alpha**3 * gamma**2 * diffB1[row-1]**3 + 24 * alpha**3 * gamma**2 * diffB0[row-1] * diffB1[row-1] * diffB2[row-1]) / (1 - 2 * alpha * diffB0[row-1])**3 + \
                      (12 * alpha**2 * gamma**2 * diffB1[row-1] * diffB2[row-1] + 2 * alpha**2 * gamma**2 * diffB0[row-1] * diffB3[row-1]) / (1 - 2 * alpha * diffB0[row-1])**2 + \
                      alpha * gamma**2 * diffB3[row-1] / (1 - 2 * alpha * diffB0[row-1])
        diffB4[row] = beta * diffB4[row-1] + 384 * alpha**5 * gamma**2 * diffB0[row-1] * diffB1[row-1]**4 / (1 - 2 * alpha * diffB0[row-1])**5 + \
                      (192 * alpha**4 * gamma**2 * diffB1[row-1]**4 + 288 * alpha**4 * gamma**2 * diffB0[row-1] * diffB1[row-1]**2 * diffB2[row-1]) / (1 - 2 * alpha * diffB0[row-1])**4 + \
                      (144 * alpha**3 * gamma**2 * diffB1[row-1]**2 * diffB2[row-1] + 24 * alpha**3 * gamma**2 * diffB0[row-1] * diffB2[row-1]**2 + 32 * alpha**3 * gamma**2 * diffB0[row-1] * diffB1[row-1] * diffB3[row-1]) / (1 - 2 * alpha * diffB0[row-1])**3 + \
                      (12 * alpha**2 * gamma**2 * diffB2[row-1]**2 + 16 * alpha**2 * gamma**2 * diffB1[row-1] * diffB3[row-1] + 2 * alpha**2 * gamma**2 * diffB0[row-1] * diffB4[row-1]) / (1 - 2 * alpha * diffB0[row-1])**2 + \
                      alpha * gamma**2 * diffB4[row-1] / (1 - 2 * alpha * diffB0[row-1])

    # Compute derivatives of A
    for row in range(1, numPoint):
        diffA0[row] = diffA0[row-1] + omega * diffB0[row-1] - 0.5 * np.log(1 - 2 * alpha * diffB0[row-1])
        diffA1[row] = diffA1[row-1] + (omega + alpha / (1 - 2 * alpha * diffB0[row-1])) * diffB1[row-1]
        diffA2[row] = (2 * alpha**2 * diffB1[row-1]**2) / (1 - 2 * alpha * diffB0[row-1])**2 + diffA2[row-1] + \
                      (omega + alpha / (1 - 2 * alpha * diffB0[row-1])) * diffB2[row-1]
        diffA3[row] = (8 * alpha**3 * diffB1[row-1]**3) / (1 - 2 * alpha * diffB0[row-1])**3 + \
                      (6 * alpha**2 * diffB1[row-1] * diffB2[row-1]) / (1 - 2 * alpha * diffB0[row-1])**2 + \
                      diffA3[row-1] + (omega + alpha / (1 - 2 * alpha * diffB0[row-1])) * diffB3[row-1]
        diffA4[row] = diffA4[row-1] + omega * diffB4[row-1] + (48 * alpha**4 * diffB1[row-1]**4) / (1 - 2 * alpha * diffB0[row-1])**4 + \
                      (48 * alpha**3 * diffB1[row-1]**2 * diffB2[row-1]) / (1 - 2 * alpha * diffB0[row-1])**3 + \
                      (6 * alpha**2 * diffB2[row-1]**2 + 8 * alpha**2 * diffB1[row-1] * diffB3[row-1]) / (1 - 2 * alpha * diffB0[row-1])**2 + \
                      alpha * diffB4[row-1] / (1 - 2 * alpha * diffB0[row-1])

    # Compute derivatives of m.g.f
    tmp1 = np.outer(np.ones(m_h), diffA0[1:]) + np.outer(hd, diffB0[1:])
    tmp2 = np.outer(np.ones(m_h), diffA1[1:]) + np.outer(hd, diffB1[1:])
    tmp3 = np.outer(np.ones(m_h), diffA2[1:]) + np.outer(hd, diffB2[1:])
    tmp4 = np.outer(np.ones(m_h), diffA3[1:]) + np.outer(hd, diffB3[1:])

    diffmgf1 = np.exp(tmp1) * tmp2
    diffmgf2 = np.exp(tmp1) * (tmp2**2 + tmp3)
    diffmgf3 = np.exp(tmp1) * (tmp2**3 + 3 * tmp2 * tmp3 + tmp4)
    diffmgf4 = np.exp(tmp1) * (tmp2**4 + 6 * tmp2**2 * tmp3 + 3 * tmp3**2 + 4 * tmp2 * tmp4 + np.outer(np.ones(m_h), diffA4[1:]) + np.outer(hd, diffB4[1:]))

    # Compute moments
    mom1 = np.dot(qht.T, diffmgf1[:, 1:])
    mom2 = np.dot(qht.T, diffmgf2[:, 1:])
    mom3 = np.dot(qht.T, diffmgf3[:, 1:])
    mom4 = np.dot(qht.T, diffmgf4[:, 1:])

    # Compute statistics
    mu = mom1
    var = mom2 - mu**2
    temp = np.sqrt(var)
    k3 = (mom3 - 3 * mu * mom2 + 2 * mu**3) / temp**3
    k4 = (mom4 - 4 * mu * mom3 + 6 * mu**2 * mom2 - 3 * mu**4) / temp**4

    G = np.vstack((mu, var, k3, k4))

    # Compute tree nodes (you need to implement Treenodes_JC_h function)
    nodes_ht = Treenodes_JC_h(G, N, m_h, gamma_h)

    return nodes_ht, mu, var, k3, k4
