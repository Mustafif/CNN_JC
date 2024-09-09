import numpy as np

class Heston:
    def __init__(self, params, r, eta2=0) -> None:
        self.r = r
        omega, alpha, beta, gamma, lambda_ = params
        # each parameter will be stored as an array where the first
        # is the param under P measure and the second is under Q measure
        self.omega = [omega, omega/(1-2*alpha*eta2)]
        self.alpha = [alpha, alpha/(1-2*alpha*eta2)**2]
        self.beta = beta
        self.gamma = [gamma, gamma*(1-2*alpha*eta2)]
        self.lambda_ = [lambda_, lambda_*(1-2*alpha*eta2)]
        self.rho = self.lambda_[1] + self.gamma[1] + 0.5

    def loglikelihood_returns(self, S):
        y = np.log(S[1:] / S[:-1])
        n = len(y)
        eps = np.zeros(n)
        h = np.zeros(n+1)
        h[0] = np.var(y)

        for i in range(n):
            eps[i] = (y[i] - (self.r + self.lambda_[0]*h[i]))/np.sqrt(h[i])
            h[i + 1] = (self.omega[0] + self.beta * h[i] +
                                    self.alpha[0] * ((y[i] - (self.r - self.lambda_[0] * h[i])) / np.sqrt(h[i]) -
                                                     self.gamma[0] * np.sqrt(h[i])) ** 2)

        loss = -0.5 * np.sum(np.log(h[1:]) + (y - (self.r + self.lambda_[0] * h[1:])) ** 2 / h[1:])

        return loss

    def loglikelihood_options(self, V, sigma):
       # needs to be figured out
       return 0.0

    def joint_loglikelihood(self, S, V, sigma):
        N = len(S) - 1
        M = len(V)
        y1 = self.loglikelihood_returns(S)
        y2 = self.loglikelihood_options(V, sigma)
        return ((N+M)/2*N)*y1 + ((N+M)/2*M)*y2

# heston = Heston([0.01, 0.1, 0.9, 0.1, 0.1], 0.05)
# print(heston.loglikelihood_returns(np.linspace(2, 4, 1000)))
