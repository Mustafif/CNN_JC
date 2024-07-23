import numpy as np
from scipy.stats import norm
from genhDelta import genhDelta
from Treenodes_ht_HN import Treenodes_ht_HN
from prob_ht import Prob_ht
from prob_Xt import Prob_Xt
from Treenodes_logSt_HN import Treenodes_logSt_HN
from impVol_HN import impVol_HN
# Price the European and American options by the willow tree method and
# Monte Carlo simulation method.

T = 100
N = 100
delta = T/N
h0 = (0.2**2)/252
r = 0.05/250
S0 = 100
K = 100
# All parameters of HN-GARCH model under Q measure
alpha = 1.33e-6
beta = 0.586
gamma = 484.69
omega = 4.96e-6
lambda_ = 1/2
dt = T/N

# generate willow tree
m_h = 6
m_ht = 6
m_x = 30
gamma_h = 0.6
gamma_x = 0.8

# generate MC paths for GARCH model
numPath = 100000
numPoint = N+1
Z = np.random.randn(numPoint+1, numPath)
Z1 = np.random.randn(numPoint, numPath)
ht = np.full((numPoint+1, numPath), np.nan)
ht[0,:] = h0 * np.ones(numPath)
Xt = np.full((numPoint, numPath), np.log(S0))
for i in range(1, numPoint):
    ht[i,:] = omega + alpha*(Z[i-1,:]-gamma*np.sqrt(ht[i-1,:]))**2 + beta*ht[i-1,:]
    Xt[i,:] = Xt[i-1,:] + (r-0.5*ht[i,:]) + np.sqrt(ht[i,:])*Z[i,:]
ht[i+1,:] = omega + alpha*(Z[i,:]-gamma*np.sqrt(ht[i,:]))**2 + beta*ht[i,:]
S = np.exp(Xt)

[hd,qhd] = genhDelta(h0, beta, alpha, gamma, omega, m_h, gamma_h)
nodes_ht, _, _, _, _ = Treenodes_ht_HN(m_ht, hd, qhd, gamma_h,alpha,beta,gamma,omega,N+1)
print(nodes_ht)
[P_ht_N, P_ht] = Prob_ht(nodes_ht,h0,alpha,beta,gamma,omega)

[nodes_Xt,mu,var,k3, k4] = Treenodes_logSt_HN(m_x,gamma_x,r,hd,qhd,S0,alpha,beta,gamma,omega,N)
[q_Xt,P_Xt,tmpHt] = Prob_Xt(nodes_ht,qhd, nodes_Xt, S0,r, alpha,beta,gamma,omega)
nodes_S = np.exp(nodes_Xt)

# 6. Generate Data
A_prices = [] # American option prices
A_sig = [] # Implied volatility
A0_prices = [] # Price of option using model parameters

for i in range(31): 
    cor_p= -1 # Call or Put
    sig, A_price, A0_price = impVol_HN(r, lambda_, omega, beta, alpha, gamma, h0, S0, K, T, N, m_h, m_x, cor_p)
    A_prices.append(A_price)
    A_sig.append(sig)
    A0_prices.append(A0_price)
    
print(A_prices)
print(A_sig)
print(A0_prices)