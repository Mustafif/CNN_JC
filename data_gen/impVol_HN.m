function [sig,V,V0] = impVol_HN(r, lambda,w, beta, alpha, gamma,h0,S0, K, T, N, m_h, m_x, CorP)
%  
% Compute implied volatilty of American options with underlying in HN-GARCH
% as under P measure
%       X_t = r+lambda*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = omega +alpha(z_t-gamma*sqrt(h_t))^2+beta h_t
%
%
% INPUT
%  r -- risk free interest rate
%  lambda,w, alpha, gamma -- parameters of HN-GARCH model
%  h0 -- initial value of ht
%  S0 -- initial stock price
%  K -- stricke price
%  T -- maturity
%  N -- number of time steps
%  m_h -- # of possible values of h_t for willow tree
%  m_x -- # of possible values of X_t for willow tree
%  CorP -- Call or Put option. 1 for call and -1 for put
%
% OUTPUT
%  sig -- implied volatilities 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Remark: the time step of the HN-GARCH is daily by default
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% compute the corresponding parameters of X_t under Q measure
%
%       X_t = r-0.5*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = omega +alpha(z_t-c*sqrt(h_t))^2+beta h_t,
% where c = lambda+gamma+0.5
%
c = gamma +lambda + 0.5;
%
% 
% Construct the willow tree for ht
%
gamma_h = 0.6;
gamma_x = 0.8;
tol = 1e-4;
itmax = 60;
[hd,qhd] = genhDelta(h0, beta, alpha, c,w, m_h, gamma_h);
[nodes_ht] = TreeNodes_ht_HN(m_h, hd, qhd, gamma_h,alpha,beta,c,w,N+1);
% [P_ht_N, P_ht] = Probility_ht(nodes_ht,h0,alpha,beta,c,omega);
% construct the willow tree for Xt
[nodes_Xt,~,~,~, ~] = TreeNodes_logSt_HN(m_x,gamma_x,r,hd,qhd,S0,alpha,beta,c,w,N);
[q_Xt,P_Xt,~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S0,r, alpha,beta,c,w);
nodes_S = exp(nodes_Xt);
%  Price the American option
 V = American(nodes_S,P_Xt,q_Xt,r,T,S0,K,CorP);
 % compute the implied volatility of the American option
[sig, V0, ~] = impvol(S0, K, T,r,V,CorP,N,m_x,gamma_x, tol, itmax);
%

