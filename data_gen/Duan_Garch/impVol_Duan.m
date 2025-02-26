function [sig,V,V0] = impVol_Duan(r, lambda,w, beta, alpha, theta,h0,S0, K, T, N, m_h, m_x, CorP)
gamma_h = 0.8;
gamma_x = 0.8;
tol = 1e-4;
itmax = 60;
[nodes_ht,qht, hmom1, hmom2, hmom3, hmom4_app] = TreeNodes_ht_D(m_h, h0, gamma_h,w, beta,alpha, theta, lambda,N+1);
[nodes_Xt,mu,var,k3, k4] = TreeNodes_logSt_D(m_x,gamma_x,r, h0, w,beta, alpha,theta, lambda,N, hmom1, hmom2);
nodes_Xt = nodes_Xt+log(S0);
[q_Xt,P_Xt,tmpHt] = Probility_Xt2(nodes_ht,qht, nodes_Xt, S0,r,w,beta,alpha,theta, lambda);
nodes_S = exp(nodes_Xt);
[V, ~] = American(nodes_S, P_Xt, q_Xt, r, T, S0, K, CorP);
[sig, V0, ~] = impvol(S0, K, T, r, V, CorP, N, m_x, gamma_x, tol, itmax);
end