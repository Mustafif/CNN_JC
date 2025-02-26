function [sig,V,V0] = impVol_GJR(r, lambda,w, beta, alpha, h0,S0, K, T, N, m_h, m_x, CorP)
    gamma_h = 0.6;
gamma_x = 0.8;
tol = 1e-4;
itmax = 60;
mu = r;

[nodes_ht,qht, hmom1, hmom2, hmom3, hmom4_app] = TreeNodes_ht_GJR(m_h, h0, gamma_h,w,beta,alpha, lambda, N);
[nodes_Xt,mu1,var1,k31, k41] = TreeNodes_logSt_GJR(m_x,m_h,gamma_x,mu,w,alpha, lambda, beta,h0, N, hmom1, hmom2);
nodes_Xt = nodes_Xt+log(S0);
nodes_S = exp(nodes_Xt);
[q_Xt,P_Xt,tmpHt] = Probility_Xt2(nodes_ht,qht, nodes_Xt, S0, mu, w, beta,alpha, lambda);

[V, ~] = American(nodes_S,P_Xt,q_Xt,r,T,S0,K,CorP);
[sig, V0, ~] = impvol(S0, K, T,r,V,CorP,N,m_x,gamma_x, tol, itmax);
end