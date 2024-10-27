function [V, V_impl] = datagen3(T, r, S0, h0, K, alpha, beta, omega, gamma, lambda, CorP)

N = 5;
m_h = 6;
m_ht = 6;
m_x = 30;
gamma_h = 0.6;
gamma_x = 0.8;

V_impl = 0;


[V_impl, V, ~] = impVol_Duan(r, lambda, omega, beta, alpha, gamma, h0, S0, K, T, N, m_h, m_x, CorP);
end
