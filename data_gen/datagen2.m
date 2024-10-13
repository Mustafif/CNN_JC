function [V, V_impl] = datagen2(T, r, S0, h0, K, alpha, beta, omega, gamma, lambda, CorP)

N = 5;
m_h = 6;
m_x = 30;
[V_impl, V, ~] = impVol_HN(r, lambda, omega, beta, alpha, gamma, h0, S0, K, T, N, m_h, m_x, CorP);

end