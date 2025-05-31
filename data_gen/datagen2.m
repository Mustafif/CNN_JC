function [V, V_impl] = datagen2(T, r, S0, h0, K, alpha, beta, omega, gamma, lambda, CorP)

N = 15; % Increased from 5 for better accuracy
m_h = 10; % Increased from 6 for better resolution
% m_ht = 6;
m_x = 40; % Increased from 30 for better resolution
% gamma_h = 0.6;
% gamma_x = 0.8;
% 
% V_impl = 0;


[V_impl, V, ~] = impVol_HN(r, lambda, omega, beta, alpha, gamma, h0, S0, K, T, N, m_h, m_x, CorP);
end
