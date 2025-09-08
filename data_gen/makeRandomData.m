clear;
T = 200/252;
N = 504;
r = 0.05/252;
M = 1;
S0 = 100;
m = 1;
alpha = 1.1e-6;
beta = 0.75;
omega = 1e-7;
gamma = 90;
lambda = 0.45;

m_h = 6;
m_ht = 6;
m_x = 30;
gamma_h = 0.6;
gamma_x = 0.8;
itmax = 100;  % More iterations
tol = 1e-6;   % Tighter tolerance
numPoint = N+1;
Z = randn(numPoint+1, M);
[S, h0] = mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);

corp = 1;
S0 = S(end, :);
c = gamma + lambda + 0.5;
[hd, qhd] = genhDelta(h0, beta, alpha, c, omega, m_h, gamma_h);
[nodes_ht] = TreeNodes_ht_HN(m_ht, hd, qhd, gamma_h, alpha, beta, c, omega, N+1);
[nodes_Xt, mu, var, k3, k4] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S0, alpha, beta, c, omega, N);
[q_Xt, P_Xt, tmpHt] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S0, r, alpha, beta, c, omega);
nodes_S = exp(nodes_Xt);
K = m * S0;

[V, ~] = American(nodes_S, P_Xt, q_Xt, r, T, S0, K, corp);
[impl, ~, ~] = impvol_fixed(S0, K, T, r, V, corp, N, m_x, gamma_x, tol, itmax);
dataset = [S0; m; r; T; corp; alpha; beta; omega; gamma; lambda; impl; V]';  % Store results in dataset
headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'sigma', 'V'};

dataset = [headers; num2cell(dataset)];
writecell(dataset, 'random_data5.csv');