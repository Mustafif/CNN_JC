clear;
T = 30/252;
N = 504;
r = 0.05/252;
M = 1;  % Keep as 1 to avoid dimension mismatch with downstream functions
S0 = 100;
m = 0.8;
alpha = 3.20e-5;    % Increased ARCH parameter significantly for higher vol
beta = 0.65;        % Decreased to maintain stationarity
omega = 8e-6;       % Increased omega for higher long-run variance
gamma = 0.40;       % Increased leverage effect
lambda = 0.45;      % Risk premium parameter

% Check stationarity conditions for Duan GARCH model
if (beta + alpha * gamma^2 >= 1 || alpha + beta >= 1)
    error('Stationarity conditions violated: Need beta + alpha * gamma^2 < 1 and alpha + beta < 1');
end

m_h = 6;
m_ht = 6;
m_x = 30;
gamma_h = 0.6;
gamma_x = 0.8;
itmax = 100;  % More iterations
tol = 1e-6;   % Tighter tolerance
numPoint = N+1;
% Use predefined seed for reproducibility
rng(42);
Z = randn(numPoint+1, M);
[S, h0] = mcDuan(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);
fprintf('Initial variance h0: %.8f\n', h0);
fprintf('Expected annualized volatility: %.4f\n', sqrt(252*h0));

corp = 1;
S0 = S(end, :);
[nodes_ht, qht, hmom1, hmom2, hmom3, hmom4_app] = TreeNodes_ht_D(m_ht, h0, gamma_h,omega, beta, alpha, gamma, lambda, N+1);
[nodes_Xt, mu, var, k3, k4] = TreeNodes_logSt_D(m_x, gamma_x, r, h0, omega, beta, alpha, gamma, lambda, N, hmom1, hmom2);
[q_Xt, P_Xt, ~] = Probility_Xt2(nodes_ht, qht, nodes_Xt, S0, r, omega, beta, alpha, gamma, lambda);
nodes_S = exp(nodes_Xt);
K = m * S0;

[V, ~] = American(nodes_S, P_Xt, q_Xt, r, T, S0, K, corp);
[impl, ~, ~] = impvol(S0, K, T, r, V, corp, N, m_x, gamma_x, tol, itmax);
dataset = [S0; m; r; T; corp; alpha; beta; omega; gamma; lambda; impl; V]';  % Store results in dataset
headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'sigma', 'V'};

dataset = [headers; num2cell(dataset)];
writecell(dataset, 'random_data_duan0.csv');
fprintf("Saved data\n");
