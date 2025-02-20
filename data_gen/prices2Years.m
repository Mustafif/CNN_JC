% N = monte carlo time points, we are considering 2 years 
% which in trading days is 504 days 
N = 504;
% We will be simulating this under 1 path for the monte carlo simulation 
M = 1;

Z = randn(N + 1, M); % z_i ~ N(0, 1)

S0 = 100;
r = 0.03; % fixed 3% 

% omega = 1e-6;  % Constant variance (low values for equities)
% alpha = 0.1;      % Short-term volatility response
% beta = 0.8;      % Volatility persistence
% gamma = 0.5;     % Leverage effect parameter
% lambda = 0.05;     % Risk premium adjustment factor

alpha = 1.33e-6;
beta = 0.8;
omega = 1e-6;
gamma = 100;
lambda = 0.5;


[S, h0] = mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);

% plot(0:1:504, S);

