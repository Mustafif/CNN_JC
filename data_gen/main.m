maturities = [5, 63, 126, 252]; % week, 3 months, 6 months and one year of trading days 
filenames = {'week.csv', 'quarter.csv', 'half.csv', 'annual.csv'};
% 1. Initialize Option Parameters
r = 0.05/252; % Risk-free rate
S0 = 100; % Initial asset price
N = 100; % Number of time steps for simulation (ex. 500)
M = 10000; % Number of Monte Carlo paths
h0 = (0.2^2)/252; % Initial volatility

% 2. Initialize HN-GARCH parameters under P Measure
alpha = 1.33e-6;
beta = 0.586;
omega = 4.96e-6;
gamma = 484.69;
lambda = 0.5;

path_days = 50;

parfor i = 1:length(maturities)
    [sig, V] = datagen(maturities(i), r, S0, N, M, h0, alpha, beta, omega, gamma, lambda, path_days);
    data = table(V, sig, 'VariableNames', {'Option Price', 'Implied Volatility'});
    writetable(data, filenames{i});
end