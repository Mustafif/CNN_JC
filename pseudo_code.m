% Pseudo Code of Pricing American Options By the Willow Tree and Monte Carlo Method
% 1. Initialize Option Parameters
T = ...; % Time to maturity
K = ...; % Strike price
r = ...; % Risk-free rate
S0 = ...; % Initial asset price
N = ...; % Number of time steps for simulation (ex. 500)
M = ...; % Number of Monte Carlo paths
delta = T/N; % Time step
h0 = ...; % Initial volatility

% 2. Initialize HN-GARCH parameters under P Measure
alpha = ...;
beta = ...;
omega = ...;
gamma = ...;
lambda = ...;

% 3. Simulate paths using Monte Carlo simulation under P measure
% Number of paths will be N so we simulate a path of asset prices 
% from S1, S2, ... S_N
points = N + 1; % number of points 

Z = randn(points + 1, N);
Z1 = randn(points, N);
ht = nan(points + 1, N);
ht(1,:) = h0*ones(1,N);
Xt(1,:) = log(S0)*ones(1,N);
for i=2:points
    ht(i,:) = omega+alpha*(Z(i-1,:)-gamma*sqrt(ht(i-1,:))).^2+beta*ht(i-1,:);
    Xt(i,:) = Xt(i-1,:)+(r-0.5*ht(i,:))+sqrt(ht(i,:)).*Z(i,:);
end
ht(i+1,:) = omega+alpha*(Z(i,:)-gamma*sqrt(ht(i,:))).^2+beta*ht(i,:);
S = exp(Xt);

% 4. Risk-neutralize GARCH parameters (Q measure)
eta = ...; 
omega_Q = omega / (1-2*alpha*eta);
gamma_Q = gamma*(1-2*alpha*eta);
alpha_Q = alpha /  (1-2*alpha*eta)^2;
lambda_Q = lambda*(1-2*alpha*eta);
rho_Q = lambda_Q + gamma_Q + 1/2;

% 5. Initialize Willow Tree parameters
m_h = ...;
m_ht = ...;
m_x = ...;
gamma_h = ...;
gamma_x = ...;

% 6. Construct the willow tree for ht (using Q measure parameters)
[hd, qhd] = genhDelta(h0, beta_Q, alpha_Q, gamma_Q, omega_Q, m_h, gamma_h);
nodes_ht = TreeNodes_ht_HN(m_ht, hd, qhd, gamma_h, alpha_Q, beta_Q, gamma_Q, omega_Q, N + 1);
[P_ht_N, P_ht] = Prob_ht(nodes_ht, h0, alpha_Q, beta_Q, gamma_Q, omega_Q);

% 7. Construct the willow tree for Xt (using Q measure parameters)
[nodes_Xt, mu, var, k3, k4] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S_0, alpha_Q, beta_Q, gamma_Q, omega_Q, N);
[q_Xt, P_Xt, tmpHt] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S_0, r, alpha_Q, beta_Q, gamma_Q, omega_Q);
nodes_S = exp(nodes_Xt);

% 8. Generate Data for last number of days 
days_to_price = ...; % could be 50 for the last 50 days
moneyness = [0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2]; % Example: from 80% to 120% of current price
maturities = [7/365, 1/12, 1/4, 1/2, 1]; % 7 days, 1 month, 3 months, 6 months, 1 year

A_prices = zeros(days_to_price, length(moneyness), length(maturities));
A_sig = zeros(days_to_price, length(moneyness), length(maturities));
A0_prices = zeros(days_to_price, length(moneyness), length(maturities));

for i = 1:days_to_price
    % Use the last simulated path as an example
    S_t = simulated_paths{1}(end-days_to_price+i); 
    % Calculate strike prices based on current stock price
    strike_prices = S_t * moneyness; 
    for j = 1:length(moneyness)
        for k = 1:length(maturities)
            CorP = -1; % Call or Put
            [A_sig(i,j,k), A_prices(i,j,k), A0_prices(i,j,k)] = impVol_HN(r, lambda_Q, 
                                ... omega_Q, beta_Q, alpha_Q, gamma_Q, h0, S_t, 
                                ... strike_prices(j), maturities(k), N, m_h, m_x, CorP);
        end
    end
end