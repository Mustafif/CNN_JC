function [A_sig, A_prices] = datagen(maturity, r, S0, N, M, h0, alpha, beta, omega, gamma, lambda)
% Pseudo Code of Pricing American Options By the Willow Tree and Monte Carlo Method
% 3. Simulate paths using Monte Carlo simulation under P measure
numPoint = N+1;
Z = randn(numPoint+1,M);
ht = nan(numPoint+1,M);
Xt = nan(numPoint+1, M);
ht(1,:) = h0*ones(1,M);
Xt(1,:) = log(S0)*ones(1,M);
for i=2:numPoint
    ht(i,:) = omega+alpha*(Z(i-1,:)-gamma*sqrt(ht(i-1,:))).^2+beta*ht(i-1,:);
    Xt(i,:) = Xt(i-1,:)+(r-0.5*ht(i,:))+sqrt(ht(i,:)).*Z(i,:);
end
S = exp(Xt);
% Get the last 50 days on the path 
S = S(end-49:end, :);

% 4. Risk-neutralize GARCH parameters (Q measure)
eta = 0;
omega_Q = omega / (1-2*alpha*eta);
gamma_Q = gamma*(1-2*alpha*eta);
alpha_Q = alpha /  (1-2*alpha*eta)^2;
lambda_Q = lambda*(1-2*alpha*eta);
rho = lambda_Q + gamma_Q + 1/2;

% 5. Initialize Willow Tree parameters
m_h = 6;
m_x = 30;

% 6. Generate Data for the days upto the maturity
% Initialize CorP, A_sig, A_prices, and A0_prices
CorP = -1; % Call or Put
% Generate strike prices based on moneyness through the maturity
strike_prices = linspace(0.8*S0, 1.2*S0, maturity);
A_sig = zeros(maturity, 1);
A_prices = zeros(maturity, 1);
% Setting the initial price to use the days from S 
% and wrapping around if necessary. 
S0 = S(mod(0:(maturity-1), 50)+1, :);
parfor j = 1:maturity % iterate through the maturities in parallel
    K = strike_prices(j);
    [A_sig(j), A_prices(j), ~] = impVol_HN(r, lambda_Q, omega_Q, rho, alpha_Q, gamma_Q, h0, S0(j), K, T, N, m_h, m_x, CorP);
end
end