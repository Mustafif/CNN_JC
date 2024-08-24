function [A_sig, A_prices] = datagen(maturity, r, S0, N, M, h0, alpha, beta, omega, gamma, lambda, path_days)
% Parameters:
%   maturity - Time to maturity of the option.
%   r - Risk-free rate.
%   S0 - Initial asset price.
%   N - Number of time steps for simulation.
%   M - Number of Monte Carlo paths.
%   h0 - Initial volatility.
%   alpha - Alpha parameter for HN-GARCH model.
%   beta - Beta parameter for HN-GARCH model.
%   omega - Omega parameter for HN-GARCH model.
%   gamma - Gamma parameter for HN-GARCH model.
%   lambda - Lambda parameter for HN-GARCH model.
%   path_days - Number of days in the path to consider.
% Output:
%   A_sig - Implied volatilities for the American options.
%   A_prices - Prices of the American options.
    
% 3. Simulate paths using Monte Carlo simulation under P measure
    numPoint = N + 1;
    Z = randn(numPoint + 1, M);
    ht = nan(numPoint + 1, M); Xt = nan(numPoint + 1, M);
    ht(1,:) = h0 * ones(1, M);
    Xt(1,:) = log(S0) * ones(1, M);
    for i = 2:numPoint
        ht(i,:) = omega+alpha*(Z(i-1,:)-gamma*sqrt(ht(i-1,:))).^2 + beta*ht(i-1,:);
        Xt(i,:) = Xt(i-1,:) + (r - 0.5 * ht(i,:)) + sqrt(ht(i,:)) .* Z(i,:);
    end
    S = exp(Xt);
    % Get the last 'path_days' days on the path
    S = S(end - (path_days - 1):end, :);
    % 4. Risk-neutralize GARCH parameters (Q measure)
    eta = 0;
    omega_Q = omega / (1 - 2 * alpha * eta);
    gamma_Q = gamma * (1 - 2 * alpha * eta);
    alpha_Q = alpha / (1 - 2 * alpha * eta)^2;
    lambda_Q = lambda * (1 - 2 * alpha * eta);
    rho = lambda_Q + gamma_Q + 1/2;
    % 5. Initialize Willow Tree parameters
    m_h = 6; m_x = 30;
    % 6. Generate Data for the days up to the maturity
    % Generate strike prices based on moneyness through the maturity
    strike_prices = linspace(0.8 * S0, 1.2 * S0, maturity);
    A_sig = zeros(maturity, 1); A_prices = zeros(maturity, 1);
    % Setting the initial price to use the days from S
    % and wrapping around if necessary.
    S0 = S(mod(0:(maturity - 1), path_days) + 1, :);
    parfor j = 1:maturity % iterate through the maturities in parallel
        [A_sig(j), A_prices(j), ~] = impVol_HN(r, lambda_Q, omega_Q, rho, alpha_Q, gamma_Q,h0, S0(j), strike_prices(j), T, N, m_h, m_x, -1);
    end
end
