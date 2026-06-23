% function [S, h0] = mcGJR(M, N,S0, Z, r, omega, alpha, beta,  lambda)
% dt = 1/N; 
% numPoint = N+1;
% S = zeros(N+1, M);
% ht = zeros(N+1, M);
% h0 = omega/(1-beta-alpha-(lambda/2));
% mu = r;
% %h0 = (0.2^2)/N;
% ht(1, :) = h0;
% S(1, :) = S0;
% for i = 2:numPoint
%     ht(i,:) = omega+beta.*ht(i-1,:)+(alpha+lambda.*(Z(i-1,:)<0)).*ht(i-1,:).*Z(i-1,:).^2;
%     Xt = mu + sqrt(ht(i, :)).*Z(i, :);
%     S(i, :) = S(i-1, :) * exp(Xt);
% end
% 
% end

function [S, h0] = mcGJR(M, N, S0, Z, r, omega, alpha, beta, lambda)
% mcGJR Monte Carlo simulation of asset prices using GJR-GARCH(1,1) volatility
%
% Inputs:
%   M      - Number of simulation paths
%   N      - Number of time steps
%   S0     - Initial asset price
%   Z      - Matrix of standard normal random variables, at least (N+1) x M
%   r      - Per-step risk-free rate/drift
%   omega  - GJR-GARCH constant parameter
%   alpha  - ARCH parameter
%   beta   - GARCH parameter
%   lambda - Leverage effect parameter
%
% Outputs:
%   S      - Matrix of simulated asset prices (N+1 x M)
%   h0     - Initial variance
%
% Note: This matches the GJR willow tree convention:
%       X_t = r + sqrt(h_t) z_t
%       h_{t+1} = omega + beta h_t + (alpha + lambda I[z_t < 0]) h_t z_t^2

    % Input validation
    % validateattributes(M, {'numeric'}, {'positive', 'integer', 'scalar'});
    % validateattributes(N, {'numeric'}, {'positive', 'integer', 'scalar'});
    % validateattributes(S0, {'numeric'}, {'positive', 'scalar'});
    % validateattributes(Z, {'numeric'}, {'size', [N+1, M]});
    % validateattributes(r, {'numeric'}, {'scalar'});
    % validateattributes(omega, {'numeric'}, {'positive', 'scalar'});
    % validateattributes(alpha, {'numeric'}, {'nonnegative', 'scalar'});
    % validateattributes(beta, {'numeric'}, {'nonnegative', 'scalar'});
    % validateattributes(lambda, {'numeric'}, {'nonnegative', 'scalar'});

    % Check stationarity condition
    if beta + alpha + (lambda/2) >= 1
        error('Stationarity condition violated: beta + alpha + lambda/2 must be less than 1');
    end

    % Initialize variables
    numPoints = N + 1;
    S = zeros(numPoints, M);  % Asset price matrix
    ht = zeros(numPoints, M); % Volatility process matrix

    % Calculate initial variance (long-run variance)
    h0 = omega / (1 - beta - alpha - (lambda/2));

    % Set initial conditions
    ht(1, :) = h0;
    S(1, :) = S0;

    % Main simulation loop
    for i = 2:numPoints
        z_prev = Z(i-1, :);
        ht(i, :) = omega + beta .* ht(i-1, :) + ...
                   (alpha + lambda .* (z_prev < 0)) .* ht(i-1, :) .* z_prev.^2;

        Xt = r + sqrt(ht(i, :)) .* Z(i, :);
        S(i, :) = S(i-1, :) .* exp(Xt);
    end
end
