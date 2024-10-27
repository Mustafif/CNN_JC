function [S, h0] = mcDuan(M, N, S0, Z, r, omega, alpha, beta, lambda)
% Monte Carlo simulation for Duan's GARCH option pricing model
%
% Inputs:
%   M - number of simulations
%   N - number of time steps
%   S0 - initial stock price
%   Z - random normal innovations matrix (N+1 x M)
%   r - risk-free rate
%   omega - GARCH constant term
%   alpha - ARCH parameter
%   beta - GARCH parameter
%   lambda - risk premium parameter
%
% Outputs:
%   S - simulated stock price paths
%   h0 - initial variance

dt = 1/N;
numPoint = N+1;
S = zeros(N+1, M);
ht = zeros(N+1, M);

% Initial variance (long-run variance under Duan's specification)
% todo!!!
h0 = omega/(1-alpha-beta);

% Initialize first values
ht(1, :) = h0;
S(1, :) = S0;

% Simulate paths
for i = 2:numPoint
    ht(i, :) = omega + beta * ht(i-1, :) + alpha*(Z(i-1, :)^2*ht(i-1, :));
    Xt = r*dt + lambda*sqrt(ht(i,:)) - 0.5*ht(i, :) + Z(i, :)*sqrt(ht(i, :));
    
    % Update stock prices
    S(i,:) = S(i-1,:) * exp(Xt);
end
end