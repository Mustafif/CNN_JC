function [sig, V, V0] = impVol_HN_fixed(r, lambda, w, beta, alpha, gamma, h0, S0, K, T, N, m_h, m_x, CorP)
%
% Fixed version: Compute implied volatility of American options with underlying in HN-GARCH
% as under P measure with improved bisection solver and error handling
%       X_t = r+lambda*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = omega +alpha(z_t-gamma*sqrt(h_t))^2+beta h_t
%
%
% INPUT
%  r -- risk free interest rate
%  lambda, w, alpha, gamma -- parameters of HN-GARCH model
%  h0 -- initial value of ht
%  S0 -- initial stock price
%  K -- strike price
%  T -- maturity
%  N -- number of time steps
%  m_h -- # of possible values of h_t for willow tree
%  m_x -- # of possible values of X_t for willow tree
%  CorP -- Call or Put option. 1 for call and -1 for put
%
% OUTPUT
%  sig -- implied volatilities
%  V -- option price from HN-GARCH model
%  V0 -- option price achieved by implied volatility solver
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Remark: the time step of the HN-GARCH is daily by default
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input validation
if nargin < 14
    error('Insufficient input arguments');
end

% Ensure inputs are scalars or compatible dimensions
if ~isscalar(K) || ~isscalar(T) || ~isscalar(CorP)
    error('K, T, and CorP must be scalars');
end

% Parameter bounds checking
if h0 <= 0
    error('Initial volatility h0 must be positive');
end
if S0 <= 0
    error('Initial stock price S0 must be positive');
end
if K <= 0
    error('Strike price K must be positive');
end
if T <= 0
    error('Time to maturity T must be positive');
end

fprintf('Computing HN-GARCH implied volatility...\n');
fprintf('Parameters: S0=%.2f, K=%.2f, T=%.4f, CorP=%d\n', S0, K, T, CorP);

%
% compute the corresponding parameters of X_t under Q measure
%
%       X_t = r-0.5*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = omega +alpha(z_t-c*sqrt(h_t))^2+beta h_t,
% where c = lambda+gamma+0.5
%
c = gamma + lambda + 0.5;

fprintf('Risk-neutral parameter c = %.4f\n', c);

%
% Construct the willow tree for ht
%
gamma_h = 0.6;
gamma_x = 0.8;
tol = 1e-6;  % Tighter tolerance
itmax = 100; % More iterations

try
    % Build volatility tree
    fprintf('Building volatility tree (m_h=%d)...\n', m_h);
    [hd, qhd] = genhDelta(h0, beta, alpha, c, w, m_h, gamma_h);
    [nodes_ht] = TreeNodes_ht_HN(m_h, hd, qhd, gamma_h, alpha, beta, c, w, N+1);

    % Build log-stock price tree
    fprintf('Building log-price tree (m_x=%d)...\n', m_x);
    [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S0, alpha, beta, c, w, N);
    [q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S0, r, alpha, beta, c, w);
    nodes_S = exp(nodes_Xt);

    % Price the American option using HN-GARCH model
    fprintf('Pricing American option...\n');
    [V, ~] = American(nodes_S, P_Xt, q_Xt, r, T, S0, K, CorP);

    fprintf('HN-GARCH option price: %.6f\n', V);

    % Validate option price
    if V < 0
        warning('Negative option price detected: %.6f', V);
        V = max(0, V);
    end

    % Check for reasonable option bounds
    if CorP == 1  % Call option
        max_payoff = S0;  % Maximum possible call value
        min_payoff = max(0, S0 - K * exp(-r * T));  % Lower bound
    else  % Put option
        max_payoff = K;   % Maximum possible put value
        min_payoff = max(0, K * exp(-r * T) - S0);  % Lower bound
    end

    if V > max_payoff * 1.1  % Allow 10% buffer for American premium
        warning('Option price %.4f exceeds theoretical maximum %.4f', V, max_payoff);
    end

    % Compute the implied volatility using improved solver
    fprintf('Computing implied volatility...\n');

    % Use the fixed implied volatility solver
    if exist('impvol_fixed', 'file') == 2
        [sig, V0, iterations] = impvol_fixed(S0, K, T, r, V, CorP, N, m_x, gamma_x, tol, itmax);
    else
        % Fallback to original solver with expanded bounds
        fprintf('Warning: Using original solver - consider implementing impvol_fixed.m\n');
        [sig, V0, iterations] = impvol_expanded_bounds(S0, K, T, r, V, CorP, N, m_x, gamma_x, tol, itmax);
    end

    fprintf('Implied volatility: %.6f (%.2f%% annualized)\n', sig, sig * sqrt(252) * 100);
    fprintf('Solver achieved price: %.6f (target: %.6f)\n', V0, V);
    fprintf('Pricing error: %.2e\n', abs(V0 - V));
    fprintf('Iterations used: %d\n', iterations);

    % Final validation
    if sig <= 0.001
        warning('Implied volatility at lower bound: %.6f', sig);
    end
    if sig >= 2.99
        warning('Implied volatility at upper bound: %.6f', sig);
    end

catch ME
    fprintf('Error in HN-GARCH implied volatility calculation:\n');
    fprintf('  %s\n', ME.message);

    % Return fallback values
    sig = 0.2;  % 20% default volatility
    V0 = V;

    % Try to price option anyway if trees were built
    if exist('nodes_S', 'var') && exist('P_Xt', 'var')
        try
            [V, ~] = American(nodes_S, P_Xt, q_Xt, r, T, S0, K, CorP);
        catch
            V = 0;
        end
    else
        V = 0;
    end

    warning('Returning fallback values: sig=%.4f, V=%.4f', sig, V);
end

end

function [sig, V0, it] = impvol_expanded_bounds(S0, K, T, r, V, index, N, m, gamma, tol, itmax)
% Fallback solver with expanded bounds when impvol_fixed is not available

z = zq(m, gamma);
[P, q] = gen_PoWiner(T, N, z);

% Expanded bounds
a = 0.001;  % 0.1% minimum
b = 3.0;    % 300% maximum
sigma = (a + b) / 2;

% Better initial guess
moneyness = K / S0;
if index == 1  % Call
    if moneyness < 1
        sigma = 0.3 + 0.2 * (1 - moneyness);
    else
        sigma = 0.2 + 0.1 * (moneyness - 1);
    end
else  % Put
    if moneyness > 1
        sigma = 0.3 + 0.2 * (moneyness - 1);
    else
        sigma = 0.2 + 0.1 * (1 - moneyness);
    end
end

sigma = max(a, min(b, sigma));

V0 = 0;
it = 0;

while abs(V0 - V) > tol && it < itmax
    Xnodes = nodes_Winer(T, N, z, r, sigma);
    nodes = S0 .* exp(Xnodes);
    V0 = American(nodes, P, q, r, T, S0, K, index);

    if V0 > V
        b = sigma;
    else
        a = sigma;
    end

    sigma = (a + b) / 2;
    it = it + 1;

    % Prevent infinite loops
    if (b - a) < 1e-8
        break;
    end
end

sig = sigma;

end
