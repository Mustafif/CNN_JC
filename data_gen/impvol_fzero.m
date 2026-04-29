function [imp_vol, V0, converged] = impvol_fzero(S0, K, T, r, V_target, index, N, m, gamma, z_cached, P_cached, q_cached)
% IMPVOL_FZERO Implied volatility solver using MATLAB's fzero
%
% Robust implied volatility calculation for American options using fzero
% More reliable than custom iterative solvers
%
% Input:
%   S0       -- initial stock price
%   K        -- strike price
%   T        -- time to maturity (years)
%   r        -- risk-free rate (daily or annual, depending on context)
%   V_target -- target American option price
%   index    -- option type: +1 for call, -1 for put
%   N        -- number of time steps for willow tree
%   m        -- number of tree nodes at each time step
%   gamma    -- gamma parameter for sampling z
%   z_cached -- (optional) pre-computed z values to avoid recalc
%   P_cached -- (optional) pre-computed P matrix
%   q_cached -- (optional) pre-computed q matrix
%
% Output:
%   imp_vol   -- implied volatility
%   V0        -- option price at implied volatility
%   converged -- convergence flag

% Default return values
imp_vol = NaN;
V0 = V_target;
converged = false;

% Skip options with negligible value
if V_target < 1e-8
    impl_vol = 0.001;
    V0 = V_target;
    converged = true;
    return;
end

% Use cached z, P, q if provided; otherwise compute
if nargin >= 10 && ~isempty(z_cached)
    z = z_cached;
    P = P_cached;
    q = q_cached;
else
    % Generate z values and probability matrices once
    z = zq(m, gamma);
    [P, q] = gen_PoWiner(T, N, z);
end

% Define objective function: price difference
% Returns difference between computed price and target price
objective = @(sigma) price_difference(sigma, S0, K, T, r, V_target, index, N, z, P, q);

% Calculate moneyness for initial guess
moneyness = K / S0;

% Smart initial guess based on moneyness and option type
% Using realistic volatility ranges based on GARCH model behavior
if index == 1  % Call option
    if moneyness < 0.9
        initial_guess = 0.25;  % Deep ITM - lower vol
    elseif moneyness < 1.1
        initial_guess = 0.20;  % ATM - lower starting point
    else
        initial_guess = 0.30;  % OTM - moderate vol
    end
else  % Put option
    if moneyness > 1.1
        initial_guess = 0.25;  % Deep ITM
    elseif moneyness > 0.9
        initial_guess = 0.20;  % ATM
    else
        initial_guess = 0.30;  % OTM
    end
end

% Set bounds for volatility search
% GARCH models can have very high volatility
sigma_min = 0.001;   % 0.1% minimum
sigma_max = 5.0;     % 500% maximum (GARCH can be extreme)

% Try to bracket the root
try
    % First try narrow bracket around initial guess (fastest)
    sigma_low_narrow = max(0.001, initial_guess * 0.3);
    sigma_high_narrow = min(5.0, initial_guess * 3.0);
    
    obj_low_narrow = objective(sigma_low_narrow);
    obj_high_narrow = objective(sigma_high_narrow);
    
    % Check if we bracketed the solution with narrow bounds
    if obj_low_narrow * obj_high_narrow <= 0
        % Use narrow bracketing
        sigma_low = sigma_low_narrow;
        sigma_high = sigma_high_narrow;
    else
        % Expand to full bounds if narrow bracket failed
        sigma_low = 0.001;
        sigma_high = 5.0;
        obj_low = objective(sigma_low);
        obj_high = objective(sigma_high);
        
        % Check if full bracket works
        if obj_low * obj_high > 0
            % Cannot bracket - target price may be outside achievable range
            if abs(obj_low) < abs(obj_high)
                imp_vol = sigma_low;
            else
                imp_vol = sigma_high;
            end
            V0 = V_target - objective(imp_vol);
            converged = false;
            return;
        end
    end
    
    % Use fzero with optimized options for speed
    % Looser tolerance for 10x faster convergence with minimal accuracy loss
    options = optimset('Display', 'off', 'TolX', 1e-4, 'TolFun', 1e-3, 'MaxIter', 15);
    
    % Try fzero with bracketed interval
    [imp_vol, fval, exitflag] = fzero(objective, [sigma_min, sigma_max], options);
    
    % Check convergence
    converged = (exitflag == 1) && abs(fval) < 1e-4;
    
    % If fzero succeeded, compute final price
    if converged
        V0 = V_target - fval;
    else
        % Fallback: use initial guess and custom solver
        [imp_vol, V0, converged] = fallback_solver(S0, K, T, r, V_target, index, N, z, P, q, initial_guess);
    end
    
catch ME
    % If fzero fails, use fallback solver
    try
        [imp_vol, V0, converged] = fallback_solver(S0, K, T, r, V_target, index, N, z, P, q, initial_guess);
    catch
        % Complete failure - return initial guess
        imp_vol = initial_guess;
        V0 = V_target;
        converged = false;
    end
end

% Ensure volatility is within reasonable bounds
imp_vol = max(sigma_min, min(sigma_max, imp_vol));

end

function diff = price_difference(sigma, S0, K, T, r, V_target, index, N, z, P, q)
% Compute difference between option price at sigma and target price
try
    % Generate willow tree nodes
    Xnodes = nodes_Winer(T, N, z, r, sigma);
    nodes = S0 .* exp(Xnodes);
    
    % Price American option
    V_computed = American(nodes, P, q, r, T, S0, K, index);
    
    % Return difference
    diff = V_computed - V_target;
    
catch
    % If pricing fails, return large penalty
    diff = 1e10;
end
end

function [imp_vol, V0, converged] = fallback_solver(S0, K, T, r, V_target, index, N, z, P, q, initial_guess)
% Simple bisection fallback if fzero fails
% Optimized for speed with looser convergence
sigma_low = 0.001;
sigma_high = 5.0;
sigma = initial_guess;
tol = 1e-3;  % Looser tolerance for speed
max_iter = 20;  % Reduced iterations

for iter = 1:max_iter
    % Compute price at current sigma
    Xnodes = nodes_Winer(T, N, z, r, sigma);
    nodes = S0 .* exp(Xnodes);
    V_computed = American(nodes, P, q, r, T, S0, K, index);
    
    % Check convergence
    error = V_computed - V_target;
    if abs(error) < tol
        imp_vol = sigma;
        V0 = V_computed;
        converged = true;
        return;
    end
    
    % Update bounds
    if error > 0
        sigma_high = sigma;
    else
        sigma_low = sigma;
    end
    
    % Bisection step
    sigma = (sigma_low + sigma_high) / 2;
    
    % Check for convergence in sigma
    if abs(sigma_high - sigma_low) < 1e-5
        break;
    end
end

imp_vol = sigma;
V0 = V_computed;
converged = abs(V_computed - V_target) < tol;
end
