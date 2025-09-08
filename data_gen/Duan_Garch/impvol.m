% function [imp_vol,V0, it] = impvol(S0, K, T,r,V,index,N,m,gamma, tol, itmax)
% %
% % compute the implied volatility of the American option by willow tree
% % through the bisection method
% %
% % Input
% %   S0 -- intial value of stock price
% %    K -- strike prices, matrix
% %    T -- maturities, vector
% %    r -- interest rate
% %    V -- American option prices
% %    N -- # of time steps of willow tree
% %    m -- # of tree nodes ate each time step
% %   gamma -- gamma for sampling z
% %
% % Output
% %   imp_vol -- implied volatilities of the American options
% %
% %
% n = length(T);
% k = size(K,1);
% z = zq(m,gamma);
% imp_vol = zeros(k,n);
% for i = 1:n  % each maturity
%     [P,q]=gen_PoWiner(T(i),N,z);
%     for j = 1:k
%         V0 = 10000;
%         it = 0;
%         a = 0;
%         b = 1;
%         sigma = (a+b)/2;
%         while abs(V0-V(j,i)) >tol & it < itmax
%             Xnodes = nodes_Winer(T(i),N,z,r, sigma);
%             nodes = S0.*exp(Xnodes);
%             V0 =American(nodes,P,q,r,T(i),S0,K(j,i),index(j,i));
%             if V0>V
%                 b = sigma;
%             else
%                 a = sigma;
%             end
%             sigma = (a+b)/2;
%             it = it +1;
%         end
%         imp_vol(j,i) = sigma;
%     end
% end
%
function [imp_vol, V0, it] = impvol(S0, K, T, r, V, index, N, m, gamma, tol, itmax)
%
% Fixed version of implied volatility solver with improved bounds and initial guess
%
% Input
%   S0    -- initial value of stock price
%   K     -- strike prices, matrix
%   T     -- maturities, vector
%   r     -- interest rate
%   V     -- American option prices
%   index -- option type: +1 for call, -1 for put
%   N     -- # of time steps of willow tree
%   m     -- # of tree nodes at each time step
%   gamma -- gamma for sampling z
%   tol   -- tolerance for convergence
%   itmax -- maximum iterations
%
% Output
%   imp_vol -- implied volatilities of the American options
%   V0      -- final option price achieved
%   it      -- iterations used
%

% Get dimensions
n = length(T);
k = size(K, 1);

% Generate z values once
z = zq(m, gamma);

% Initialize output matrices
imp_vol = zeros(k, n);
V0 = zeros(k, n);
it = zeros(k, n);

% Improved bounds
sigma_min = 0.001;  % 0.1% minimum volatility
sigma_max = 3.0;    % 300% maximum volatility

fprintf('Starting implied volatility calculation...\n');
fprintf('Options to process: %d\n', k*n);

% Loop over each maturity
for i = 1:n
    T_i = T(i); % Current maturity

    % Generate probability matrices once per maturity
    [P, q] = gen_PoWiner(T_i, N, z);

    % Loop over each strike
    for j = 1:k
        % Target option price
        target_price = V(j, i);
        K_curr = K(j, i);

        % Skip if target price is too small (likely numerical noise)
        if target_price < 1e-6
            imp_vol(j, i) = sigma_min;
            V0(j, i) = target_price;
            it(j, i) = 0;
            continue;
        end

        % Calculate Black-Scholes initial guess
        moneyness = K_curr / S0;
        if index == 1  % Call
            % For calls, higher IV for ITM options
            if moneyness < 1
                initial_guess = 0.3 + 0.2 * (1 - moneyness);
            else
                initial_guess = 0.2 + 0.1 * (moneyness - 1);
            end
        else  % Put
            % For puts, higher IV for OTM options
            if moneyness > 1
                initial_guess = 0.3 + 0.2 * (moneyness - 1);
            else
                initial_guess = 0.2 + 0.1 * (1 - moneyness);
            end
        end

        % Ensure guess is within bounds
        initial_guess = max(sigma_min, min(sigma_max, initial_guess));

        % Initialize bisection parameters
        sigma_low = sigma_min;
        sigma_high = sigma_max;

        % Test bounds to ensure target is bracketed
        % Lower bound test
        Xnodes = nodes_Winer(T_i, N, z, r, sigma_low);
        nodes = S0 .* exp(Xnodes);
        price_low = American(nodes, P, q, r, T_i, S0, K_curr, index);

        % Upper bound test
        Xnodes = nodes_Winer(T_i, N, z, r, sigma_high);
        nodes = S0 .* exp(Xnodes);
        price_high = American(nodes, P, q, r, T_i, S0, K_curr, index);

        % Check if target is within bounds
        if target_price < price_low
            % Target too low, use minimum volatility
            imp_vol(j, i) = sigma_low;
            V0(j, i) = price_low;
            it(j, i) = 0;
            continue;
        elseif target_price > price_high
            % Target too high, expand upper bound
            sigma_high = sigma_max * 2;  % Try expanding
            Xnodes = nodes_Winer(T_i, N, z, r, sigma_high);
            nodes = S0 .* exp(Xnodes);
            price_high = American(nodes, P, q, r, T_i, S0, K_curr, index);

            if target_price > price_high
                % Still too high, use maximum
                imp_vol(j, i) = sigma_high;
                V0(j, i) = price_high;
                it(j, i) = 0;
                continue;
            end
        end

        % Start with initial guess
        sigma = initial_guess;
        iter_count = 0;

        % Get initial option price
        Xnodes = nodes_Winer(T_i, N, z, r, sigma);
        nodes = S0 .* exp(Xnodes);
        current_price = American(nodes, P, q, r, T_i, S0, K_curr, index);

        % Bisection method
        while abs(current_price - target_price) > tol && iter_count < itmax
            % Update bounds based on current price
            if current_price > target_price
                sigma_high = sigma;  % Decrease volatility
            else
                sigma_low = sigma;   % Increase volatility
            end

            % New midpoint
            sigma = (sigma_low + sigma_high) / 2;

            % Prevent bounds from getting too close
            if (sigma_high - sigma_low) < 1e-8
                break;
            end

            % Compute new option price
            Xnodes = nodes_Winer(T_i, N, z, r, sigma);
            nodes = S0 .* exp(Xnodes);
            current_price = American(nodes, P, q, r, T_i, S0, K_curr, index);

            % Increment iteration counter
            iter_count = iter_count + 1;
        end

        % Store results
        imp_vol(j, i) = sigma;
        V0(j, i) = current_price;
        it(j, i) = iter_count;

        % Warning for non-convergence
        if abs(current_price - target_price) > tol && iter_count >= itmax
            fprintf('Warning: No convergence for K=%.2f, T=%.3f, Target=%.4f, Final=%.4f, Error=%.6f\n', ...
                    K_curr, T_i, target_price, current_price, abs(current_price - target_price));
        end
    end

    % Progress indicator
    if mod(i, max(1, floor(n/10))) == 0
        fprintf('Completed maturity %d/%d\n', i, n);
    end
end

% Display summary statistics
total_options = k * n;
converged = (abs(V0(:) - V(:)) <= tol);
num_converged = sum(converged);
avg_iterations = mean(it(:));

fprintf('\nImplied Volatility Calculation Summary:\n');
fprintf('  Total options: %d\n', total_options);
fprintf('  Converged: %d (%.1f%%)\n', num_converged, 100*num_converged/total_options);
fprintf('  Failed: %d (%.1f%%)\n', total_options-num_converged, 100*(total_options-num_converged)/total_options);
fprintf('  Average iterations: %.1f\n', avg_iterations);
fprintf('  Volatility range: [%.4f, %.4f]\n', min(imp_vol(:)), max(imp_vol(:)));

% Check for boundary issues
boundary_low = sum(imp_vol(:) <= sigma_min + 0.001);
boundary_high = sum(imp_vol(:) >= sigma_max - 0.001);
if boundary_low > 0
    fprintf('  Warning: %d options hit lower bound\n', boundary_low);
end
if boundary_high > 0
    fprintf('  Warning: %d options hit upper bound\n', boundary_high);
end

end
