function [imp_vol,V0, it] = impvol(S0, K, T,r,V,index,N,m,gamma, tol, itmax)
%
% compute the implied volatility of the American option by willow tree
% through the bisection method
%
% Input
%   S0 -- intial value of stock price
%    K -- strike prices, matrix
%    T -- maturities, vector
%    r -- interest rate
%    V -- American option prices
%    N -- # of time steps of willow tree
%    m -- # of tree nodes ate each time step
%   gamma -- gamma for sampling z
%
% Output
%   imp_vol -- implied volatilities of the American options
%
%
n = length(T);
k = size(K,1);
z = zq(m,gamma);
imp_vol = zeros(k,n);
for i = 1:n  % each maturity
    T_i = T(i);  % Extract current maturity as scalar
    [P,q]=gen_PoWiner(T_i,N,z);
    for j = 1:k
        V0 = 10000;
        it = 0;
        a = 0;
        b = 1;
        sigma = (a+b)/2;

        % Store target price for clarity
        target_price = V(j,i);

        while abs(V0-target_price) > tol && it < itmax
            Xnodes = nodes_Winer(T_i,N,z,r,sigma);
            nodes = S0.*exp(Xnodes);

            % Price American option with current tree
            V0 = American(nodes,P,q,r,T_i,S0,K(j,i),index);

            % Update bisection bounds
            if V0 > target_price
                b = sigma;
            else
                a = sigma;
            end
            sigma = (a+b)/2;
            it = it + 1;
        end
        imp_vol(j,i) = sigma;
    end
end

% function [imp_vol, converged, iterations] = impvol(S0, K, T, r, V, index, N, m, gamma, tol, itmax)
% %
% % Compute the implied volatility of American options using willow tree
% % through the bisection method
% %
% % Input
% % S0     -- initial value of stock price
% % K      -- strike prices, matrix (k x n)
% % T      -- maturities, vector (1 x n)
% % r      -- interest rate
% % V      -- American option prices, matrix (k x n)
% % index  -- option type: +1 for call, -1 for put
% % N      -- # of time steps of willow tree
% % m      -- # of tree nodes at each time step
% % gamma  -- gamma parameter for sampling z
% % tol    -- tolerance for convergence
% % itmax  -- maximum iterations
% %
% % Output
% % imp_vol    -- implied volatilities of the American options (k x n)
% % converged  -- convergence status for each option (k x n)
% % iterations -- number of iterations used for each option (k x n)
% %
% 
% % Get dimensions
% n = length(T);
% k = size(K, 1);
% 
% % Generate z values once (assuming zq function exists)
% z = zq(m, gamma);
% 
% % Initialize output matrices
% imp_vol = zeros(k, n);
% converged = false(k, n);
% iterations = zeros(k, n);
% 
% % Loop over each maturity
% for i = 1:n
%     T_i = T(i); % Current maturity
% 
%     % Generate probability matrices once per maturity
%     [P, q] = gen_PoWiner(T_i, N, z);
% 
%     % Loop over each strike
%     for j = 1:k
%         % Target option price
%         target_price = V(j, i);
% 
%         % Initialize bisection parameters
%         sigma_low = 0.001;   % Lower bound (0.1% volatility)
%         sigma_high = 1.0;    % Upper bound (300% volatility)
%         sigma = (sigma_low + sigma_high) / 2;
% 
%         % Initialize iteration counter
%         it = 0;
% 
%         % Get initial option price
%         Xnodes = nodes_Winer(T_i, N, z, r, sigma);
%         nodes = S0 .* exp(Xnodes);
%         current_price = American(nodes, P, q, r, T_i, S0, K(j,i), index);
% 
%         % Bisection method
%         while abs(current_price - target_price) > tol && it < itmax
%             % Update bounds based on current price
%             if current_price > target_price
%                 sigma_high = sigma;  % Decrease volatility
%             else
%                 sigma_low = sigma;   % Increase volatility
%             end
% 
%             % New midpoint
%             sigma = (sigma_low + sigma_high) / 2;
% 
%             % Compute new option price
%             Xnodes = nodes_Winer(T_i, N, z, r, sigma);
%             nodes = S0 .* exp(Xnodes);
%             current_price = American(nodes, P, q, r, T_i, S0, K(j,i), index);
% 
%             % Increment iteration counter
%             it = it + 1;
%         end
% 
%         % Store results
%         imp_vol(j, i) = sigma;
%         converged(j, i) = (abs(current_price - target_price) <= tol);
%         iterations(j, i) = it;
% 
%         % Warning for non-convergence
%         if ~converged(j, i)
%             warning('Convergence failed for K=%.4f, T=%.4f after %d iterations. Error=%.6f', ...
%                     K(j,i), T_i, it, abs(current_price - target_price));
%         end
%     end
% end
% 
% % Display convergence summary
% total_options = k * n;
% num_converged = sum(converged(:));
% fprintf('Implied volatility calculation completed:\n');
% fprintf('  Total options: %d\n', total_options);
% fprintf('  Converged: %d (%.1f%%)\n', num_converged, 100*num_converged/total_options);
% fprintf('  Failed: %d (%.1f%%)\n', total_options-num_converged, 100*(total_options-num_converged)/total_options);
% if any(~converged(:))
%     fprintf('  Average iterations for failed cases: %.1f\n', mean(iterations(~converged)));
% end
% 
% end