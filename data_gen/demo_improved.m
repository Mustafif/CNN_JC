clear all
% demo_improved.m
% Improved version of demo.m with fixed implied volatility solver
% This addresses the boundary issues and poor initial guesses in the original solver

fprintf('=================================================================\n');
fprintf('      IMPROVED HESTON-NANDI GARCH OPTION PRICING DEMO\n');
fprintf('=================================================================\n\n');

T = [30, 60, 90, 180, 360]./252;
m = linspace(0.8, 1.2, 9);
N = 504; % 100;
delta=T/N;
r = 0.05/252;
S0 = 100;
K = 100;

% GARCH parameters
alpha = 1.33e-6;
beta = 0.8;
omega = 1e-6;
gamma = 5;
lambda = 0.2;

dt = T/N;

% Willow tree parameters
m_h = 6;
m_ht = 6;
m_x = 30;
gamma_h = 0.6;
gamma_x = 0.8;

% Generate MC paths for GARCH model
M = 1;
numPoint = N+1;
Z = randn(numPoint+1,M);

fprintf('Generating Heston-Nandi GARCH paths...\n');
[S, h0] = mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);

% Contract timing
day_per_contract = 100;
day = 1;
S0 = S(end + (-day_per_contract + day), :);
t = T - (day - 1);

% Adjust maturities if needed
for i = 1:length(T)
    if t(i) <= 0
        t(i) = T(i) - mod(abs(t(i)), T(i));
    end
end

T_len = length(T);
m_len = length(m);

% Enhanced dataset with additional metrics
dataset = zeros(12, T_len * m_len * 2);  % Expanded for more metrics

fprintf('\nBuilding Willow trees for option pricing...\n');
fprintf('Parameters: S0=%.2f, h0=%.2e\n', S0, h0);
fprintf('GARCH: alpha=%.2e, beta=%.2f, omega=%.2e, gamma=%.1f, lambda=%.1f\n', ...
        alpha, beta, omega, gamma, lambda);

% Construct the willow tree for ht
c = gamma + lambda + 0.5;  % Risk-neutral parameter
fprintf('Risk-neutral parameter c = %.4f\n', c);

try
    [hd, qhd] = genhDelta(h0, beta, alpha, c, omega, m_h, gamma_h);
    [nodes_ht] = TreeNodes_ht_HN(m_ht, hd, qhd, gamma_h, alpha, beta, c, omega, N+1);

    % Construct the willow tree for Xt
    [nodes_Xt, mu, var, k3, k4] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S0, alpha, beta, c, omega, N);
    [q_Xt, P_Xt, tmpHt] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S0, r, alpha, beta, c, omega);
    nodes_S = exp(nodes_Xt);

    fprintf('Trees constructed successfully.\n');
    fprintf('Log-price nodes: %dx%d, Volatility nodes: %dx%d\n', ...
            size(nodes_Xt, 1), size(nodes_Xt, 2), size(nodes_ht, 1), size(nodes_ht, 2));

catch ME
    fprintf('Error building trees: %s\n', ME.message);
    return;
end

% Improved solver parameters
itmax = 100;  % More iterations
tol = 1e-6;   % Tighter tolerance
idx = 1;

% Statistics tracking
total_options = T_len * m_len * 2;
pricing_errors = 0;
solver_failures = 0;
boundary_issues = 0;

fprintf('\nPricing %d options (%d maturities × %d strikes × 2 types)...\n', ...
        total_options, T_len, m_len);

% Process each maturity and strike
for i = 1:T_len
    fprintf('Processing maturity %d/%d (T=%.3f)...\n', i, T_len, t(i));

    for j = 1:m_len
        K_strike = m(j) * S0 + 0.01 * S0 * i * j; % Strike price

        try
            % Price American options
            [V_C, ~] = American(nodes_S, P_Xt, q_Xt, r, t(i), S0, K_strike, 1);
            [V_P, ~] = American(nodes_S, P_Xt, q_Xt, r, t(i), S0, K_strike, -1);

            % Validate option prices
            if V_C < 0 || V_P < 0 || isnan(V_C) || isnan(V_P)
                fprintf('Warning: Invalid option prices at K=%.2f, T=%.3f\n', K_strike, t(i));
                pricing_errors = pricing_errors + 1;
            end

            % Compute implied volatilities using improved solver
            if exist('impvol_improved', 'file') == 2
                % Use improved solver
                [impl_c, V0_c, it_c, conv_c] = impvol_improved(S0, K_strike, t(i), r, V_C, 1, N, m_x, gamma_x, tol, itmax);
                [impl_p, V0_p, it_p, conv_p] = impvol_improved(S0, K_strike, t(i), r, V_P, -1, N, m_x, gamma_x, tol, itmax);

                % Track solver performance
                if ~conv_c || ~conv_p
                    solver_failures = solver_failures + 1;
                end
                if impl_c <= 0.002 || impl_c >= 2.99 || impl_p <= 0.002 || impl_p >= 2.99
                    boundary_issues = boundary_issues + 1;
                end

            else
                % Fallback to original solver with expanded bounds
                fprintf('Warning: Using fallback solver for case %d\n', idx);

                % Improved version of original solver
                z = zq(m_x, gamma_x);
                [P, q] = gen_PoWiner(t(i), N, z);

                % Call option
                sigma_min = 0.001; sigma_max = 3.0;
                moneyness = K_strike / S0;
                if moneyness < 1
                    init_guess_c = 0.3 + 0.2 * (1 - moneyness);
                else
                    init_guess_c = 0.2 + 0.1 * (moneyness - 1);
                end
                init_guess_c = max(sigma_min, min(sigma_max, init_guess_c));

                a = sigma_min; b = sigma_max; sigma = init_guess_c;
                it_c = 0; V0_c = 0;
                while abs(V0_c - V_C) > tol && it_c < itmax
                    Xnodes = nodes_Winer(t(i), N, z, r, sigma);
                    nodes = S0 .* exp(Xnodes);
                    V0_c = American(nodes, P, q, r, t(i), S0, K_strike, 1);
                    if V0_c > V_C
                        b = sigma;
                    else
                        a = sigma;
                    end
                    sigma = (a + b) / 2;
                    it_c = it_c + 1;
                    if (b - a) < 1e-10; break; end
                end
                impl_c = sigma;
                conv_c = (abs(V0_c - V_C) <= tol);

                % Put option
                if moneyness > 1
                    init_guess_p = 0.3 + 0.2 * (moneyness - 1);
                else
                    init_guess_p = 0.2 + 0.1 * (1 - moneyness);
                end
                init_guess_p = max(sigma_min, min(sigma_max, init_guess_p));

                a = sigma_min; b = sigma_max; sigma = init_guess_p;
                it_p = 0; V0_p = 0;
                while abs(V0_p - V_P) > tol && it_p < itmax
                    Xnodes = nodes_Winer(t(i), N, z, r, sigma);
                    nodes = S0 .* exp(Xnodes);
                    V0_p = American(nodes, P, q, r, t(i), S0, K_strike, -1);
                    if V0_p > V_P
                        b = sigma;
                    else
                        a = sigma;
                    end
                    sigma = (a + b) / 2;
                    it_p = it_p + 1;
                    if (b - a) < 1e-10; break; end
                end
                impl_p = sigma;
                conv_p = (abs(V0_p - V_P) <= tol);
            end

            % Calculate additional metrics
            moneyness_val = K_strike / S0;
            pcp_error = abs((V_C - V_P) - (S0 - K_strike * exp(-r * t(i))));

            % Store enhanced call option data
            dataset(:, idx) = [
                S0;                    % 1: Spot price
                K_strike/S0;          % 2: Moneyness
                r;                    % 3: Risk-free rate
                t(i);                 % 4: Time to maturity
                1;                    % 5: Call/Put flag
                alpha;                % 6: GARCH alpha
                beta;                 % 7: GARCH beta
                omega;                % 8: GARCH omega
                gamma;                % 9: GARCH gamma
                lambda;               % 10: GARCH lambda
                impl_c;               % 11: Implied volatility
                V_C;                  % 12: Option value
                % conv_c;               % 13: Convergence flag
                % it_c;                 % 14: Iterations used
                % pcp_error             % 15: Put-call parity error
            ];
            idx = idx + 1;

            % Store enhanced put option data
            dataset(:, idx) = [
                S0;                    % 1: Spot price
                K_strike/S0;          % 2: Moneyness
                r;                    % 3: Risk-free rate
                t(i);                 % 4: Time to maturity
                -1;                   % 5: Call/Put flag
                alpha;                % 6: GARCH alpha
                beta;                 % 7: GARCH beta
                omega;                % 8: GARCH omega
                gamma;                % 9: GARCH gamma
                lambda;               % 10: GARCH lambda
                impl_p;               % 11: Implied volatility
                V_P;                  % 12: Option value
                % conv_p;               % 13: Convergence flag
                % it_p;                 % 14: Iterations used
                % pcp_error             % 15: Put-call parity error
            ];
            idx = idx + 1;

        catch ME
            fprintf('Error processing K=%.2f, T=%.3f: %s\n', K_strike, t(i), ME.message);
            pricing_errors = pricing_errors + 1;

            % Store error placeholders
            for k = 1:2  % Call and put
                dataset(:, idx) = [S0; K_strike/S0; r; t(i); (-1)^k; alpha; beta; omega; gamma; lambda; 0.2; 0];
                idx = idx + 1;
            end
        end
    end
end

% Save the enhanced dataset
headers = {'S0', 'Moneyness', 'r', 'T', 'CallPut', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'sigma', 'V'}';
filename = 'impl_demo_improved.csv';
dataset_enhanced = [headers'; num2cell(dataset')];
writecell(dataset_enhanced, filename);

% fprintf('\n=================================================================\n');
% fprintf('                    PROCESSING COMPLETE\n');
% fprintf('=================================================================\n');
% 
% % Comprehensive summary statistics
% call_data = dataset(:, dataset(5,:) == 1);
% put_data = dataset(:, dataset(5,:) == -1);
% 
% fprintf('\nDATASET SUMMARY:\n');
% fprintf('  Total options: %d\n', size(dataset, 2));
% fprintf('  Calls: %d, Puts: %d\n', size(call_data, 2), size(put_data, 2));
% fprintf('  Enhanced dataset saved as: %s\n', filename);
% 
% fprintf('\nSOLVER PERFORMANCE:\n');
% fprintf('  Pricing errors: %d (%.1f%%)\n', pricing_errors, 100*pricing_errors/total_options);
% fprintf('  Solver failures: %d (%.1f%%)\n', solver_failures, 100*solver_failures/total_options);
% fprintf('  Boundary issues: %d (%.1f%%)\n', boundary_issues, 100*boundary_issues/total_options);
% 
% % Convergence statistics
% total_converged = sum(dataset(13,:));
% fprintf('  Overall convergence: %d/%d (%.1f%%)\n', total_converged, size(dataset,2), 100*total_converged/size(dataset,2));
% 
% % Implied volatility statistics
% all_ivs = dataset(11,:);
% call_ivs = call_data(11,:);
% put_ivs = put_data(11,:);
% 
% fprintf('\nIMPLIED VOLATILITY ANALYSIS:\n');
% fprintf('  Overall range: [%.4f, %.4f] (%.1f%% - %.1f%% annualized)\n', ...
%         min(all_ivs), max(all_ivs), min(all_ivs)*sqrt(252)*100, max(all_ivs)*sqrt(252)*100);
% 
% fprintf('  Calls:  Mean=%.4f (%.1f%%), Range=[%.4f, %.4f]\n', ...
%         mean(call_ivs), mean(call_ivs)*sqrt(252)*100, min(call_ivs), max(call_ivs));
% fprintf('  Puts:   Mean=%.4f (%.1f%%), Range=[%.4f, %.4f]\n', ...
%         mean(put_ivs), mean(put_ivs)*sqrt(252)*100, min(put_ivs), max(put_ivs));
% fprintf('  IV Spread (Call-Put): %.4f (%.1f%% annualized)\n', ...
%         mean(call_ivs) - mean(put_ivs), (mean(call_ivs) - mean(put_ivs))*sqrt(252)*100);
% 
% % Put-call parity analysis
% pcp_errors = dataset(15,:);
% large_pcp_violations = sum(pcp_errors > 0.01);
% fprintf('\nPUT-CALL PARITY:\n');
% fprintf('  Mean PCP error: %.4f\n', mean(pcp_errors));
% fprintf('  Max PCP error: %.4f\n', max(pcp_errors));
% fprintf('  Large violations (>0.01): %d (%.1f%%)\n', large_pcp_violations, 100*large_pcp_violations/size(dataset,2));
% 
% % Quality assessment
% fprintf('\nQUALITY ASSESSMENT:\n');
% boundary_calls = sum(call_ivs <= 0.005 | call_ivs >= 2.95);
% boundary_puts = sum(put_ivs <= 0.005 | put_ivs >= 2.95);
% fprintf('  Calls at boundaries: %d/%d (%.1f%%)\n', boundary_calls, length(call_ivs), 100*boundary_calls/length(call_ivs));
% fprintf('  Puts at boundaries: %d/%d (%.1f%%)\n', boundary_puts, length(put_ivs), 100*boundary_puts/length(put_ivs));
% 
% realistic_spread = abs(mean(call_ivs) - mean(put_ivs)) < 0.1;  % Less than 10% spread
% fprintf('  Realistic IV spread: %s\n', realistic_spread * 'Yes' + (1-realistic_spread) * 'No');
% 
% fprintf('\n=================================================================\n');
% if realistic_spread && total_converged/size(dataset,2) > 0.9 && boundary_calls + boundary_puts < size(dataset,2)*0.1
%     fprintf('SUCCESS: Improved solver significantly reduced boundary issues!\n');
% else
%     fprintf('PARTIAL SUCCESS: Further improvements may be needed.\n');
% end
% fprintf('=================================================================\n');
% 
% fprintf('\nRecommendations for further improvement:\n');
% if boundary_calls + boundary_puts > size(dataset,2)*0.1
%     fprintf('- Consider further expanding volatility bounds\n');
% end
% if large_pcp_violations > size(dataset,2)*0.1
%     fprintf('- Investigate tree construction accuracy\n');
% end
% if total_converged/size(dataset,2) < 0.95
%     fprintf('- Increase maximum iterations or tighten tolerance\n');
% end
% 
% fprintf('\nAnalysis complete. Enhanced dataset available in %s\n', filename);
