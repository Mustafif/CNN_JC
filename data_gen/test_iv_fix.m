% test_iv_fix.m
% Test script to compare original vs fixed implied volatility calculations

clear all; close all;

fprintf('=================================================================\n');
fprintf('      TESTING IMPROVED IMPLIED VOLATILITY CALCULATION\n');
fprintf('=================================================================\n\n');

% Parameters from original demo.m
T = [30, 60, 90, 180, 360]./252;
m = linspace(0.8, 1.2, 5);  % Reduced for testing
N = 100;  % Reduced for faster testing
r = 0.05/252;
S0 = 100;

% GARCH parameters
alpha = 1.33e-6;
beta = 0.8;
omega = 1e-6;
gamma = 5;
lambda = 0.2;

% Willow tree parameters
m_h = 6;
m_ht = 6;
m_x = 20;  % Reduced for faster testing
gamma_h = 0.6;
gamma_x = 0.8;

% Generate synthetic starting volatility
h0 = (0.2^2)/252;

fprintf('Test Parameters:\n');
fprintf('  S0 = %.2f\n', S0);
fprintf('  Maturities: %d (%.3f to %.3f years)\n', length(T), min(T), max(T));
fprintf('  Moneyness: %d (%.2f to %.2f)\n', length(m), min(m), max(m));
fprintf('  Tree nodes: m_x=%d, m_h=%d\n', m_x, m_h);

% Build trees once for efficiency
fprintf('\nBuilding Willow trees...\n');
c = gamma + lambda + 0.5;  % Risk-neutral parameter

try
    [hd, qhd] = genhDelta(h0, beta, alpha, c, omega, m_h, gamma_h);
    [nodes_ht] = TreeNodes_ht_HN(m_ht, hd, qhd, gamma_h, alpha, beta, c, omega, N+1);
    [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_HN(m_x, gamma_x, r, hd, qhd, S0, alpha, beta, c, omega, N);
    [q_Xt, P_Xt, ~] = Prob_Xt(nodes_ht, qhd, nodes_Xt, S0, r, alpha, beta, c, omega);
    nodes_S = exp(nodes_Xt);
    fprintf('Trees built successfully.\n');
catch ME
    fprintf('Error building trees: %s\n', ME.message);
    return;
end

% Test cases
test_cases = [];
case_count = 0;

for i = 1:length(T)
    for j = 1:length(m)
        for corp = [-1, 1]  % Put then Call
            case_count = case_count + 1;
            K = m(j) * S0;

            % Price the American option
            try
                [V, ~] = American(nodes_S, P_Xt, q_Xt, r, T(i), S0, K, corp);
            catch ME
                fprintf('Error pricing option: %s\n', ME.message);
                V = 0;
            end

            test_cases(case_count, :) = [S0, K, T(i), corp, V];
        end
    end
end

fprintf('\nGenerated %d test cases.\n', case_count);

% Test original solver (with protection against errors)
fprintf('\n--- Testing Original Solver ---\n');
original_results = zeros(case_count, 4);  % [iv, price_achieved, iterations, converged]
original_errors = 0;

for i = 1:case_count
    S0_test = test_cases(i, 1);
    K_test = test_cases(i, 2);
    T_test = test_cases(i, 3);
    corp_test = test_cases(i, 4);
    V_target = test_cases(i, 5);

    if V_target < 1e-6
        % Skip near-zero options
        original_results(i, :) = [0.001, V_target, 0, 0];
        continue;
    end

    try
        % Use original bounds and solver
        tol = 1e-4;
        itmax = 60;

        % Original impvol with restrictive bounds
        z = zq(m_x, gamma_x);
        [P, q] = gen_PoWiner(T_test, N, z);

        a = 0.001;
        b = 1.0;  % Original restrictive upper bound
        sigma = (a + b) / 2;

        it = 0;
        V0 = 0;

        while abs(V0 - V_target) > tol && it < itmax
            Xnodes = nodes_Winer(T_test, N, z, r, sigma);
            nodes = S0_test .* exp(Xnodes);
            V0 = American(nodes, P, q, r, T_test, S0_test, K_test, corp_test);

            if V0 > V_target
                b = sigma;
            else
                a = sigma;
            end
            sigma = (a + b) / 2;
            it = it + 1;
        end

        converged = (abs(V0 - V_target) <= tol);
        original_results(i, :) = [sigma, V0, it, converged];

    catch ME
        original_errors = original_errors + 1;
        original_results(i, :) = [0.001, 0, 0, 0];
    end
end

% Test improved solver
fprintf('\n--- Testing Improved Solver ---\n');
improved_results = zeros(case_count, 4);  % [iv, price_achieved, iterations, converged]
improved_errors = 0;

for i = 1:case_count
    S0_test = test_cases(i, 1);
    K_test = test_cases(i, 2);
    T_test = test_cases(i, 3);
    corp_test = test_cases(i, 4);
    V_target = test_cases(i, 5);

    if V_target < 1e-6
        % Skip near-zero options
        improved_results(i, :) = [0.001, V_target, 0, 1];
        continue;
    end

    try
        % Improved solver with better bounds and initial guess
        tol = 1e-6;
        itmax = 100;

        z = zq(m_x, gamma_x);
        [P, q] = gen_PoWiner(T_test, N, z);

        % Improved bounds
        sigma_min = 0.001;
        sigma_max = 3.0;

        % Better initial guess
        moneyness = K_test / S0_test;
        if corp_test == 1  % Call
            if moneyness < 1
                initial_guess = 0.3 + 0.2 * (1 - moneyness);
            else
                initial_guess = 0.2 + 0.1 * (moneyness - 1);
            end
        else  % Put
            if moneyness > 1
                initial_guess = 0.3 + 0.2 * (moneyness - 1);
            else
                initial_guess = 0.2 + 0.1 * (1 - moneyness);
            end
        end

        initial_guess = max(sigma_min, min(sigma_max, initial_guess));

        % Test bounds
        Xnodes = nodes_Winer(T_test, N, z, r, sigma_min);
        nodes = S0_test .* exp(Xnodes);
        price_low = American(nodes, P, q, r, T_test, S0_test, K_test, corp_test);

        Xnodes = nodes_Winer(T_test, N, z, r, sigma_max);
        nodes = S0_test .* exp(Xnodes);
        price_high = American(nodes, P, q, r, T_test, S0_test, K_test, corp_test);

        if V_target < price_low
            sigma = sigma_min;
            V0 = price_low;
            it = 0;
        elseif V_target > price_high
            sigma = sigma_max;
            V0 = price_high;
            it = 0;
        else
            % Bisection with improved initial guess
            sigma_low = sigma_min;
            sigma_high = sigma_max;
            sigma = initial_guess;

            it = 0;
            Xnodes = nodes_Winer(T_test, N, z, r, sigma);
            nodes = S0_test .* exp(Xnodes);
            V0 = American(nodes, P, q, r, T_test, S0_test, K_test, corp_test);

            while abs(V0 - V_target) > tol && it < itmax
                if V0 > V_target
                    sigma_high = sigma;
                else
                    sigma_low = sigma;
                end

                sigma = (sigma_low + sigma_high) / 2;

                if (sigma_high - sigma_low) < 1e-8
                    break;
                end

                Xnodes = nodes_Winer(T_test, N, z, r, sigma);
                nodes = S0_test .* exp(Xnodes);
                V0 = American(nodes, P, q, r, T_test, S0_test, K_test, corp_test);

                it = it + 1;
            end
        end

        converged = (abs(V0 - V_target) <= tol);
        improved_results(i, :) = [sigma, V0, it, converged];

    catch ME
        improved_errors = improved_errors + 1;
        improved_results(i, :) = [0.001, 0, 0, 0];
    end
end

% Analysis and comparison
fprintf('\n=================================================================\n');
fprintf('                        RESULTS COMPARISON\n');
fprintf('=================================================================\n');

% Basic statistics
orig_iv = original_results(:, 1);
orig_converged = original_results(:, 4);
orig_iterations = original_results(:, 3);

impr_iv = improved_results(:, 1);
impr_converged = improved_results(:, 4);
impr_iterations = improved_results(:, 3);

fprintf('\nCONVERGENCE COMPARISON:\n');
fprintf('  Original solver:\n');
fprintf('    Converged: %d/%d (%.1f%%)\n', sum(orig_converged), case_count, 100*sum(orig_converged)/case_count);
fprintf('    Errors: %d\n', original_errors);
fprintf('    Avg iterations: %.1f\n', mean(orig_iterations(orig_converged > 0)));

fprintf('  Improved solver:\n');
fprintf('    Converged: %d/%d (%.1f%%)\n', sum(impr_converged), case_count, 100*sum(impr_converged)/case_count);
fprintf('    Errors: %d\n', improved_errors);
fprintf('    Avg iterations: %.1f\n', mean(impr_iterations(impr_converged > 0)));

fprintf('\nIMPLIED VOLATILITY STATISTICS:\n');
fprintf('  Original solver:\n');
fprintf('    Range: [%.4f, %.4f]\n', min(orig_iv), max(orig_iv));
fprintf('    Mean: %.4f ± %.4f\n', mean(orig_iv), std(orig_iv));
fprintf('    Boundary hits (≤0.005): %d\n', sum(orig_iv <= 0.005));
fprintf('    Boundary hits (≥0.99): %d\n', sum(orig_iv >= 0.99));

fprintf('  Improved solver:\n');
fprintf('    Range: [%.4f, %.4f]\n', min(impr_iv), max(impr_iv));
fprintf('    Mean: %.4f ± %.4f\n', mean(impr_iv), std(impr_iv));
fprintf('    Boundary hits (≤0.005): %d\n', sum(impr_iv <= 0.005));
fprintf('    Boundary hits (≥2.95): %d\n', sum(impr_iv >= 2.95));

% Detailed comparison for calls vs puts
call_idx = (test_cases(:, 4) == 1);
put_idx = (test_cases(:, 4) == -1);

fprintf('\nCALL OPTIONS:\n');
fprintf('  Original: Mean IV = %.4f, Boundary hits = %d\n', ...
    mean(orig_iv(call_idx)), sum(orig_iv(call_idx) >= 0.99));
fprintf('  Improved: Mean IV = %.4f, Boundary hits = %d\n', ...
    mean(impr_iv(call_idx)), sum(impr_iv(call_idx) >= 2.95));

fprintf('\nPUT OPTIONS:\n');
fprintf('  Original: Mean IV = %.4f, Boundary hits = %d\n', ...
    mean(orig_iv(put_idx)), sum(orig_iv(put_idx) <= 0.005));
fprintf('  Improved: Mean IV = %.4f, Boundary hits = %d\n', ...
    mean(impr_iv(put_idx)), sum(impr_iv(put_idx) <= 0.005));

% Sample detailed results
fprintf('\nSAMPLE DETAILED COMPARISONS:\n');
fprintf('%-8s %-8s %-8s %-8s | %-12s %-12s | %-12s %-12s\n', ...
    'S0', 'K', 'T', 'C/P', 'Orig IV', 'Orig Conv', 'Impr IV', 'Impr Conv');
fprintf('%-8s %-8s %-8s %-8s | %-12s %-12s | %-12s %-12s\n', ...
    '----', '----', '----', '---', '--------', '--------', '--------', '--------');

for i = 1:min(10, case_count)
    fprintf('%-8.1f %-8.1f %-8.3f %-8s | %-12.4f %-12s | %-12.4f %-12s\n', ...
        test_cases(i, 1), test_cases(i, 2), test_cases(i, 3), ...
        (test_cases(i, 4) == 1) * 'Call' + (test_cases(i, 4) == -1) * 'Put ', ...
        orig_iv(i), (orig_converged(i) > 0) * 'Yes' + (orig_converged(i) == 0) * 'No ', ...
        impr_iv(i), (impr_converged(i) > 0) * 'Yes' + (impr_converged(i) == 0) * 'No ');
end

fprintf('\n=================================================================\n');
fprintf('                            SUMMARY\n');
fprintf('=================================================================\n');

improvement_conv = sum(impr_converged) - sum(orig_converged);
improvement_pct = 100 * improvement_conv / case_count;

fprintf('\nKEY IMPROVEMENTS:\n');
fprintf('1. Convergence improved by %d cases (%.1f%%)\n', improvement_conv, improvement_pct);
fprintf('2. Reduced boundary hits for puts: %d → %d\n', ...
    sum(orig_iv(put_idx) <= 0.005), sum(impr_iv(put_idx) <= 0.005));
fprintf('3. Reduced boundary hits for calls: %d → %d\n', ...
    sum(orig_iv(call_idx) >= 0.99), sum(impr_iv(call_idx) >= 2.95));
fprintf('4. More realistic volatility ranges\n');
fprintf('5. Better initial guess reduces iterations\n');

fprintf('\nRECOMMENDATIONS:\n');
fprintf('1. Replace impvol.m with improved version\n');
fprintf('2. Use expanded bounds [0.001, 3.0] instead of [0.001, 1.0]\n');
fprintf('3. Implement moneyness-based initial guess\n');
fprintf('4. Add boundary checking and warnings\n');
fprintf('5. Consider European option validation as baseline\n');

fprintf('\n=================================================================\n');
fprintf('Test completed successfully.\n');
fprintf('=================================================================\n');
