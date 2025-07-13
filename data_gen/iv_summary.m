% iv_summary.m
% Summary of Implied Volatility Analysis for Heston-Nandi GARCH Options

clear all; close all;

% Load the data (skip header row)
data = dlmread('impl_demo.csv', ',', 1, 0);

% Extract relevant columns
S0 = data(:,1);
m = data(:,2);  % moneyness
r = data(:,3);
T = data(:,4);
corp = data(:,5);  % 1 for call, -1 for put
sigma = data(:,11);  % implied volatility
V = data(:,12);  % option value
K = m .* S0;  % strike price

% Separate calls and puts
call_idx = (corp == 1);
put_idx = (corp == -1);

fprintf('=======================================================\n');
fprintf('         IMPLIED VOLATILITY ANALYSIS SUMMARY\n');
fprintf('=======================================================\n\n');

fprintf('DATASET OVERVIEW:\n');
fprintf('  Total options: %d (Calls: %d, Puts: %d)\n', length(data), sum(call_idx), sum(put_idx));
fprintf('  Spot price: %.2f\n', S0(1));
fprintf('  Maturities: %.3f to %.3f years\n', min(T), max(T));
fprintf('  Moneyness range: %.2f to %.2f\n', min(m), max(m));

fprintf('\nKEY FINDINGS:\n');
fprintf('=============\n');

% 1. Boundary Issues
boundary_calls = sum(sigma(call_idx) >= 0.99);
boundary_puts = sum(sigma(put_idx) <= 0.02);

fprintf('\n1. SOLVER BOUNDARY ISSUES:\n');
fprintf('   - Calls hitting upper bound (IV≥0.99): %d/%d (%.1f%%)\n', ...
    boundary_calls, sum(call_idx), 100*boundary_calls/sum(call_idx));
fprintf('   - Puts hitting lower bound (IV≤0.02): %d/%d (%.1f%%)\n', ...
    boundary_puts, sum(put_idx), 100*boundary_puts/sum(put_idx));

% 2. IV Statistics
fprintf('\n2. IMPLIED VOLATILITY STATISTICS:\n');
fprintf('   Calls:  Mean=%.4f, Std=%.4f, Range=[%.4f, %.4f]\n', ...
    mean(sigma(call_idx)), std(sigma(call_idx)), min(sigma(call_idx)), max(sigma(call_idx)));
fprintf('   Puts:   Mean=%.4f, Std=%.4f, Range=[%.4f, %.4f]\n', ...
    mean(sigma(put_idx)), std(sigma(put_idx)), min(sigma(put_idx)), max(sigma(put_idx)));
fprintf('   IV Spread (Call-Put): %.4f\n', mean(sigma(call_idx)) - mean(sigma(put_idx)));

% 3. Maturity Pattern
unique_T = unique(T);
fprintf('\n3. MATURITY PATTERN:\n');
for i = 1:length(unique_T)
    t_idx = (T == unique_T(i));
    call_iv = mean(sigma(call_idx & t_idx));
    put_iv = mean(sigma(put_idx & t_idx));
    fprintf('   T=%.3f: Call IV=%.4f, Put IV=%.4f, Spread=%.4f\n', ...
        unique_T(i), call_iv, put_iv, call_iv - put_iv);
end

% 4. Put-Call Parity Violations
fprintf('\n4. PUT-CALL PARITY VIOLATIONS:\n');
violations = 0;
max_violation = 0;
for i = 1:length(unique_T)
    for j = 1:length(unique(m))
        t_val = unique_T(i);
        m_val = unique(m)(j);

        call_match = call_idx & (T == t_val) & (m == m_val);
        put_match = put_idx & (T == t_val) & (m == m_val);

        if sum(call_match) == 1 && sum(put_match) == 1
            C = V(call_match);
            P = V(put_match);
            S = S0(call_match);
            K_val = K(call_match);
            r_val = r(call_match);
            T_val = T(call_match);

            pcp_left = C - P;
            pcp_right = S - K_val * exp(-r_val * T_val);
            pcp_error = abs(pcp_left - pcp_right);

            if pcp_error > 0.01
                violations = violations + 1;
                max_violation = max(max_violation, pcp_error);
            end
        end
    end
end
fprintf('   Total violations (error > 0.01): %d\n', violations);
fprintf('   Maximum violation: %.4f\n', max_violation);

fprintf('\nROOT CAUSE ANALYSIS:\n');
fprintf('====================\n');

fprintf('\n1. BISECTION SOLVER ISSUES:\n');
fprintf('   - Upper bound (1.0) too restrictive for calls\n');
fprintf('   - Lower bound (0.001) causing puts to hit floor\n');
fprintf('   - Poor initial guess leading to boundary convergence\n');

fprintf('\n2. NUMERICAL ISSUES:\n');
fprintf('   - Deep OTM options have near-zero values\n');
fprintf('   - Numerical precision limits in tree pricing\n');
fprintf('   - Willow tree discretization errors\n');

fprintf('\n3. MODEL CALIBRATION:\n');
fprintf('   - GARCH parameters may be causing extreme volatility clustering\n');
fprintf('   - Risk-neutral transformation (c = gamma + lambda + 0.5) issues\n');
fprintf('   - Tree construction parameters need adjustment\n');

fprintf('\nRECOMMENDATIONS:\n');
fprintf('================\n');

fprintf('\n1. IMMEDIATE FIXES:\n');
fprintf('   - Expand bisection bounds: [0.001, 3.0] instead of [0.001, 1.0]\n');
fprintf('   - Improve initial guess using Black-Scholes approximation\n');
fprintf('   - Add convergence diagnostics and error handling\n');

fprintf('\n2. MODEL IMPROVEMENTS:\n');
fprintf('   - Increase tree resolution (m_x, m_h) for better accuracy\n');
fprintf('   - Verify Willow tree probability calibration\n');
fprintf('   - Check American vs European option pricing consistency\n');

fprintf('\n3. PARAMETER VALIDATION:\n');
fprintf('   - Verify GARCH parameters are reasonable\n');
fprintf('   - Test with known analytical solutions (European case)\n');
fprintf('   - Cross-validate with Monte Carlo pricing\n');

fprintf('\n4. DATA QUALITY:\n');
fprintf('   - Filter out deep OTM options (m < 0.8 or m > 1.3)\n');
fprintf('   - Focus on liquid strike ranges\n');
fprintf('   - Add bid-ask spread considerations\n');

fprintf('\nPRIORITY ACTIONS:\n');
fprintf('=================\n');
fprintf('1. Fix bisection bounds in impvol.m\n');
fprintf('2. Add debugging output to trace solver behavior\n');
fprintf('3. Validate put-call parity for European options first\n');
fprintf('4. Test with simpler Black-Scholes model as baseline\n');

fprintf('\n=======================================================\n');
fprintf('Analysis complete. See impvol.m and demo.m for fixes.\n');
fprintf('=======================================================\n');
