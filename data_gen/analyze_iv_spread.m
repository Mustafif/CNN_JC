% analyze_iv_spread.m
% Analysis of implied volatility spread between calls and puts

clear all; close all;

% Load the data (skip header row)
data = dlmread('impl_demo.csv', ',', 1, 0);

% Extract relevant columns (based on CSV structure)
% Headers: S0,m,r,T,corp,alpha,beta,omega,gamma,lambda,sigma,V
S0 = data(:,1);
m = data(:,2);  % moneyness
r = data(:,3);
T = data(:,4);
corp = data(:,5);  % 1 for call, -1 for put
alpha = data(:,6);
beta = data(:,7);
omega = data(:,8);
gamma = data(:,9);
lambda = data(:,10);
sigma = data(:,11);  % implied volatility
V = data(:,12);  % option value
K = m .* S0;  % strike price

% Separate calls and puts
call_idx = (corp == 1);
put_idx = (corp == -1);

% Note: No need for separate tables in Octave

fprintf('=== IMPLIED VOLATILITY ANALYSIS ===\n\n');

% Basic statistics
fprintf('Call Options:\n');
fprintf('  Count: %d\n', sum(call_idx));
fprintf('  IV Range: %.4f - %.4f\n', min(sigma(call_idx)), max(sigma(call_idx)));
fprintf('  IV Mean: %.4f\n', mean(sigma(call_idx)));
fprintf('  IV Std: %.4f\n', std(sigma(call_idx)));
fprintf('  Options with IV = 1.0: %d\n', sum(sigma(call_idx) == 1.0));
fprintf('  Options with IV > 0.8: %d\n', sum(sigma(call_idx) > 0.8));

fprintf('\nPut Options:\n');
fprintf('  Count: %d\n', sum(put_idx));
fprintf('  IV Range: %.4f - %.4f\n', min(sigma(put_idx)), max(sigma(put_idx)));
fprintf('  IV Mean: %.4f\n', mean(sigma(put_idx)));
fprintf('  IV Std: %.4f\n', std(sigma(put_idx)));
fprintf('  Options with IV < 0.1: %d\n', sum(sigma(put_idx) < 0.1));
fprintf('  Options with IV < 0.05: %d\n', sum(sigma(put_idx) < 0.05));

% Check for boundary hitting (common IV solver issue)
fprintf('\n=== BOUNDARY ANALYSIS ===\n');
fprintf('Calls hitting upper bound (IV=1.0): %d (%.1f%%)\n', ...
    sum(sigma(call_idx) == 1.0), 100*sum(sigma(call_idx) == 1.0)/sum(call_idx));
fprintf('Puts hitting lower bound (IV<0.01): %d (%.1f%%)\n', ...
    sum(sigma(put_idx) < 0.01), 100*sum(sigma(put_idx) < 0.01)/sum(put_idx));

% Analyze by maturity and moneyness
unique_T = unique(T);
unique_m = unique(m);

fprintf('\n=== MATURITY ANALYSIS ===\n');
for i = 1:length(unique_T)
    t_idx = (T == unique_T(i));
    call_t_idx = call_idx & t_idx;
    put_t_idx = put_idx & t_idx;

    fprintf('T = %.3f:\n', unique_T(i));
    fprintf('  Call IV: %.4f ± %.4f\n', mean(sigma(call_t_idx)), std(sigma(call_t_idx)));
    fprintf('  Put IV:  %.4f ± %.4f\n', mean(sigma(put_t_idx)), std(sigma(put_t_idx)));
    fprintf('  IV Spread: %.4f\n', mean(sigma(call_t_idx)) - mean(sigma(put_t_idx)));
end

fprintf('\n=== MONEYNESS ANALYSIS ===\n');
for i = 1:length(unique_m)
    m_idx = (m == unique_m(i));
    call_m_idx = call_idx & m_idx;
    put_m_idx = put_idx & m_idx;

    fprintf('m = %.2f:\n', unique_m(i));
    if sum(call_m_idx) > 0 && sum(put_m_idx) > 0
        fprintf('  Call IV: %.4f ± %.4f\n', mean(sigma(call_m_idx)), std(sigma(call_m_idx)));
        fprintf('  Put IV:  %.4f ± %.4f\n', mean(sigma(put_m_idx)), std(sigma(put_m_idx)));
        fprintf('  IV Spread: %.4f\n', mean(sigma(call_m_idx)) - mean(sigma(put_m_idx)));
    end
end

% Check put-call parity (for European options, this should hold approximately)
fprintf('\n=== PUT-CALL PARITY CHECK ===\n');
% For each matching pair, check if C - P ≈ S - K*exp(-rT)
for i = 1:length(unique_T)
    for j = 1:length(unique_m)
        t_idx = (T == unique_T(i));
        m_idx = (m == unique_m(j));

        call_match = call_idx & t_idx & m_idx;
        put_match = put_idx & t_idx & m_idx;

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

            if pcp_error > 0.01  % Flag significant violations
                fprintf('PCP Violation: T=%.3f, m=%.2f, Error=%.4f\n', ...
                    unique_T(i), unique_m(j), pcp_error);
            end
        end
    end
end

% Create visualization
figure('Position', [100, 100, 1200, 800]);

% Plot 1: IV by moneyness for different maturities
subplot(2,3,1);
colors = lines(length(unique_T));
hold on;
for i = 1:length(unique_T)
    t_idx = (T == unique_T(i));
    call_t = call_idx & t_idx;
    put_t = put_idx & t_idx;

    if sum(call_t) > 0
        plot(m(call_t), sigma(call_t), 'o-', 'Color', colors(i,:), ...
            'MarkerFaceColor', colors(i,:), 'LineWidth', 1.5);
    end
    if sum(put_t) > 0
        plot(m(put_t), sigma(put_t), 's--', 'Color', colors(i,:), ...
            'MarkerFaceColor', colors(i,:), 'LineWidth', 1.5);
    end
end
xlabel('Moneyness (K/S0)');
ylabel('Implied Volatility');
title('Implied Volatility vs Moneyness');
legend_labels = {};
for i = 1:length(unique_T)
    legend_labels{end+1} = sprintf('Call T=%.3f', unique_T(i));
    legend_labels{end+1} = sprintf('Put T=%.3f', unique_T(i));
end
legend(legend_labels, 'Location', 'best');
grid on;

% Plot 2: Option values by moneyness
subplot(2,3,2);
hold on;
for i = 1:length(unique_T)
    t_idx = (T == unique_T(i));
    call_t = call_idx & t_idx;
    put_t = put_idx & t_idx;

    if sum(call_t) > 0
        plot(m(call_t), V(call_t), 'o-', 'Color', colors(i,:), ...
            'MarkerFaceColor', colors(i,:), 'LineWidth', 1.5);
    end
    if sum(put_t) > 0
        plot(m(put_t), V(put_t), 's--', 'Color', colors(i,:), ...
            'MarkerFaceColor', colors(i,:), 'LineWidth', 1.5);
    end
end
xlabel('Moneyness (K/S0)');
ylabel('Option Value');
title('Option Values vs Moneyness');
grid on;

% Plot 3: IV spread by maturity
subplot(2,3,3);
iv_spreads = [];
for i = 1:length(unique_T)
    t_idx = (T == unique_T(i));
    call_iv_mean = mean(sigma(call_idx & t_idx));
    put_iv_mean = mean(sigma(put_idx & t_idx));
    iv_spreads(i) = call_iv_mean - put_iv_mean;
end
bar(unique_T, iv_spreads);
xlabel('Time to Maturity');
ylabel('IV Spread (Call - Put)');
title('IV Spread by Maturity');
grid on;

% Plot 4: Distribution of IVs
subplot(2,3,4);
histogram(sigma(call_idx), 20, 'Alpha', 0.7, 'DisplayName', 'Calls');
hold on;
histogram(sigma(put_idx), 20, 'Alpha', 0.7, 'DisplayName', 'Puts');
xlabel('Implied Volatility');
ylabel('Frequency');
title('Distribution of Implied Volatilities');
legend;
grid on;

% Plot 5: IV vs Option Value
subplot(2,3,5);
scatter(V(call_idx), sigma(call_idx), 50, 'filled', 'DisplayName', 'Calls');
hold on;
scatter(V(put_idx), sigma(put_idx), 50, 'filled', 'DisplayName', 'Puts');
xlabel('Option Value');
ylabel('Implied Volatility');
title('IV vs Option Value');
legend;
grid on;

% Plot 6: Problematic cases
subplot(2,3,6);
% Highlight boundary cases
boundary_calls = call_idx & (sigma >= 0.99);  % Near upper bound
boundary_puts = put_idx & (sigma <= 0.02);    % Near lower bound

if sum(boundary_calls) > 0
    scatter(m(boundary_calls), sigma(boundary_calls), 100, 'r', 'filled', ...
        'DisplayName', 'Boundary Calls');
    hold on;
end
if sum(boundary_puts) > 0
    scatter(m(boundary_puts), sigma(boundary_puts), 100, 'b', 'filled', ...
        'DisplayName', 'Boundary Puts');
end
xlabel('Moneyness');
ylabel('Implied Volatility');
title('Boundary Cases');
legend;
grid on;

sgtitle('Implied Volatility Analysis');

% Save the analysis
fprintf('\n=== RECOMMENDATIONS ===\n');
fprintf('1. Many calls have IV = 1.0, suggesting the solver hits upper bounds\n');
fprintf('2. Many puts have very low IVs, suggesting lower bound issues\n');
fprintf('3. Consider:\n');
fprintf('   - Expanding the search bounds in impvol.m\n');
fprintf('   - Improving initial guess for bisection method\n');
fprintf('   - Checking option pricing accuracy\n');
fprintf('   - Verifying tree construction parameters\n');
fprintf('4. Put-call parity violations may indicate pricing issues\n');

fprintf('\nAnalysis complete. Plots saved to figure window.\n');
