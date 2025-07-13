% analyze_smile_patterns.m
% Comprehensive analysis and visualization of volatility smile patterns
% This script analyzes the realistic volatility smile dataset

clear all; close all;

fprintf('=================================================================\n');
fprintf('           VOLATILITY SMILE PATTERN ANALYSIS\n');
fprintf('=================================================================\n\n');

% Load the dataset with smile patterns
try
    data = dlmread('impl_demo_with_smile.csv', ',', 1, 0);
    fprintf('✓ Dataset loaded: %d options\n', size(data, 1));
catch
    fprintf('✗ Error loading dataset (impl_demo_with_smile.csv)\n');
    fprintf('Please run generate_dataset_with_smile.m first\n');
    return;
end

% Extract columns
% Headers: S0,m,r,T,corp,alpha,beta,omega,gamma,lambda,sigma,V,converged,iterations,pcp_error,log_m
S0 = data(:,1);
m = data(:,2);        % moneyness
r = data(:,3);
T = data(:,4);        % time to maturity
corp = data(:,5);     % 1=call, -1=put
sigma = data(:,11);   % implied volatility
V = data(:,12);       % option value
log_m = data(:,16);   % log moneyness

% Separate calls and puts
call_idx = (corp == 1);
put_idx = (corp == -1);

% Get unique values
unique_T = unique(T);
unique_m = unique(m);

fprintf('Dataset overview:\n');
fprintf('  Total options: %d (%d calls, %d puts)\n', length(sigma), sum(call_idx), sum(put_idx));
fprintf('  Maturities: %d (%.3f to %.3f years)\n', length(unique_T), min(T), max(T));
fprintf('  Moneyness range: %.2f to %.2f\n', min(m), max(m));
fprintf('  IV range: %.1f%% to %.1f%% (annualized)\n', min(sigma)*sqrt(252)*100, max(sigma)*sqrt(252)*100);

fprintf('\n=================================================================\n');
fprintf('                  VOLATILITY SMILE ANALYSIS\n');
fprintf('=================================================================\n');

% Create smile curves for each maturity
fprintf('\nVolatility Smile by Maturity:\n');
fprintf('-----------------------------\n');

figure('Position', [100, 100, 1400, 1000]);

% Colors for different maturities
colors = ['b'; 'r'; 'g'; 'm'; 'c'];
markers = ['o'; 's'; '^'; 'd'; 'v'];

% Plot 1: Volatility Smile Curves
subplot(2, 3, 1);
hold on;
grid on;

legend_entries = {};
smile_stats = [];

for i = 1:length(unique_T)
    T_val = unique_T(i);
    color = colors(i);
    marker = markers(i);

    % Get all options for this maturity
    T_mask = (T == T_val);
    T_moneyness = m(T_mask);
    T_sigma = sigma(T_mask);
    T_corp = corp(T_mask);

    % Combine calls and puts for smoother curve
    [sorted_m, sort_idx] = sort(T_moneyness);
    sorted_sigma = T_sigma(sort_idx);

    % Plot the smile curve
    plot(sorted_m, sorted_sigma * sqrt(252) * 100, ...
         [color marker '-'], 'LineWidth', 2, 'MarkerSize', 6);

    legend_entries{i} = sprintf('T=%.3f years', T_val);

    % Calculate smile statistics
    atm_idx = find(abs(sorted_m - 1.0) == min(abs(sorted_m - 1.0)), 1);
    atm_vol = sorted_sigma(atm_idx);

    otm_put_idx = find(sorted_m < 0.9);
    otm_call_idx = find(sorted_m > 1.1);

    if ~isempty(otm_put_idx)
        otm_put_vol = mean(sorted_sigma(otm_put_idx));
    else
        otm_put_vol = atm_vol;
    end

    if ~isempty(otm_call_idx)
        otm_call_vol = mean(sorted_sigma(otm_call_idx));
    else
        otm_call_vol = atm_vol;
    end

    smile_stats = [smile_stats; T_val, atm_vol, otm_put_vol, otm_call_vol, otm_put_vol - otm_call_vol];

    fprintf('T=%.3f: ATM=%.1f%%, OTM Put=%.1f%%, OTM Call=%.1f%%, Skew=%.1f%%\n', ...
        T_val, atm_vol*sqrt(252)*100, otm_put_vol*sqrt(252)*100, ...
        otm_call_vol*sqrt(252)*100, (otm_put_vol - otm_call_vol)*sqrt(252)*100);
end

xlabel('Moneyness (K/S)');
ylabel('Implied Volatility (% annual)');
title('Volatility Smile by Maturity');
legend(legend_entries, 'Location', 'northeast');
xlim([min(m) - 0.02, max(m) + 0.02]);

% Plot 2: Call vs Put IV Comparison
subplot(2, 3, 2);
hold on;
grid on;

call_m = m(call_idx);
call_sigma = sigma(call_idx);
put_m = m(put_idx);
put_sigma = sigma(put_idx);

scatter(call_m, call_sigma * sqrt(252) * 100, 50, 'b', 'filled', 'o');
scatter(put_m, put_sigma * sqrt(252) * 100, 50, 'r', 'filled', 's');

xlabel('Moneyness (K/S)');
ylabel('Implied Volatility (% annual)');
title('Call vs Put Implied Volatilities');
legend({'Calls', 'Puts'}, 'Location', 'northeast');

% Plot 3: Volatility Skew (Put - Call IV)
subplot(2, 3, 3);
hold on;
grid on;

for i = 1:length(unique_T)
    T_val = unique_T(i);
    color = colors(i);

    skew_data = [];
    moneyness_data = [];

    for j = 1:length(unique_m)
        m_val = unique_m(j);

        call_mask = call_idx & (T == T_val) & (m == m_val);
        put_mask = put_idx & (T == T_val) & (m == m_val);

        if sum(call_mask) > 0 && sum(put_mask) > 0
            call_iv = sigma(call_mask);
            put_iv = sigma(put_mask);
            skew = put_iv - call_iv;

            skew_data = [skew_data, skew * sqrt(252) * 100];
            moneyness_data = [moneyness_data, m_val];
        end
    end

    if ~isempty(skew_data)
        plot(moneyness_data, skew_data, [color 'o-'], 'LineWidth', 2, 'MarkerSize', 6);
    end
end

xlabel('Moneyness (K/S)');
ylabel('Skew: Put IV - Call IV (% annual)');
title('Volatility Skew by Maturity');
legend(legend_entries, 'Location', 'northeast');

% Plot 4: ATM Term Structure
subplot(2, 3, 4);
hold on;
grid on;

atm_vols_by_maturity = smile_stats(:, 2) * sqrt(252) * 100;
plot(unique_T, atm_vols_by_maturity, 'ko-', 'LineWidth', 2, 'MarkerSize', 8);

xlabel('Time to Maturity (years)');
ylabel('ATM Implied Volatility (% annual)');
title('ATM Volatility Term Structure');

% Plot 5: Smile Intensity by Maturity
subplot(2, 3, 5);
hold on;
grid on;

% Calculate smile intensity (difference between wing vols and ATM)
left_wing_intensity = (smile_stats(:, 3) - smile_stats(:, 2)) * sqrt(252) * 100;
right_wing_intensity = (smile_stats(:, 4) - smile_stats(:, 2)) * sqrt(252) * 100;

plot(unique_T, left_wing_intensity, 'ro-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'OTM Puts vs ATM');
plot(unique_T, right_wing_intensity, 'bo-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'OTM Calls vs ATM');
plot(unique_T, zeros(size(unique_T)), 'k--', 'LineWidth', 1);

xlabel('Time to Maturity (years)');
ylabel('IV Premium over ATM (% annual)');
title('Smile Wing Intensity');
legend('Location', 'northeast');

% Plot 6: Overall Skew Term Structure
subplot(2, 3, 6);
hold on;
grid on;

overall_skew = smile_stats(:, 5) * sqrt(252) * 100;
plot(unique_T, overall_skew, 'mo-', 'LineWidth', 2, 'MarkerSize', 8);
plot(unique_T, zeros(size(unique_T)), 'k--', 'LineWidth', 1);

xlabel('Time to Maturity (years)');
ylabel('Volatility Skew (% annual)');
title('Skew Term Structure');

sgtitle('Comprehensive Volatility Smile Analysis', 'FontSize', 16, 'FontWeight', 'bold');

% Print detailed statistics
fprintf('\n=================================================================\n');
fprintf('                 DETAILED SMILE STATISTICS\n');
fprintf('=================================================================\n');

fprintf('\nSMILE SHAPE CHARACTERISTICS:\n');
fprintf('Maturity    ATM Vol    Wing Spread    Skew      Smile Min\n');
fprintf('--------    -------    -----------    ----      ---------\n');

for i = 1:size(smile_stats, 1)
    T_val = smile_stats(i, 1);
    atm_vol = smile_stats(i, 2) * sqrt(252) * 100;
    wing_spread = max(smile_stats(i, 3), smile_stats(i, 4)) - min(smile_stats(i, 3), smile_stats(i, 4));
    wing_spread = wing_spread * sqrt(252) * 100;
    skew = smile_stats(i, 5) * sqrt(252) * 100;

    % Find minimum volatility for this maturity
    T_mask = (T == T_val);
    min_vol = min(sigma(T_mask)) * sqrt(252) * 100;

    fprintf('%.3f       %.1f%%       %.1f%%         %.1f%%     %.1f%%\n', ...
        T_val, atm_vol, wing_spread, skew, min_vol);
end

% Calculate moneyness-based statistics
fprintf('\nMONEYNESS-BASED ANALYSIS:\n');
fprintf('Moneyness    Avg IV     Call IV    Put IV     C-P Spread\n');
fprintf('---------    ------     -------    ------     ----------\n');

for j = 1:length(unique_m)
    m_val = unique_m(j);

    all_m_mask = (m == m_val);
    call_m_mask = call_idx & all_m_mask;
    put_m_mask = put_idx & all_m_mask;

    if sum(all_m_mask) > 0
        avg_iv = mean(sigma(all_m_mask)) * sqrt(252) * 100;

        if sum(call_m_mask) > 0
            call_iv = mean(sigma(call_m_mask)) * sqrt(252) * 100;
        else
            call_iv = NaN;
        end

        if sum(put_m_mask) > 0
            put_iv = mean(sigma(put_m_mask)) * sqrt(252) * 100;
        else
            put_iv = NaN;
        end

        if ~isnan(call_iv) && ~isnan(put_iv)
            spread = call_iv - put_iv;
        else
            spread = NaN;
        end

        fprintf('%.2f         %.1f%%      %.1f%%     %.1f%%      %.1f%%\n', ...
            m_val, avg_iv, call_iv, put_iv, spread);
    end
end

% Market realism assessment
fprintf('\n=================================================================\n');
fprintf('                  MARKET REALISM ASSESSMENT\n');
fprintf('=================================================================\n');

% Check for realistic smile features
fprintf('\nREALISTIC SMILE FEATURES:\n');

% 1. Volatility smile (U-shape)
atm_zone = abs(m - 1.0) < 0.05;
wing_zone = (m < 0.85) | (m > 1.15);

if sum(atm_zone) > 0 && sum(wing_zone) > 0
    atm_avg = mean(sigma(atm_zone));
    wing_avg = mean(sigma(wing_zone));

    if wing_avg > atm_avg
        fprintf('✓ Volatility smile present (wings higher than ATM)\n');
        smile_intensity = (wing_avg - atm_avg) * sqrt(252) * 100;
        fprintf('  Smile intensity: %.1f%% (wing premium over ATM)\n', smile_intensity);
    else
        fprintf('✗ No volatility smile detected\n');
    end
end

% 2. Put skew (OTM puts > OTM calls)
otm_puts = (m < 0.9) & put_idx;
otm_calls = (m > 1.1) & call_idx;

if sum(otm_puts) > 0 && sum(otm_calls) > 0
    otm_put_avg = mean(sigma(otm_puts));
    otm_call_avg = mean(sigma(otm_calls));

    if otm_put_avg > otm_call_avg
        fprintf('✓ Put skew present (OTM puts > OTM calls)\n');
        skew_magnitude = (otm_put_avg - otm_call_avg) * sqrt(252) * 100;
        fprintf('  Skew magnitude: %.1f%%\n', skew_magnitude);
    else
        fprintf('✗ No put skew detected\n');
    end
end

% 3. Term structure effects
if length(unique_T) > 1
    short_term = smile_stats(1, 5);  % First maturity skew
    long_term = smile_stats(end, 5); % Last maturity skew

    if abs(short_term) > abs(long_term)
        fprintf('✓ Term structure effect (skew decreases with maturity)\n');
        fprintf('  Short-term skew: %.1f%%, Long-term skew: %.1f%%\n', ...
            short_term*sqrt(252)*100, long_term*sqrt(252)*100);
    else
        fprintf('~ Weak term structure effect\n');
    end
end

% 4. Reasonable volatility levels
min_iv_annual = min(sigma) * sqrt(252) * 100;
max_iv_annual = max(sigma) * sqrt(252) * 100;

if min_iv_annual > 10 && max_iv_annual < 200
    fprintf('✓ Realistic volatility range (%.0f%% - %.0f%%)\n', min_iv_annual, max_iv_annual);
elseif min_iv_annual > 5 && max_iv_annual < 300
    fprintf('~ Acceptable volatility range (%.0f%% - %.0f%%)\n', min_iv_annual, max_iv_annual);
else
    fprintf('⚠ Extreme volatility range (%.0f%% - %.0f%%)\n', min_iv_annual, max_iv_annual);
end

% Summary assessment
fprintf('\n=================================================================\n');
fprintf('                        SUMMARY\n');
fprintf('=================================================================\n');

total_checks = 4;
passed_checks = 0;

% Count realistic features
if sum(wing_zone) > 0 && sum(atm_zone) > 0 && mean(sigma(wing_zone)) > mean(sigma(atm_zone))
    passed_checks = passed_checks + 1;
end

if sum(otm_puts) > 0 && sum(otm_calls) > 0 && mean(sigma(otm_puts)) > mean(sigma(otm_calls))
    passed_checks = passed_checks + 1;
end

if length(unique_T) > 1 && abs(smile_stats(1, 5)) > abs(smile_stats(end, 5))
    passed_checks = passed_checks + 1;
end

if min_iv_annual > 5 && max_iv_annual < 300
    passed_checks = passed_checks + 1;
end

fprintf('\nOVERALL ASSESSMENT: %d/%d realistic features present\n', passed_checks, total_checks);

if passed_checks >= 3
    fprintf('🎉 EXCELLENT: Dataset exhibits realistic volatility smile patterns\n');
    fprintf('   Suitable for machine learning and quantitative research\n');
elseif passed_checks >= 2
    fprintf('👍 GOOD: Dataset shows most expected smile characteristics\n');
    fprintf('   Minor refinements could improve realism\n');
else
    fprintf('⚠️ NEEDS IMPROVEMENT: Limited smile realism detected\n');
    fprintf('   Consider adjusting smile generation parameters\n');
end

fprintf('\nKEY FINDINGS:\n');
fprintf('• Volatility range: %.0f%% - %.0f%% (annualized)\n', min_iv_annual, max_iv_annual);
fprintf('• Average skew: %.1f%% (OTM puts premium)\n', mean(smile_stats(:, 5)) * sqrt(252) * 100);
fprintf('• Smile intensity: %.1f%% (wing premium)\n', ...
    (mean([smile_stats(:, 3); smile_stats(:, 4)]) - mean(smile_stats(:, 2))) * sqrt(252) * 100);
fprintf('• Term structure: %s\n', length(unique_T) > 1 ? 'Present' : 'Single maturity');

fprintf('\n=================================================================\n');
fprintf('Volatility smile analysis completed.\n');
fprintf('Plots saved to current figure window.\n');
fprintf('=================================================================\n');
