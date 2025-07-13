% compare_datasets.m
% Comprehensive comparison between original and improved implied volatility datasets

clear all; close all;

fprintf('=================================================================\n');
fprintf('         COMPREHENSIVE DATASET COMPARISON ANALYSIS\n');
fprintf('=================================================================\n\n');

% Load original dataset
fprintf('Loading datasets...\n');
try
    original_data = dlmread('impl_demo.csv', ',', 1, 0);
    fprintf('✓ Original dataset loaded: %d options\n', size(original_data, 1));
catch
    fprintf('✗ Error loading original dataset (impl_demo.csv)\n');
    return;
end

% Load improved dataset
try
    improved_data = dlmread('impl_demo_improved.csv', ',', 1, 0);
    fprintf('✓ Improved dataset loaded: %d options\n', size(improved_data, 1));
catch
    fprintf('✗ Error loading improved dataset (impl_demo_improved.csv)\n');
    return;
end

% Extract data columns
% Original: S0,m,r,T,corp,alpha,beta,omega,gamma,lambda,sigma,V
% Improved: S0,m,r,T,corp,alpha,beta,omega,gamma,lambda,sigma,V,converged,iterations,pcp_error

orig_S0 = original_data(:,1);
orig_m = original_data(:,2);
orig_T = original_data(:,4);
orig_corp = original_data(:,5);
orig_sigma = original_data(:,11);
orig_V = original_data(:,12);

impr_S0 = improved_data(:,1);
impr_m = improved_data(:,2);
impr_T = improved_data(:,4);
impr_corp = improved_data(:,5);
impr_sigma = improved_data(:,11);
impr_V = improved_data(:,12);
impr_converged = improved_data(:,13);
impr_iterations = improved_data(:,14);
impr_pcp_error = improved_data(:,15);

% Separate calls and puts
orig_calls = (orig_corp == 1);
orig_puts = (orig_corp == -1);
impr_calls = (impr_corp == 1);
impr_puts = (impr_corp == -1);

fprintf('\n=================================================================\n');
fprintf('                     BASIC STATISTICS\n');
fprintf('=================================================================\n');

fprintf('\nDATASET OVERVIEW:\n');
fprintf('                        ORIGINAL    IMPROVED\n');
fprintf('                        --------    --------\n');
fprintf('Total options:          %8d    %8d\n', length(orig_sigma), length(impr_sigma));
fprintf('Call options:           %8d    %8d\n', sum(orig_calls), sum(impr_calls));
fprintf('Put options:            %8d    %8d\n', sum(orig_puts), sum(impr_puts));

% Convergence statistics (only available for improved)
if size(improved_data, 2) >= 13
    fprintf('Convergence rate:       %8s    %7.1f%%\n', 'N/A', 100*sum(impr_converged)/length(impr_converged));
    fprintf('Avg iterations:         %8s    %7.1f\n', 'N/A', mean(impr_iterations));
end

fprintf('\n=================================================================\n');
fprintf('                 IMPLIED VOLATILITY ANALYSIS\n');
fprintf('=================================================================\n');

% Calculate statistics for both datasets
orig_call_iv = orig_sigma(orig_calls);
orig_put_iv = orig_sigma(orig_puts);
orig_all_iv = orig_sigma;

impr_call_iv = impr_sigma(impr_calls);
impr_put_iv = impr_sigma(impr_puts);
impr_all_iv = impr_sigma;

fprintf('\nRAW IMPLIED VOLATILITIES:\n');
fprintf('                        ORIGINAL    IMPROVED    CHANGE\n');
fprintf('                        --------    --------    ------\n');
fprintf('Call IV (mean):         %8.4f    %8.4f    %+7.4f\n', mean(orig_call_iv), mean(impr_call_iv), mean(impr_call_iv) - mean(orig_call_iv));
fprintf('Call IV (std):          %8.4f    %8.4f    %+7.4f\n', std(orig_call_iv), std(impr_call_iv), std(impr_call_iv) - std(orig_call_iv));
fprintf('Put IV (mean):          %8.4f    %8.4f    %+7.4f\n', mean(orig_put_iv), mean(impr_put_iv), mean(impr_put_iv) - mean(orig_put_iv));
fprintf('Put IV (std):           %8.4f    %8.4f    %+7.4f\n', std(orig_put_iv), std(impr_put_iv), std(impr_put_iv) - std(orig_put_iv));

% IV Spread (the key metric)
orig_spread = mean(orig_call_iv) - mean(orig_put_iv);
impr_spread = mean(impr_call_iv) - mean(impr_put_iv);
spread_improvement = ((orig_spread - impr_spread) / orig_spread) * 100;

fprintf('\nIV SPREAD ANALYSIS:\n');
fprintf('Original spread:        %8.4f (%.1f%% annualized)\n', orig_spread, orig_spread*sqrt(252)*100);
fprintf('Improved spread:        %8.4f (%.1f%% annualized)\n', impr_spread, impr_spread*sqrt(252)*100);
fprintf('Improvement:            %8.4f (%.1f%% reduction)\n', orig_spread - impr_spread, spread_improvement);

fprintf('\nANNUALIZED VOLATILITIES (252 trading days):\n');
fprintf('                        ORIGINAL    IMPROVED    CHANGE\n');
fprintf('                        --------    --------    ------\n');
fprintf('Call IV (annual %%):      %7.1f%%    %7.1f%%    %+6.1f%%\n', mean(orig_call_iv)*sqrt(252)*100, mean(impr_call_iv)*sqrt(252)*100, (mean(impr_call_iv) - mean(orig_call_iv))*sqrt(252)*100);
fprintf('Put IV (annual %%):       %7.1f%%    %7.1f%%    %+6.1f%%\n', mean(orig_put_iv)*sqrt(252)*100, mean(impr_put_iv)*sqrt(252)*100, (mean(impr_put_iv) - mean(orig_put_iv))*sqrt(252)*100);
fprintf('Spread (annual %%):       %7.1f%%    %7.1f%%    %+6.1f%%\n', orig_spread*sqrt(252)*100, impr_spread*sqrt(252)*100, (impr_spread - orig_spread)*sqrt(252)*100);

fprintf('\n=================================================================\n');
fprintf('                    BOUNDARY ANALYSIS\n');
fprintf('=================================================================\n');

% Boundary analysis
orig_call_boundaries = sum(orig_call_iv <= 0.005 | orig_call_iv >= 0.99);
orig_put_boundaries = sum(orig_put_iv <= 0.005 | orig_put_iv >= 0.99);
impr_call_boundaries = sum(impr_call_iv <= 0.005 | impr_call_iv >= 2.95);
impr_put_boundaries = sum(impr_put_iv <= 0.005 | impr_put_iv >= 2.95);

fprintf('\nBOUNDARY HITS (options at solver limits):\n');
fprintf('                        ORIGINAL    IMPROVED    CHANGE\n');
fprintf('                        --------    --------    ------\n');
fprintf('Call boundary hits:     %8d    %8d    %+7d\n', orig_call_boundaries, impr_call_boundaries, impr_call_boundaries - orig_call_boundaries);
fprintf('Put boundary hits:      %8d    %8d    %+7d\n', orig_put_boundaries, impr_put_boundaries, impr_put_boundaries - orig_put_boundaries);
fprintf('Total boundary %%:       %7.1f%%    %7.1f%%    %+6.1f%%\n', ...
    100*(orig_call_boundaries + orig_put_boundaries)/length(orig_sigma), ...
    100*(impr_call_boundaries + impr_put_boundaries)/length(impr_sigma), ...
    100*((impr_call_boundaries + impr_put_boundaries)/length(impr_sigma) - (orig_call_boundaries + orig_put_boundaries)/length(orig_sigma)));

% Detailed boundary analysis
fprintf('\nDETAILED BOUNDARY ANALYSIS:\n');
fprintf('Original - Very low IV (≤0.02):  %d calls, %d puts\n', sum(orig_call_iv <= 0.02), sum(orig_put_iv <= 0.02));
fprintf('Original - High IV (≥0.99):      %d calls, %d puts\n', sum(orig_call_iv >= 0.99), sum(orig_put_iv >= 0.99));
fprintf('Improved - Very low IV (≤0.02):  %d calls, %d puts\n', sum(impr_call_iv <= 0.02), sum(impr_put_iv <= 0.02));
fprintf('Improved - High IV (≥0.99):      %d calls, %d puts\n', sum(impr_call_iv >= 0.99), sum(impr_put_iv >= 0.99));

fprintf('\n=================================================================\n');
fprintf('                  PUT-CALL PARITY ANALYSIS\n');
fprintf('=================================================================\n');

% Put-call parity analysis (for pairs with same K, T)
unique_T = unique(orig_T);
unique_m = unique(orig_m);

orig_pcp_violations = 0;
orig_max_pcp_error = 0;
orig_total_pairs = 0;

impr_pcp_violations = 0;
impr_max_pcp_error = 0;
impr_total_pairs = 0;

for i = 1:length(unique_T)
    for j = 1:length(unique_m)
        % Original dataset
        orig_call_idx = orig_calls & (orig_T == unique_T(i)) & (orig_m == unique_m(j));
        orig_put_idx = orig_puts & (orig_T == unique_T(i)) & (orig_m == unique_m(j));

        if sum(orig_call_idx) == 1 && sum(orig_put_idx) == 1
            orig_total_pairs = orig_total_pairs + 1;
            C = orig_V(orig_call_idx);
            P = orig_V(orig_put_idx);
            S = orig_S0(orig_call_idx);
            K = unique_m(j) * S;
            r = 0.05/252;  % From original demo
            T_val = unique_T(i);

            pcp_error = abs((C - P) - (S - K * exp(-r * T_val)));
            if pcp_error > 0.01
                orig_pcp_violations = orig_pcp_violations + 1;
            end
            orig_max_pcp_error = max(orig_max_pcp_error, pcp_error);
        end

        % Improved dataset
        impr_call_idx = impr_calls & (impr_T == unique_T(i)) & (impr_m == unique_m(j));
        impr_put_idx = impr_puts & (impr_T == unique_T(i)) & (impr_m == unique_m(j));

        if sum(impr_call_idx) == 1 && sum(impr_put_idx) == 1
            impr_total_pairs = impr_total_pairs + 1;
            C = impr_V(impr_call_idx);
            P = impr_V(impr_put_idx);
            S = impr_S0(impr_call_idx);
            K = unique_m(j) * S;
            r = 0.05/252;
            T_val = unique_T(i);

            pcp_error = abs((C - P) - (S - K * exp(-r * T_val)));
            if pcp_error > 0.01
                impr_pcp_violations = impr_pcp_violations + 1;
            end
            impr_max_pcp_error = max(impr_max_pcp_error, pcp_error);
        end
    end
end

fprintf('\nPUT-CALL PARITY VIOLATIONS:\n');
fprintf('                        ORIGINAL    IMPROVED    CHANGE\n');
fprintf('                        --------    --------    ------\n');
fprintf('Total pairs checked:    %8d    %8d    %+7d\n', orig_total_pairs, impr_total_pairs, impr_total_pairs - orig_total_pairs);
fprintf('Violations (>0.01):     %8d    %8d    %+7d\n', orig_pcp_violations, impr_pcp_violations, impr_pcp_violations - orig_pcp_violations);
fprintf('Violation rate:         %7.1f%%    %7.1f%%    %+6.1f%%\n', ...
    100*orig_pcp_violations/orig_total_pairs, 100*impr_pcp_violations/impr_total_pairs, ...
    100*(impr_pcp_violations/impr_total_pairs - orig_pcp_violations/orig_total_pairs));
fprintf('Maximum error:          %8.4f    %8.4f    %+7.4f\n', orig_max_pcp_error, impr_max_pcp_error, impr_max_pcp_error - orig_max_pcp_error);

fprintf('\n=================================================================\n');
fprintf('                   MATURITY ANALYSIS\n');
fprintf('=================================================================\n');

fprintf('\nIV SPREAD BY MATURITY:\n');
fprintf('Maturity   Original Spread   Improved Spread   Improvement\n');
fprintf('--------   ---------------   ---------------   -----------\n');

for i = 1:length(unique_T)
    T_val = unique_T(i);

    % Original data
    orig_T_calls = orig_call_iv(orig_T(orig_calls) == T_val);
    orig_T_puts = orig_put_iv(orig_T(orig_puts) == T_val);
    orig_T_spread = mean(orig_T_calls) - mean(orig_T_puts);

    % Improved data
    impr_T_calls = impr_call_iv(impr_T(impr_calls) == T_val);
    impr_T_puts = impr_put_iv(impr_T(impr_puts) == T_val);
    impr_T_spread = mean(impr_T_calls) - mean(impr_T_puts);

    improvement = ((orig_T_spread - impr_T_spread) / orig_T_spread) * 100;

    fprintf('%.3f      %15.4f   %15.4f   %10.1f%%\n', T_val, orig_T_spread, impr_T_spread, improvement);
end

fprintf('\n=================================================================\n');
fprintf('                  MONEYNESS ANALYSIS\n');
fprintf('=================================================================\n');

fprintf('\nIV PATTERNS BY MONEYNESS (Call - Put Spread):\n');
fprintf('Moneyness   Original Spread   Improved Spread   Improvement\n');
fprintf('---------   ---------------   ---------------   -----------\n');

for j = 1:length(unique_m)
    m_val = unique_m(j);

    % Original data
    orig_m_calls = orig_call_iv(orig_m(orig_calls) == m_val);
    orig_m_puts = orig_put_iv(orig_m(orig_puts) == m_val);
    if ~isempty(orig_m_calls) && ~isempty(orig_m_puts)
        orig_m_spread = mean(orig_m_calls) - mean(orig_m_puts);
    else
        orig_m_spread = 0;
    end

    % Improved data
    impr_m_calls = impr_call_iv(impr_m(impr_calls) == m_val);
    impr_m_puts = impr_put_iv(impr_m(impr_puts) == m_val);
    if ~isempty(impr_m_calls) && ~isempty(impr_m_puts)
        impr_m_spread = mean(impr_m_calls) - mean(impr_m_puts);
    else
        impr_m_spread = 0;
    end

    if orig_m_spread ~= 0
        improvement = ((orig_m_spread - impr_m_spread) / orig_m_spread) * 100;
    else
        improvement = 0;
    end

    fprintf('%.2f        %15.4f   %15.4f   %10.1f%%\n', m_val, orig_m_spread, impr_m_spread, improvement);
end

fprintf('\n=================================================================\n');
fprintf('                     QUALITY METRICS\n');
fprintf('=================================================================\n');

% Calculate various quality metrics
fprintf('\nVOLATILITY RANGES:\n');
fprintf('                        ORIGINAL              IMPROVED\n');
fprintf('                        --------              --------\n');
fprintf('Overall range:          [%.4f, %.4f]      [%.4f, %.4f]\n', min(orig_all_iv), max(orig_all_iv), min(impr_all_iv), max(impr_all_iv));
fprintf('Call range:             [%.4f, %.4f]      [%.4f, %.4f]\n', min(orig_call_iv), max(orig_call_iv), min(impr_call_iv), max(impr_call_iv));
fprintf('Put range:              [%.4f, %.4f]      [%.4f, %.4f]\n', min(orig_put_iv), max(orig_put_iv), min(impr_put_iv), max(impr_put_iv));

% Realistic ranges (annualized)
fprintf('\nANNUALIZED RANGES:\n');
fprintf('                        ORIGINAL              IMPROVED\n');
fprintf('                        --------              --------\n');
fprintf('Overall range:          [%.1f%%, %.1f%%]        [%.1f%%, %.1f%%]\n', ...
    min(orig_all_iv)*sqrt(252)*100, max(orig_all_iv)*sqrt(252)*100, ...
    min(impr_all_iv)*sqrt(252)*100, max(impr_all_iv)*sqrt(252)*100);

% Quality assessment
fprintf('\nQUALITY ASSESSMENT:\n');
realistic_orig = (orig_spread < 0.1);  % Less than 10% spread is realistic
realistic_impr = (impr_spread < 0.1);
if realistic_orig
    realistic_orig_str = 'YES';
else
    realistic_orig_str = 'NO';
end
if realistic_impr
    realistic_impr_str = 'YES';
else
    realistic_impr_str = 'NO';
end
fprintf('Realistic IV spread:    %-15s   %-15s\n', realistic_orig_str, realistic_impr_str);

boundary_rate_orig = 100*(orig_call_boundaries + orig_put_boundaries)/length(orig_sigma);
boundary_rate_impr = 100*(impr_call_boundaries + impr_put_boundaries)/length(impr_sigma);
low_boundary_orig = (boundary_rate_orig < 10);
low_boundary_impr = (boundary_rate_impr < 10);
if low_boundary_orig
    low_boundary_orig_str = 'YES';
else
    low_boundary_orig_str = 'NO';
end
if low_boundary_impr
    low_boundary_impr_str = 'YES';
else
    low_boundary_impr_str = 'NO';
end
fprintf('Low boundary rate:      %-15s   %-15s\n', low_boundary_orig_str, low_boundary_impr_str);

pcp_rate_orig = 100*orig_pcp_violations/orig_total_pairs;
pcp_rate_impr = 100*impr_pcp_violations/impr_total_pairs;
good_pcp_orig = (pcp_rate_orig < 20);
good_pcp_impr = (pcp_rate_impr < 20);
if good_pcp_orig
    good_pcp_orig_str = 'YES';
else
    good_pcp_orig_str = 'NO';
end
if good_pcp_impr
    good_pcp_impr_str = 'YES';
else
    good_pcp_impr_str = 'NO';
end
fprintf('Good PCP compliance:    %-15s   %-15s\n', good_pcp_orig_str, good_pcp_impr_str);

fprintf('\n=================================================================\n');
fprintf('                      SUMMARY\n');
fprintf('=================================================================\n');

% Overall assessment
total_improvements = 0;
if impr_spread < orig_spread
    total_improvements = total_improvements + 1;
    fprintf('✓ IV spread significantly reduced (%.1f%% improvement)\n', spread_improvement);
else
    fprintf('✗ IV spread not improved\n');
end

if boundary_rate_impr < boundary_rate_orig
    total_improvements = total_improvements + 1;
    fprintf('✓ Boundary issues reduced (%.1f%% → %.1f%%)\n', boundary_rate_orig, boundary_rate_impr);
else
    fprintf('✗ Boundary issues not reduced\n');
end

if pcp_rate_impr < pcp_rate_orig
    total_improvements = total_improvements + 1;
    fprintf('✓ Put-call parity violations reduced (%.1f%% → %.1f%%)\n', pcp_rate_orig, pcp_rate_impr);
else
    fprintf('✗ Put-call parity not improved\n');
end

if size(improved_data, 2) >= 13 && mean(impr_converged) > 0.9
    total_improvements = total_improvements + 1;
    fprintf('✓ High convergence rate achieved (%.1f%%)\n', 100*mean(impr_converged));
end

fprintf('\n');
if total_improvements >= 3
    fprintf('🎉 OVERALL ASSESSMENT: SIGNIFICANT IMPROVEMENT\n');
    fprintf('   The improved solver successfully addresses the main issues.\n');
elseif total_improvements >= 2
    fprintf('👍 OVERALL ASSESSMENT: MODERATE IMPROVEMENT\n');
    fprintf('   The improved solver shows promising results.\n');
else
    fprintf('⚠️  OVERALL ASSESSMENT: LIMITED IMPROVEMENT\n');
    fprintf('   Further refinements may be needed.\n');
end

fprintf('\nKEY RECOMMENDATIONS:\n');
if boundary_rate_impr > 10
    fprintf('• Consider expanding volatility bounds further\n');
end
if pcp_rate_impr > 20
    fprintf('• Investigate option pricing accuracy\n');
end
if impr_spread > 0.05
    fprintf('• Fine-tune initial guess methodology\n');
end

fprintf('\n=================================================================\n');
fprintf('Comparison analysis completed.\n');
fprintf('=================================================================\n');
