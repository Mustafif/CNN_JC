% plot_volatility_smile.m
% Comprehensive plotting script for volatility smile visualization
% Creates multiple plots showing realistic smile patterns

clear all; close all;

fprintf('=================================================================\n');
fprintf('            VOLATILITY SMILE VISUALIZATION\n');
fprintf('=================================================================\n\n');

% Load the dataset
try
    data = dlmread('impl_demo_with_smile.csv', ',', 1, 0);
    fprintf('Dataset loaded: %d options\n', size(data, 1));
catch
    fprintf('Error loading dataset. Please run generate_dataset_with_smile.m first\n');
    return;
end

% Extract data
S0 = data(:,1);
m = data(:,2);        % moneyness
T = data(:,4);        % time to maturity
corp = data(:,5);     % 1=call, -1=put
sigma = data(:,11);   % implied volatility (daily)

% Convert to annualized percentage
sigma_annual = sigma * sqrt(252) * 100;

% Separate calls and puts
call_idx = (corp == 1);
put_idx = (corp == -1);

% Get unique values
unique_T = unique(T);
unique_m = unique(m);

fprintf('Creating volatility smile plots...\n');

% Create figure with multiple subplots
figure('Position', [100, 100, 1400, 1000]);

% Colors and markers for different maturities
colors = ['b'; 'r'; 'g'; 'm'; 'c'];
line_styles = {'-', '--', '-.', ':', '-'};
markers = ['o'; 's'; '^'; 'd'; 'v'];

%% Plot 1: Main Volatility Smile Curves
subplot(2, 3, 1);
hold on;
grid on;

legend_entries = {};
for i = 1:length(unique_T)
    T_val = unique_T(i);

    % Get data for this maturity
    T_mask = (T == T_val);
    T_moneyness = m(T_mask);
    T_sigma = sigma_annual(T_mask);

    % Sort by moneyness for smooth curves
    [sorted_m, sort_idx] = sort(T_moneyness);
    sorted_sigma = T_sigma(sort_idx);

    % Plot smile curve
    color_idx = mod(i-1, length(colors)) + 1;
    plot(sorted_m, sorted_sigma, ...
         [colors(color_idx) markers(color_idx) line_styles{color_idx}], ...
         'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', colors(color_idx));

    legend_entries{i} = sprintf('T=%.0f days', T_val * 252);
end

xlabel('Moneyness (K/S)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Implied Volatility (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Volatility Smile by Maturity', 'FontSize', 14, 'FontWeight', 'bold');
legend(legend_entries, 'Location', 'northeast', 'FontSize', 10);
xlim([min(m) - 0.02, max(m) + 0.02]);
ylim([min(sigma_annual) - 20, max(sigma_annual) + 20]);

%% Plot 2: Call vs Put Scatter
subplot(2, 3, 2);
hold on;
grid on;

% Plot calls and puts separately
scatter(m(call_idx), sigma_annual(call_idx), 60, 'b', 'filled', 'o', 'DisplayName', 'Call Options');
scatter(m(put_idx), sigma_annual(put_idx), 60, 'r', 'filled', 's', 'DisplayName', 'Put Options');

xlabel('Moneyness (K/S)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Implied Volatility (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Call vs Put Implied Volatilities', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 10);

%% Plot 3: Volatility Skew
subplot(2, 3, 3);
hold on;
grid on;

for i = 1:length(unique_T)
    T_val = unique_T(i);

    skew_data = [];
    moneyness_data = [];

    % Calculate skew for each moneyness level
    for j = 1:length(unique_m)
        m_val = unique_m(j);

        call_mask = call_idx & (T == T_val) & (m == m_val);
        put_mask = put_idx & (T == T_val) & (m == m_val);

        if sum(call_mask) > 0 && sum(put_mask) > 0
            call_iv = sigma_annual(call_mask);
            put_iv = sigma_annual(put_mask);
            skew = put_iv - call_iv;  % Put premium

            skew_data = [skew_data, skew];
            moneyness_data = [moneyness_data, m_val];
        end
    end

    if ~isempty(skew_data)
        color_idx = mod(i-1, length(colors)) + 1;
        plot(moneyness_data, skew_data, ...
             [colors(color_idx) markers(color_idx) line_styles{color_idx}], ...
             'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', colors(color_idx));
    end
end

% Add zero line
plot([min(m), max(m)], [0, 0], 'k--', 'LineWidth', 1);

xlabel('Moneyness (K/S)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Skew: Put IV - Call IV (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Volatility Skew by Maturity', 'FontSize', 14, 'FontWeight', 'bold');
legend(legend_entries, 'Location', 'northeast', 'FontSize', 10);

%% Plot 4: ATM Term Structure
subplot(2, 3, 4);
hold on;
grid on;

atm_vols = [];
for i = 1:length(unique_T)
    T_val = unique_T(i);
    T_mask = (T == T_val);
    T_moneyness = m(T_mask);
    T_sigma = sigma_annual(T_mask);

    % Find ATM volatility (closest to moneyness = 1.0)
    [~, atm_idx] = min(abs(T_moneyness - 1.0));
    atm_vol = T_sigma(atm_idx);
    atm_vols = [atm_vols, atm_vol];
end

plot(unique_T * 252, atm_vols, 'ko-', 'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', 'k');

xlabel('Days to Maturity', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('ATM Implied Volatility (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('ATM Volatility Term Structure', 'FontSize', 14, 'FontWeight', 'bold');

%% Plot 5: Smile Intensity Evolution
subplot(2, 3, 5);
hold on;
grid on;

smile_intensities = [];
skew_values = [];

for i = 1:length(unique_T)
    T_val = unique_T(i);
    T_mask = (T == T_val);
    T_moneyness = m(T_mask);
    T_sigma = sigma_annual(T_mask);
    T_corp = corp(T_mask);

    % Find ATM
    [~, atm_idx] = min(abs(T_moneyness - 1.0));
    atm_vol = T_sigma(atm_idx);

    % Find wings
    wing_vols = T_sigma((T_moneyness <= 0.85) | (T_moneyness >= 1.15));
    if ~isempty(wing_vols)
        max_wing_vol = max(wing_vols);
        smile_intensity = max_wing_vol - atm_vol;
    else
        smile_intensity = 0;
    end
    smile_intensities = [smile_intensities, smile_intensity];

    % Calculate average skew
    otm_puts = T_sigma((T_moneyness <= 0.9) & (T_corp == -1));
    otm_calls = T_sigma((T_moneyness >= 1.1) & (T_corp == 1));

    if ~isempty(otm_puts) && ~isempty(otm_calls)
        avg_skew = mean(otm_puts) - mean(otm_calls);
    else
        avg_skew = 0;
    end
    skew_values = [skew_values, avg_skew];
end

% Plot both metrics
yyaxis left;
plot(unique_T * 252, smile_intensities, 'bo-', 'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
ylabel('Smile Intensity (%)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');

yyaxis right;
plot(unique_T * 252, skew_values, 'rs-', 'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
ylabel('Average Skew (%)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');

xlabel('Days to Maturity', 'FontSize', 12, 'FontWeight', 'bold');
title('Smile Evolution with Maturity', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Smile Intensity', 'Put Skew'}, 'Location', 'northeast', 'FontSize', 10);

%% Plot 6: Moneyness Distribution
subplot(2, 3, 6);
hold on;
grid on;

% Create moneyness bins and calculate average IV
moneyness_bins = 0.7:0.05:1.3;
bin_centers = moneyness_bins(1:end-1) + 0.025;
bin_ivs = [];

for i = 1:length(bin_centers)
    bin_mask = (m >= moneyness_bins(i)) & (m < moneyness_bins(i+1));
    if sum(bin_mask) > 0
        bin_ivs = [bin_ivs, mean(sigma_annual(bin_mask))];
    else
        bin_ivs = [bin_ivs, NaN];
    end
end

% Remove NaN values
valid_idx = ~isnan(bin_ivs);
valid_centers = bin_centers(valid_idx);
valid_ivs = bin_ivs(valid_idx);

bar(valid_centers, valid_ivs, 0.8, 'FaceColor', [0.3, 0.7, 0.9], 'EdgeColor', 'k', 'LineWidth', 1);

% Overlay smooth curve
if length(valid_centers) > 3
    smooth_x = linspace(min(valid_centers), max(valid_centers), 100);
    smooth_y = interp1(valid_centers, valid_ivs, smooth_x, 'spline');
    plot(smooth_x, smooth_y, 'r-', 'LineWidth', 3);
end

xlabel('Moneyness (K/S)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Average IV (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('IV Distribution by Moneyness', 'FontSize', 14, 'FontWeight', 'bold');

% Add vertical line at ATM
plot([1, 1], [min(valid_ivs) - 10, max(valid_ivs) + 10], 'k--', 'LineWidth', 2);
text(1.02, mean(valid_ivs), 'ATM', 'FontSize', 10, 'FontWeight', 'bold');

% Overall title
sgtitle('Comprehensive Volatility Smile Analysis', 'FontSize', 16, 'FontWeight', 'bold');

% Print summary statistics
fprintf('\n=================================================================\n');
fprintf('                    SMILE VISUALIZATION SUMMARY\n');
fprintf('=================================================================\n');

fprintf('\nKey Smile Characteristics:\n');
fprintf('  Overall IV range: %.0f%% - %.0f%%\n', min(sigma_annual), max(sigma_annual));
fprintf('  ATM volatilities: %.0f%% - %.0f%% across maturities\n', min(atm_vols), max(atm_vols));
fprintf('  Maximum smile intensity: %.0f%%\n', max(smile_intensities));
fprintf('  Maximum skew: %.0f%%\n', max(skew_values));

fprintf('\nSmile Features by Maturity:\n');
for i = 1:length(unique_T)
    fprintf('  %d days: ATM=%.0f%%, Smile=%.0f%%, Skew=%.0f%%\n', ...
        round(unique_T(i) * 252), atm_vols(i), smile_intensities(i), skew_values(i));
end

fprintf('\nPlot Elements:\n');
fprintf('  Plot 1: Smile curves by maturity (main visualization)\n');
fprintf('  Plot 2: Call vs put scatter (shows option type differences)\n');
fprintf('  Plot 3: Skew patterns (put premium over calls)\n');
fprintf('  Plot 4: ATM term structure (volatility vs time)\n');
fprintf('  Plot 5: Smile evolution (intensity and skew vs maturity)\n');
fprintf('  Plot 6: Moneyness distribution (IV across strikes)\n');

fprintf('\n=================================================================\n');
fprintf('Volatility smile plots generated successfully!\n');
fprintf('The plots show realistic market-like smile patterns with:\n');
fprintf('• Clear U-shaped volatility smiles\n');
fprintf('• Positive put skew (OTM puts > OTM calls)\n');
fprintf('• Term structure effects (smile flattens with maturity)\n');
fprintf('• Smooth, market-realistic curves\n');
fprintf('=================================================================\n');

% Adjust layout
set(gcf, 'Color', 'white');
