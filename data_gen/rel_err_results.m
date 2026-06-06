function rel_err_results(varargin)
% REL_ERR_RESULTS Compute relative error analysis for parameter comparison
%
% SYNTAX:
%   rel_err_results()  % Uses default 'compare' folder
%   rel_err_results('folder', 'compare')
%   rel_err_results('folder', 'compare', 'plot', true)
%
% OPTIONAL NAME-VALUE PARAMETERS:
%   'folder' - Folder containing the CSV files (default: 'compare')
%   'plot'   - Whether to generate plots (default: false)
%   'save'   - Whether to save results table (default: true)
%
% DESCRIPTION:
%   Analyzes the relative error between initial/calibrated parameters
%   vs true parameters. Computes mean relative error in implied volatility
%   grouped by maturity and moneyness categories.
%
% EXAMPLE:
%   rel_err_results()
%   rel_err_results('folder', 'my_comparison', 'plot', true)

% Parse input arguments
p = inputParser;
addParameter(p, 'folder', 'compare2', @ischar);
addParameter(p, 'plot', false, @islogical);
addParameter(p, 'save', true, @islogical);
parse(p, varargin{:});

folder = p.Results.folder;
do_plot = p.Results.plot;
do_save = p.Results.save;

fprintf('=================================================================\n');
fprintf('         RELATIVE ERROR ANALYSIS FOR PARAMETER COMPARISON\n');
fprintf('=================================================================\n\n');

% Check if folder exists
if ~exist(folder, 'dir')
    error('Folder not found: %s', folder);
end

fprintf('Reading data from folder: %s\n', folder);

%% LOAD DATA
init_file = fullfile(folder, 'initial_params.csv');
calib_file = fullfile(folder, 'calibrated_params.csv');
true_file = fullfile(folder, 'true_params.csv');

% Check if files exist
if ~exist(init_file, 'file')
    error('File not found: %s', init_file);
end
if ~exist(calib_file, 'file')
    error('File not found: %s', calib_file);
end
if ~exist(true_file, 'file')
    error('File not found: %s', true_file);
end

fprintf('Loading datasets...\n');

% Use csvread for Octave compatibility
% Column order: S0,m,r,T,corp,alpha,beta,omega,gamma,lambda,sigma,V
init_data = csvread(init_file, 1, 0);  % Skip header
calib_data = csvread(calib_file, 1, 0);  % Skip header
true_data = csvread(true_file, 1, 0);  % Skip header

% Convert to structure for easier access
init.S0 = init_data(:, 1);
init.m = init_data(:, 2);
init.r = init_data(:, 3);
init.T = init_data(:, 4);
init.corp = init_data(:, 5);
init.sigma = init_data(:, 11);
init.V = init_data(:, 12);

calib.S0 = calib_data(:, 1);
calib.m = calib_data(:, 2);
calib.r = calib_data(:, 3);
calib.T = calib_data(:, 4);
calib.corp = calib_data(:, 5);
calib.sigma = calib_data(:, 11);
calib.V = calib_data(:, 12);

true_.S0 = true_data(:, 1);
true_.m = true_data(:, 2);
true_.r = true_data(:, 3);
true_.T = true_data(:, 4);
true_.corp = true_data(:, 5);
true_.sigma = true_data(:, 11);
true_.V = true_data(:, 12);

fprintf('  Initial params: %d rows\n', length(init.S0));
fprintf('  Calibrated params: %d rows\n', length(calib.S0));
fprintf('  True params: %d rows\n\n', length(true_.S0));

%% FILTER DATA
% Remove worthless contracts (V <= 0) to focus on meaningful pricing differences
% Note: sigma values at lower bound (0.001) are valid - they indicate near-zero option value
fprintf('Applying filters...\n');
fprintf('  Original rows: %d\n', length(true_.S0));

% Filter: Keep only options with meaningful value (V > 0)
% This removes deep OTM options that have negligible value
valid_filter = (true_.V > 0);
fprintf('  Valid rows (V > 0): %d\n', sum(valid_filter));
fprintf('  Filtered out: %d rows\n\n', sum(~valid_filter));

% COMPUTE RELATIVE ERRORS
fprintf('Computing relative errors...\n');

% Calculate relative errors ONLY for valid rows
% Add protection against division by zero
denominator = true_.sigma(valid_filter);
denominator(denominator == 0) = NaN;  % Mark zero denominators as NaN
rel_err_init = abs(init.sigma(valid_filter) - true_.sigma(valid_filter)) ./ denominator;
rel_err_calib = abs(calib.sigma(valid_filter) - true_.sigma(valid_filter)) ./ denominator;

% Remove inf values
rel_err_init(isinf(rel_err_init)) = NaN;
rel_err_calib(isinf(rel_err_calib)) = NaN;

% Extract time and moneyness for valid rows
T_days = calib.T(valid_filter) .* 252;
m = calib.m(valid_filter);

% Use mean with ignoring inf (since we set inf to NaN, we need a custom approach)
% For Octave compatibility, use mean of non-NaN values
valid_rel_err_init = rel_err_init(~isnan(rel_err_init));
valid_rel_err_calib = rel_err_calib(~isnan(rel_err_calib));

if length(valid_rel_err_init) > 0
    mean_init = mean(valid_rel_err_init);
else
    mean_init = NaN;
end

if length(valid_rel_err_calib) > 0
    mean_calib = mean(valid_rel_err_calib);
else
    mean_calib = NaN;
end

fprintf('  Mean relative error (Initial): %.4f (%.2f%%)\n', mean_init, mean_init*100);
fprintf('  Mean relative error (Calibrated): %.4f (%.2f%%)\n\n', mean_calib, mean_calib*100);

%% CATEGORIZE BY MATURITY AND MONEYNESS
fprintf('Categorizing by maturity and moneyness...\n');

% Maturity categories: 1: < 30 days, 2: 30~180 days, 3: > 180 days
maturity_idx = zeros(size(T_days));
maturity_idx(T_days < 30) = 1;
maturity_idx(T_days >= 30 & T_days < 180) = 2;
maturity_idx(T_days >= 180) = 3;

% Moneyness categories: 1: < 0.8, 2: 0.8~1.0, 3: 1.0~1.2, 4: > 1.2
moneyness_idx = zeros(size(m));
moneyness_idx(m < 0.8) = 1;
moneyness_idx(m >= 0.8 & m < 1.0) = 2;
moneyness_idx(m >= 1.0 & m < 1.2) = 3;
moneyness_idx(m >= 1.2) = 4;

% Compute mean relative error for each category
valid_idx = (maturity_idx > 0) & (moneyness_idx > 0);

mean_rel_err_matrix_init = accumarray([maturity_idx(valid_idx), moneyness_idx(valid_idx)], ...
                                       rel_err_init(valid_idx), ...
                                       [3, 4], @mean, NaN);

mean_rel_err_matrix_calib = accumarray([maturity_idx(valid_idx), moneyness_idx(valid_idx)], ...
                                        rel_err_calib(valid_idx), ...
                                        [3, 4], @mean, NaN);

%% CREATE DISPLAY
fprintf('Creating summary tables...\n\n');

% Define table labels
colNames = {'m_lt_0_8', 'm_0_8_to_1_0', 'm_1_0_to_1_2', 'm_gt_1_2'};
rowNames = {'Days_lt_30', 'Days_30_to_180', 'Days_gt_180'};

%% DISPLAY RESULTS
fprintf('=================================================================\n');
fprintf('   Mean Relative Error Table (Initial vs True)\n');
fprintf('=================================================================\n');

% Print header
fprintf('%-15s', 'Maturity');
for j = 1:length(colNames)
    fprintf('%15s', colNames{j});
end
fprintf('\n');

% Print rows for initial errors
for i = 1:3
    fprintf('%-15s', rowNames{i});
    for j = 1:4
        val = mean_rel_err_matrix_init(i, j);
        if isnan(val)
            fprintf('%15s', 'NaN');
        else
            fprintf('%15.6f', val);
        end
    end
    fprintf('\n');
end

fprintf('\n=================================================================\n');
fprintf('   Mean Relative Error Table (Calibrated vs True)\n');
fprintf('=================================================================\n');

% Print header
fprintf('%-15s', 'Maturity');
for j = 1:length(colNames)
    fprintf('%15s', colNames{j});
end
fprintf('\n');

% Print rows for calibrated errors
for i = 1:3
    fprintf('%-15s', rowNames{i});
    for j = 1:4
        val = mean_rel_err_matrix_calib(i, j);
        if isnan(val)
            fprintf('%15s', 'NaN');
        else
            fprintf('%15.6f', val);
        end
    end
    fprintf('\n');
end

%% COMPUTE IMPROVEMENT
fprintf('\n=================================================================\n');
fprintf('   Improvement (Calibrated vs Initial)\n');
fprintf('=================================================================\n');

% Fix improvement calculation to handle NaN and avoid division by zero
improvement_matrix = zeros(size(mean_rel_err_matrix_init));
for i = 1:size(mean_rel_err_matrix_init, 1)
    for j = 1:size(mean_rel_err_matrix_init, 2)
        init_err = mean_rel_err_matrix_init(i, j);
        calib_err = mean_rel_err_matrix_calib(i, j);
        
        % Only compute if both values are valid
        if ~isnan(init_err) && ~isnan(calib_err) && init_err > 1e-10
            improvement_matrix(i, j) = (init_err - calib_err) / init_err * 100;
        else
            improvement_matrix(i, j) = NaN;
        end
    end
end

fprintf('\n=================================================================\n');
fprintf('   Improvement (Calibrated vs Initial)\n');
fprintf('=================================================================\n');

% Print header
fprintf('%-15s', 'Maturity');
for j = 1:length(colNames)
    fprintf('%15s', colNames{j});
end
fprintf('\n');

% Print rows for improvements
for i = 1:3
    fprintf('%-15s', rowNames{i});
    for j = 1:4
        val = improvement_matrix(i, j);
        if isnan(val)
            fprintf('%15s', 'NaN');
        else
            fprintf('%15.2f', val);
        end
    end
    fprintf('\n');
end
fprintf('(Positive values indicate improvement with calibration)\n\n');

%% SAVE RESULTS
if do_save
    output_file = fullfile(folder, 'rel_err_results.csv');
    fprintf('Saving results to: %s\n', output_file);
    
    % Build output matrix with results
    % Rows: 3 maturity categories + overall
    % Cols: Category name, then for each moneyness: Init, Calib, Improvement
    
    % Prepare data
    results_matrix = [];
    for i = 1:3  % 3 maturity categories
        row = [];
        for j = 1:4  % 4 moneyness categories
            init_err = mean_rel_err_matrix_init(i, j);
            calib_err = mean_rel_err_matrix_calib(i, j);
            impr = improvement_matrix(i, j);
            row = [row, init_err, calib_err, impr];
        end
        results_matrix = [results_matrix; row];
    end
    
    % Add overall row
    overall_row = [];
    for j = 1:4
        % Octave-compatible mean of non-NaN values
        col_init = mean_rel_err_matrix_init(:, j);
        col_calib = mean_rel_err_matrix_calib(:, j);
        col_impr = improvement_matrix(:, j);
        
        init_err = mean(col_init(~isnan(col_init)));
        calib_err = mean(col_calib(~isnan(col_calib)));
        impr = mean(col_impr(~isnan(col_impr)));
        
        if isempty(init_err), init_err = NaN; end
        if isempty(calib_err), calib_err = NaN; end
        if isempty(impr), impr = NaN; end
        
        overall_row = [overall_row, init_err, calib_err, impr];
    end
    results_matrix = [results_matrix; overall_row];
    
    % Save as CSV
    csvwrite(output_file, results_matrix);
    fprintf('Results saved successfully!\n\n');
end

%% PLOT RESULTS
if do_plot
    fprintf('Generating plots...\n');
    
    figure('Position', [100, 100, 1200, 500]);
    
    % Plot 1: Initial vs True
    subplot(1, 3, 1);
    imagesc(mean_rel_err_matrix_init);
    colorbar;
    title('Initial vs True');
    xlabel('Moneyness Category');
    ylabel('Maturity Category');
    set(gca, 'XTick', 1:4, 'XTickLabel', {'<0.8', '0.8-1.0', '1.0-1.2', '>1.2'});
    set(gca, 'YTick', 1:3, 'YTickLabel', {'<30d', '30-180d', '>180d'});
    
    % Plot 2: Calibrated vs True
    subplot(1, 3, 2);
    imagesc(mean_rel_err_matrix_calib);
    colorbar;
    title('Calibrated vs True');
    xlabel('Moneyness Category');
    ylabel('Maturity Category');
    set(gca, 'XTick', 1:4, 'XTickLabel', {'<0.8', '0.8-1.0', '1.0-1.2', '>1.2'});
    set(gca, 'YTick', 1:3, 'YTickLabel', {'<30d', '30-180d', '>180d'});
    
    % Plot 3: Improvement
    subplot(1, 3, 3);
    imagesc(improvement_matrix);
    colorbar;
    title('Improvement (%)');
    xlabel('Moneyness Category');
    ylabel('Maturity Category');
    set(gca, 'XTick', 1:4, 'XTickLabel', {'<0.8', '0.8-1.0', '1.0-1.2', '>1.2'});
    set(gca, 'YTick', 1:3, 'YTickLabel', {'<30d', '30-180d', '>180d'});
    
    % Save plot
    plot_file = fullfile(folder, 'rel_err_heatmaps.png');
    saveas(gcf, plot_file);
    fprintf('Plot saved: %s\n', plot_file);
end

%% SUMMARY STATISTICS
fprintf('\n=================================================================\n');
fprintf('                    SUMMARY STATISTICS\n');
fprintf('=================================================================\n');
fprintf('Overall mean relative error:\n');
fprintf('  Initial:    %.4f (%.2f%%)\n', mean_init, mean_init*100);
fprintf('  Calibrated: %.4f (%.2f%%)\n', mean_calib, mean_calib*100);

if mean_init > 0
    improvement_pct = (mean_init - mean_calib) / mean_init * 100;
else
    improvement_pct = NaN;
end
fprintf('  Improvement: %.2f%%\n', improvement_pct);
fprintf('\n');

fprintf('Median relative error:\n');
med_init = median(rel_err_init(~isnan(rel_err_init)));
med_calib = median(rel_err_calib(~isnan(rel_err_calib)));
fprintf('  Initial:    %.4f (%.2f%%)\n', med_init, med_init*100);
fprintf('  Calibrated: %.4f (%.2f%%)\n', med_calib, med_calib*100);
fprintf('=================================================================\n');

end
