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
addParameter(p, 'folder', 'compare', @ischar);
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
init = readtable(init_file);
calib = readtable(calib_file);
true_ = readtable(true_file);

fprintf('  Initial params: %d rows\n', height(init));
fprintf('  Calibrated params: %d rows\n', height(calib));
fprintf('  True params: %d rows\n\n', height(true_));

%% FILTER DATA
% Remove outliers: super small sigma, V < 0.5
fprintf('Applying filters...\n');
fprintf('  Original rows: %d\n', height(true_));

valid_filter = (true_.sigma > 0.01) & (true_.V >= 0.5);
fprintf('  Valid rows (sigma > 0.01 & V >= 0.5): %d\n', sum(valid_filter));
fprintf('  Filtered out: %d rows\n\n', sum(~valid_filter));

%% COMPUTE RELATIVE ERRORS
fprintf('Computing relative errors...\n');

% Calculate relative errors ONLY for valid rows
rel_err_init = abs(init.sigma(valid_filter) - true_.sigma(valid_filter)) ./ true_.sigma(valid_filter);
rel_err_calib = abs(calib.sigma(valid_filter) - true_.sigma(valid_filter)) ./ true_.sigma(valid_filter);

% Extract time and moneyness for valid rows
T_days = calib.T(valid_filter) .* 252;
m = calib.m(valid_filter);

fprintf('  Mean relative error (Initial): %.4f (%.2f%%)\n', mean(rel_err_init), mean(rel_err_init)*100);
fprintf('  Mean relative error (Calibrated): %.4f (%.2f%%)\n\n', mean(rel_err_calib), mean(rel_err_calib)*100);

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

%% CREATE TABLES
fprintf('Creating summary tables...\n\n');

% Define table labels
colNames = {'m_lt_0_8', 'm_0_8_to_1_0', 'm_1_0_to_1_2', 'm_gt_1_2'};
rowNames = {'Days_lt_30', 'Days_30_to_180', 'Days_gt_180'};

% Convert to tables
table_init = array2table(mean_rel_err_matrix_init, ...
    'VariableNames', colNames, ...
    'RowNames', rowNames);

table_calib = array2table(mean_rel_err_matrix_calib, ...
    'VariableNames', colNames, ...
    'RowNames', rowNames);

%% DISPLAY RESULTS
fprintf('=================================================================\n');
fprintf('   Mean Relative Error Table (Initial vs True)\n');
fprintf('=================================================================\n');
disp(table_init);

fprintf('\n=================================================================\n');
fprintf('   Mean Relative Error Table (Calibrated vs True)\n');
fprintf('=================================================================\n');
disp(table_calib);

%% COMPUTE IMPROVEMENT
fprintf('\n=================================================================\n');
fprintf('   Improvement (Calibrated vs Initial)\n');
fprintf('=================================================================\n');

improvement_matrix = (mean_rel_err_matrix_init - mean_rel_err_matrix_calib) ./ mean_rel_err_matrix_init * 100;
table_improvement = array2table(improvement_matrix, ...
    'VariableNames', colNames, ...
    'RowNames', rowNames);

disp(table_improvement);
fprintf('(Positive values indicate improvement with calibration)\n\n');

%% SAVE RESULTS
if do_save
    output_file = fullfile(folder, 'rel_err_results.csv');
    fprintf('Saving results to: %s\n', output_file);
    
    % Combine tables
    combined_table = table();
    combined_table.Category = [rowNames'; {'Overall'}];
    
    for i = 1:length(colNames)
        col = colNames{i};
        combined_table.([col '_Init']) = [mean_rel_err_matrix_init(:,i); mean(mean_rel_err_matrix_init(:,i), 'omitnan')];
        combined_table.([col '_Calib']) = [mean_rel_err_matrix_calib(:,i); mean(mean_rel_err_matrix_calib(:,i), 'omitnan')];
        combined_table.([col '_Improvement']) = [improvement_matrix(:,i); mean(improvement_matrix(:,i), 'omitnan')];
    end
    
    writetable(combined_table, output_file);
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
fprintf('  Initial:    %.4f (%.2f%%)\n', mean(rel_err_init, 'omitnan'), mean(rel_err_init, 'omitnan')*100);
fprintf('  Calibrated: %.4f (%.2f%%)\n', mean(rel_err_calib, 'omitnan'), mean(rel_err_calib, 'omitnan')*100);
fprintf('  Improvement: %.2f%%\n', (mean(rel_err_init, 'omitnan') - mean(rel_err_calib, 'omitnan')) / mean(rel_err_init, 'omitnan') * 100);
fprintf('\n');

fprintf('Median relative error:\n');
fprintf('  Initial:    %.4f (%.2f%%)\n', median(rel_err_init, 'omitnan'), median(rel_err_init, 'omitnan')*100);
fprintf('  Calibrated: %.4f (%.2f%%)\n', median(rel_err_calib, 'omitnan'), median(rel_err_calib, 'omitnan')*100);
fprintf('=================================================================\n');

end
