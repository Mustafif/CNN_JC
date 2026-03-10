clear all; clc;

folder = "set4/";

init = readtable(folder + "initial_params.csv");
calib = readtable(folder + "calibrated_params.csv");
true_ = readtable(folder + "true_params.csv");

% ---------------------------------------------------------
% APPLY FILTERS: Remove outliers (super small sigma, V < 0.5)
% ---------------------------------------------------------
% We keep rows where true sigma is > 0.01 and true Option Price (V) >= 0.5
valid_filter = (true_.sigma > 0.01) & (true_.V >= 0.5);

% Calculate relative errors ONLY for valid rows
rel_err_init = abs(init.sigma(valid_filter) - true_.sigma(valid_filter)) ./ true_.sigma(valid_filter);
rel_err_calib = abs(calib.sigma(valid_filter) - true_.sigma(valid_filter)) ./ true_.sigma(valid_filter);

% Use only filtered Time and Moneyness
T_days = calib.T(valid_filter) .* 252;
m = calib.m(valid_filter);

% Assign categories for Maturity
% 1: < 30 days, 2: 30~180 days, 3: > 180 days
maturity_idx = zeros(size(T_days));
maturity_idx(T_days < 30) = 1;
maturity_idx(T_days >= 30 & T_days < 180) = 2;
maturity_idx(T_days >= 180) = 3;

% Assign categories for Moneyness
% 1: < 0.8, 2: 0.8~1.0, 3: 1.0~1.2, 4: > 1.2
moneyness_idx = zeros(size(m));
moneyness_idx(m < 0.8) = 1;
moneyness_idx(m >= 0.8 & m < 1.0) = 2;
moneyness_idx(m >= 1.0 & m < 1.2) = 3;
moneyness_idx(m >= 1.2) = 4;

% Compute the mean relative error for each category using accumarray
% We build a 3x4 matrix (3 maturity groups, 4 moneyness groups)
% Any empty buckets will be filled with NaN
valid_idx = (maturity_idx > 0) & (moneyness_idx > 0);

mean_rel_err_matrix_init = accumarray([maturity_idx(valid_idx), moneyness_idx(valid_idx)], ...
                                 rel_err_init(valid_idx), ...
                                 [3, 4], @mean, NaN);

mean_rel_err_matrix_calib = accumarray([maturity_idx(valid_idx), moneyness_idx(valid_idx)], ...
                                 rel_err_calib(valid_idx), ...
                                 [3, 4], @mean, NaN);

% Define valid MATLAB variable names for columns and rows
colNames = {'m_lt_0_8', 'm_0_8_to_1_0', 'm_1_0_to_1_2', 'm_gt_1_2'};
rowNames = {'Days_lt_30', 'Days_30_to_180', 'Days_gt_180'};

% Convert matrices to tables
table_init = array2table(mean_rel_err_matrix_init, ...
    'VariableNames', colNames, ...
    'RowNames', rowNames);

table_calib = array2table(mean_rel_err_matrix_calib, ...
    'VariableNames', colNames, ...
    'RowNames', rowNames);

% Display the tables
disp('======================================================');
disp('   Mean Relative Error Table (Initial vs True)        ');
disp('======================================================');
disp(table_init);

disp('======================================================');
disp('   Mean Relative Error Table (Calibrated vs True)     ');
disp('======================================================');
disp(table_calib);