% EXAMPLE_COMPARE_USAGE.m
% Example usage of the centralized parameter comparison system

clear all; clc;

fprintf('=================================================================\n');
fprintf('    PARAMETER COMPARISON SYSTEM - USAGE EXAMPLES\n');
fprintf('=================================================================\n\n');

%% EXAMPLE 1: Basic HN model comparison (no noise)
fprintf('EXAMPLE 1: Heston-Nandi model, no noise\n');
fprintf('---------------------------------------------------------\n');
fprintf('Running: compare_dataset(''compare_config.toml'')\n\n');

% Uncomment to run:
% compare_dataset('compare_config.toml');
% rel_err_results();

fprintf('This will generate:\n');
fprintf('  - compare/true_params.csv\n');
fprintf('  - compare/initial_params.csv\n');
fprintf('  - compare/calibrated_params.csv\n');
fprintf('  - compare/rel_err_results.csv\n\n');

%% EXAMPLE 2: Duan model with noise
fprintf('EXAMPLE 2: Duan GARCH model, 3%% noise\n');
fprintf('---------------------------------------------------------\n');
fprintf('Running: compare_dataset(''compare_config_duan_example.toml'')\n\n');

% Uncomment to run:
% compare_dataset('compare_config_duan_example.toml');
% rel_err_results('folder', 'compare_duan_3pct_noise', 'plot', true);

fprintf('This will generate:\n');
fprintf('  - compare_duan_3pct_noise/true_params.csv\n');
fprintf('  - compare_duan_3pct_noise/initial_params.csv\n');
fprintf('  - compare_duan_3pct_noise/calibrated_params.csv\n');
fprintf('  - compare_duan_3pct_noise/rel_err_results.csv\n');
fprintf('  - compare_duan_3pct_noise/rel_err_heatmaps.png\n\n');

%% EXAMPLE 3: Custom configuration
fprintf('EXAMPLE 3: Creating a custom configuration\n');
fprintf('---------------------------------------------------------\n\n');

fprintf('Create a new TOML file with your parameters:\n\n');
fprintf('1. Copy compare_config.toml to my_config.toml\n');
fprintf('2. Edit the parameters:\n');
fprintf('   - Set model type (HN, Duan, or GJR)\n');
fprintf('   - Set true, initial, and calibrated parameters\n');
fprintf('   - Configure noise level (0.0 = no noise, 0.05 = 5%% noise)\n');
fprintf('   - Adjust scaling parameters (contracts_per_day, num_days)\n\n');
fprintf('3. Run:\n');
fprintf('   compare_dataset(''my_config.toml'')\n');
fprintf('   rel_err_results(''folder'', ''compare'')\n\n');

%% EXAMPLE 4: Analysis only (if data already exists)
fprintf('EXAMPLE 4: Analyze existing comparison data\n');
fprintf('---------------------------------------------------------\n\n');

fprintf('If you already have the CSV files, just run the analysis:\n\n');
fprintf('  rel_err_results()\n');
fprintf('  rel_err_results(''plot'', true)\n');
fprintf('  rel_err_results(''folder'', ''custom_folder'', ''plot'', true)\n\n');

%% COMPARING DIFFERENT NOISE LEVELS
fprintf('EXAMPLE 5: Compare effect of different noise levels\n');
fprintf('---------------------------------------------------------\n\n');

fprintf('Create multiple configs with different noise levels:\n');
fprintf('  - compare_config_noise_0pct.toml   (noise.level = 0.0)\n');
fprintf('  - compare_config_noise_1pct.toml   (noise.level = 0.01)\n');
fprintf('  - compare_config_noise_5pct.toml   (noise.level = 0.05)\n\n');

fprintf('Run each one and compare results:\n\n');
fprintf('  %% No noise\n');
fprintf('  compare_dataset(''compare_config_noise_0pct.toml'')\n');
fprintf('  rel_err_results(''folder'', ''compare_0pct'', ''plot'', true)\n\n');

fprintf('  %% 1%% noise\n');
fprintf('  compare_dataset(''compare_config_noise_1pct.toml'')\n');
fprintf('  rel_err_results(''folder'', ''compare_1pct'', ''plot'', true)\n\n');

fprintf('  %% 5%% noise\n');
fprintf('  compare_dataset(''compare_config_noise_5pct.toml'')\n');
fprintf('  rel_err_results(''folder'', ''compare_5pct'', ''plot'', true)\n\n');

%% NOTES
fprintf('=================================================================\n');
fprintf('NOTES:\n');
fprintf('=================================================================\n');
fprintf('1. All parameter sets use the SAME randomness for fair comparison\n');
fprintf('2. Noise is added AFTER pricing (if configured)\n');
fprintf('3. Results are filtered to remove outliers (sigma<0.01, V<0.5)\n');
fprintf('4. See COMPARE_README.md for detailed documentation\n');
fprintf('5. Uncomment the actual function calls above to run examples\n');
fprintf('=================================================================\n');
